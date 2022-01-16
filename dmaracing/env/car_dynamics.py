import torch   
from typing import Dict, List, Tuple
import numpy as np

from dmaracing.env.car_dynamics_utils import resolve_collsions

@torch.jit.script
def step_cars(state : torch.Tensor, 
              actions : torch.Tensor,
              wheel_locations: torch.Tensor,
              R: torch.Tensor, 
              contact_wrenches : torch.Tensor, 
              mod_par : Dict[str, float],  
              sim_par : Dict[str, float], 
              vn : Dict[str, int],
              collision_pairs : List[List[int]],
              collision_verts : torch.Tensor,
              P_tot : torch.Tensor,
              D_tot : torch.Tensor,
              S_mat : torch.Tensor,
              Repf_mat : torch.Tensor,
              Ds : List[int],
              num_envs : int,
              zero_pad : torch.Tensor,
              collide: int,
              wheels_on_track_segments : torch.Tensor,
              active_track_mask: torch.Tensor,
              A_track: torch.Tensor,
              b_track: torch.Tensor,
              S_track: torch.Tensor 
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    #set steering angle
    #dir = torch.sign(actions[:, :, vn['A_STEER']] - state[:, :, vn['S_STEER']])
    #val = torch.abs(actions[:, :, vn['A_STEER']] - state[:, :, vn['S_STEER']])

    theta = state[:, :, vn['S_THETA']]
    delta = state[:, :, vn['S_STEER']]

    #state[:, :, vn['S_STEER']] += sim_par['dt']*dir * torch.min(50.0 * val, 3.0 +  0*val)
    #state[:, :, vn['S_STEER']] = torch.clamp(state[:, :, vn['S_STEER']], -np.pi/4, np.pi/4)
    state[:, :, vn['S_STEER']] = torch.clamp(actions[:, :, vn['A_STEER']], -np.pi/4, np.pi/4)
    
    #compute wheel forward and side directions, plus locations in the global frame
    dir_fwd_ft = torch.cat((torch.cos(theta + delta).unsqueeze(2), torch.sin(theta + delta).unsqueeze(2)), dim = 2)
    dir_fwd_bk = torch.cat((torch.cos(theta).unsqueeze(2), torch.sin(theta).unsqueeze(2)), dim = 2)

    #(numenv, num_ag, 2)
    wheel_dirs_forward = torch.cat((dir_fwd_ft.unsqueeze(2),
                                    dir_fwd_ft.unsqueeze(2),
                                    dir_fwd_bk.unsqueeze(2),
                                    dir_fwd_bk.unsqueeze(2)), dim = 2)

    dir_sid_ft = torch.cat((-torch.sin(theta + delta).unsqueeze(2), torch.cos(theta + delta).unsqueeze(2)), dim = 2)
    dir_sid_bk = torch.cat((-torch.sin(theta).unsqueeze(2), torch.cos(theta).unsqueeze(2)), dim = 2)

    wheel_dirs_side =  torch.cat((dir_sid_ft.unsqueeze(2),
                                  dir_sid_ft.unsqueeze(2),
                                  dir_sid_bk.unsqueeze(2),
                                  dir_sid_bk.unsqueeze(2)), dim = 2)
    

    R[:, :, 0, 0 ] = torch.cos(theta)
    R[:, :, 0, 1 ] = -torch.sin(theta)
    R[:, :, 1, 0 ] = torch.sin(theta)
    R[:, :, 1, 1 ] = torch.cos(theta)
    #(num_envs, num_agents, 2, 2) x (num_envs, num_agents, num_wheels, 2)
    wheel_locations_bodycentric_world = torch.einsum('klij, dj->kldi', R, wheel_locations)
    wheel_locations_world = wheel_locations_bodycentric_world + torch.tile(state[:,:, vn['S_X']:vn['S_Y']+1].unsqueeze(2), (1,1,4,1))
    wheel_locations_bodycentric_world = torch.nn.functional.pad(wheel_locations_bodycentric_world, (0,1))
   
    #set gas 
    diff = actions[:, :, vn['A_GAS']] - state[:, :, vn['S_GAS']] 
    state[:, :, vn['S_GAS']] += torch.clamp(diff, max=0.1)
    state[:, :, vn['S_GAS']] = torch.clamp(state[:, :, vn['S_GAS']], 0, 1)
    
    #set wheel speeds
    num = sim_par['dt'] * mod_par['ENGINE_POWER']*state[:, :, vn['S_GAS']]
    den = mod_par['WHEEL_MOMENT_OF_INERTIA']*(torch.abs(state[:, :, vn['S_W2']:vn['S_W3']+1]) + 5)
    state[:, :, vn['S_W2']:vn['S_W3']+1] += torch.div( num.unsqueeze(2),den)

    #set brake
    #break >0.9 -> lock up wheels
    dir =  -torch.sign(state[:, :, vn['S_W0']:vn['S_W3']+1])
    val = mod_par['BREAKFORCE'] * actions[:, :, vn['A_BRAKE']].unsqueeze(2)
    need_clip = torch.abs(val)>torch.abs(state[:, :, vn['S_W0']:vn['S_W3']+1])
    val = need_clip * torch.abs(state[:, :, vn['S_W0']:vn['S_W3']+1]) + ~need_clip*val
    state[:, :, vn['S_W0']:vn['S_W3']+1] += dir*val
    state[:, :, vn['S_W0']:vn['S_W3']+1] = (0.9 >= actions[:, :, vn['A_BRAKE']]).unsqueeze(2) * state[:, :, vn['S_W0']:vn['S_W3']+1]
    
    vr = state[:, :, vn['S_W0']:vn['S_W3']+1] * mod_par['WHEEL_R']
    omega_body  = torch.nn.functional.pad(state[:, :, vn['S_DTHETA']].unsqueeze(2), (2, 0))
    
    wheel_vels_fl = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_bodycentric_world[:,:, 0, :],  omega_body)[:, :, 0:2]
    wheel_vels_fr = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_bodycentric_world[:,:, 1, :],  omega_body)[:, :, 0:2]
    wheel_vels_br = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_bodycentric_world[:,:, 2, :],  omega_body)[:, :, 0:2]
    wheel_vels_bl = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_bodycentric_world[:,:, 3, :],  omega_body)[:, :, 0:2] 
    wheel_vels = torch.cat((wheel_vels_fl.unsqueeze(2),
                            wheel_vels_fr.unsqueeze(2),
                            wheel_vels_br.unsqueeze(2),
                            wheel_vels_bl.unsqueeze(2)), dim = 2)
    
    #wheel vels (num_env, num_agnt, 4, 2) wheel_dir (num_env, num_agnt, 4, 2) -> wheel force proj (num_env, num_agnt, 4, )
    vf = torch.einsum('ijkl, ijkl -> ijk', wheel_vels, wheel_dirs_forward)                        
    vs = torch.einsum('ijkl, ijkl -> ijk', wheel_vels, wheel_dirs_side)                        
    f_force = -vf + vr
    p_force = -vs*15
    f_force *= 205000 *mod_par['SIZE']**2
    p_force *= 205000 *mod_par['SIZE']**2
    

    #check which tires are on track
    # Multi track A_track [ntracks, polygon = 4*300, coords = 2]
    # single track A_track [polygon = 4*300, coords = 2]

    wheels_on_track_segments_concat = 1.0 * (torch.einsum('es, stc, eawc  -> eawt', active_track_mask, A_track, wheel_locations_world)\
                                             - torch.einsum('es, st -> et', active_track_mask, b_track).view(num_envs,1,1,-1) +0.1 >= 0 )
    wheels_on_track_segments[:] = torch.einsum('jt, eawt -> eawj', S_track, wheels_on_track_segments_concat) >= 3.5
    wheel_on_track = torch.any(wheels_on_track_segments, dim = 3)

    f_tot = torch.sqrt(torch.square(f_force) +torch.square(p_force)) + 1e-9
    f_lim = ((1-mod_par['OFFTRACK_FRICTION_SCALE'])*mod_par['FRICTION_LIMIT'])*wheel_on_track + mod_par['OFFTRACK_FRICTION_SCALE']*mod_par['FRICTION_LIMIT']
    slip = f_tot > f_lim
    f_force = slip * (0.9*f_lim * torch.div(f_force, f_tot)) + ~slip * f_force
    p_force = slip * (0.9*f_lim * torch.div(p_force, f_tot)) + ~slip * p_force

    state[:, :, vn['S_W0']:vn['S_W3']+1] -= sim_par['dt']*mod_par['WHEEL_R']/mod_par['WHEEL_MOMENT_OF_INERTIA'] * f_force

    #apply force to center
    wheel_forces = f_force.unsqueeze(3)*wheel_dirs_forward + p_force.unsqueeze(3)*wheel_dirs_side
    net_force = torch.sum(wheel_forces, dim = 2)
    net_torque = torch.sum(torch.cross(wheel_locations_bodycentric_world, torch.nn.functional.pad(wheel_forces, (0,1)), dim = 3)[..., 2], dim = 2)

    #net_torque
    ddx = 1/mod_par['M']*(net_force[:,:,0] + contact_wrenches[:,:,0])
    ddy = 1/mod_par['M']*(net_force[:,:,1] + contact_wrenches[:,:,1])
    ddtheta = 1/mod_par['I']*(net_torque + contact_wrenches[:,:,2])
    acc = torch.cat((ddx.unsqueeze(2),ddy.unsqueeze(2),ddtheta.unsqueeze(2)), dim= 2)
    state[:, :, vn['S_X']:vn['S_THETA'] + 1] += sim_par['dt']*state[:,:,vn['S_DX']:vn['S_DTHETA']+1]
    state[:, :, vn['S_DX']:vn['S_DTHETA'] + 1] += sim_par['dt']*acc

    if collide:
        contact_wrenches = resolve_collsions(contact_wrenches,
                                             state,
                                             collision_pairs,
                                             mod_par['lf'],
                                             collision_verts,
                                             R[:,0,:,:],
                                             P_tot,
                                             D_tot,
                                             S_mat,
                                             Repf_mat,
                                             Ds,
                                             num_envs,
                                             zero_pad)

    return state, contact_wrenches, wheels_on_track_segments