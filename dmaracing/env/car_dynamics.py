import torch   
from typing import Dict, List
import numpy as np

from dmaracing.env.car_dynamics_utils import get_contact_wrenches, rotate_vec, transform_col_verts


def get_varnames()->Dict[str, int]:
    varnames = {}
    varnames['S_X'] = 0
    varnames['S_Y'] = 1
    varnames['S_THETA'] = 2
    varnames['S_DX'] = 3
    varnames['S_DY'] = 4
    varnames['S_DTHETA'] = 5
    varnames['S_W0'] = 6
    varnames['S_W1'] = 7
    varnames['S_W2'] = 8
    varnames['S_W3'] = 9
    varnames['S_STEER'] = 10
    varnames['S_GAS'] = 11
    
    varnames['A_STEER'] = 0
    varnames['A_GAS'] = 1
    varnames['A_BREAK'] = 2
    return varnames

@torch.jit.script
def resolve_collsions(contact_wrenches : torch.Tensor,
                      states : torch.Tensor,
                      collision_pairs : List[List[int]],
                      lf : float,
                      collision_verts : torch.Tensor,
                      R : torch.Tensor,
                      P_tot : torch.Tensor,
                      D_tot : torch.Tensor,
                      S_mat : torch.Tensor,
                      Repf_mat : torch.Tensor,
                      Ds : List[int],
                      num_envs : int,
                      zero_pad : torch.Tensor
                      ) -> torch.Tensor:

    contact_wrenches[:,:,:] =0.0

    if len(collision_pairs):
        for colp in collision_pairs:

            idx_comp = torch.where(torch.norm(states[:, colp[0], 0:2] -  states[:, colp[1], 0:2], dim =1)<=2.8*lf)[0]
            
            if  len(idx_comp):
                states_A = states[idx_comp, colp[0], 0:3]
                states_B = states[idx_comp, colp[1], 0:3]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] -states_A[:,:2]
                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], R[idx_comp,:,:])

                force_B, torque_A, torque_B = get_contact_wrenches(P_tot, 
                                                                   D_tot, 
                                                                   S_mat, 
                                                                   Repf_mat,
                                                                   Ds,
                                                                   verts_tf,
                                                                   num_envs,
                                                                   zero_pad
                                                                   )

                #rotate forces into global frame from frame A
                force_B = rotate_vec(force_B, states_A[:, 2], R[idx_comp,:,:])
                contact_wrenches[ idx_comp, colp[0], :2] += -force_B
                contact_wrenches[ idx_comp, colp[1], :2] += force_B
                contact_wrenches[ idx_comp, colp[0], 2] += torque_A
                contact_wrenches[ idx_comp, colp[1], 2] += torque_B

                #flip and check other direction
                states_A = states[idx_comp, colp[1], 0:3]
                states_B = states[idx_comp, colp[0], 0:3]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] - states_A[:,:2]
                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], R[idx_comp,:,:])

                force_B, torque_A, torque_B = get_contact_wrenches(P_tot, 
                                                                   D_tot, 
                                                                   S_mat, 
                                                                   Repf_mat,
                                                                   Ds,
                                                                   verts_tf,
                                                                   num_envs,
                                                                   zero_pad
                                                                   )

                #rotate forces into global frame from frame A
                force_B = rotate_vec(force_B, states_A[:, 2], R[idx_comp,:,:])
                contact_wrenches[ idx_comp, colp[1], :2] += -force_B
                contact_wrenches[ idx_comp, colp[0], :2] += force_B
                contact_wrenches[ idx_comp, colp[1], 2] += torque_A
                contact_wrenches[ idx_comp, colp[0], 2] += torque_B
    return contact_wrenches


def set_dependent_params(mod_par):
    SIZE = mod_par['SIZE']
    mod_par['ENGINE_POWER'] = 100000000*SIZE**2
    mod_par['WHEEL_MOMENT_OF_INERTIA'] = 4000*SIZE**2
    mod_par['FRICTION_LIMIT'] = 1000000 * SIZE * SIZE
    mod_par['WHEEL_R'] = SIZE*27
    L = 160.0 *SIZE
    W = L/2
    M = L*W
    mod_par['M'] = L
    mod_par['L'] = L
    mod_par['W'] = W 
    mod_par['I'] = M*(L**2 + W**2 )/12.0
    mod_par['lf'] = L/2
    mod_par['lr'] = L/2
    
@torch.jit.script
def step_cars(state : torch.Tensor, 
              actions : torch.Tensor,
              wheel_locations: torch.Tensor,
              R: torch.Tensor, 
              col_wrenches : torch.Tensor, 
              mod_par : Dict[str, float],  
              sim_par : Dict[str, float], 
              vn : Dict[str, int]) -> torch.Tensor:
    

    #set steering angle
    dir = torch.sign(actions[:, :, vn['A_STEER']] - state[:, :, vn['S_STEER']])
    val = torch.abs(actions[:, :, vn['A_STEER']] - state[:, :, vn['S_STEER']])

    theta = state[:, :, vn['S_THETA']]
    delta = state[:, :, vn['S_STEER']]

    state[:, :, vn['S_STEER']] += sim_par['dt']*dir * torch.min(50.0 * val, 3.0 +  0*val)
    torch.clamp(state[:, :, vn['S_STEER']], -np.pi/3, np.pi/3)
    
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
    wheel_locations_world = torch.nn.functional.pad(torch.einsum('klij, dj->kldi', R, wheel_locations), (0,1))

   
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
    val = mod_par['BREAKFORCE'] * actions[:, :, vn['A_BREAK']].unsqueeze(2)
    need_clip = torch.abs(val)>torch.abs(state[:, :, vn['S_W0']:vn['S_W3']+1])
    val = need_clip * torch.abs(state[:, :, vn['S_W0']:vn['S_W3']+1]) + ~need_clip*val
    state[:, :, vn['S_W0']:vn['S_W3']+1] += dir*val
    state[:, :, vn['S_W0']:vn['S_W3']+1] = (0.9 >= actions[:, :, vn['A_BREAK']]).unsqueeze(2) * state[:, :, vn['S_W0']:vn['S_W3']+1]
    
    vr = state[:, :, vn['S_W0']:vn['S_W3']+1] * mod_par['WHEEL_R']
    omega_body  = torch.nn.functional.pad(state[:, :, vn['S_DTHETA']].unsqueeze(2), (2, 0))
    
    wheel_vels_fl = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_world[:,:, 0, :],  omega_body)[:, :, 0:2]
    wheel_vels_fr = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_world[:,:, 1, :],  omega_body)[:, :, 0:2]
    wheel_vels_br = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_world[:,:, 2, :],  omega_body)[:, :, 0:2]
    wheel_vels_bl = state[:, :, vn['S_DX']:vn['S_DY']+1] - torch.cross(wheel_locations_world[:,:, 3, :],  omega_body)[:, :, 0:2] 
    wheel_vels = torch.cat((wheel_vels_fl.unsqueeze(2),
                            wheel_vels_fr.unsqueeze(2),
                            wheel_vels_br.unsqueeze(2),
                            wheel_vels_bl.unsqueeze(2)), dim = 2)
    
    #wheel vels (num_env, num_agnt, 4, 2) wheel_dir (num_env, num_agnt, 4, 2) -> wheel force proj (num_env, num_agnt, 4, )
    vf = torch.einsum('ijkl, ijkl -> ijk', wheel_vels, wheel_dirs_forward)                        
    vs = torch.einsum('ijkl, ijkl -> ijk', wheel_vels, wheel_dirs_side)                        
    f_force = -vf + vr
    p_force = -vs
    f_force *= 205000 *mod_par['SIZE']**2
    p_force *= 205000 *mod_par['SIZE']**2

    f_tot = torch.sqrt(torch.square(f_force) +torch.square(p_force)) + 1e-9
    f_lim = mod_par['FRICTION_LIMIT']
    slip = f_tot > f_lim
    f_force = slip * (f_lim * torch.div(f_force, f_tot)) + ~slip * f_force
    p_force = slip * (f_lim * torch.div(p_force, f_tot)) + ~slip * p_force

    state[:, :, vn['S_W0']:vn['S_W3']+1] -= sim_par['dt']*mod_par['WHEEL_R']/mod_par['WHEEL_MOMENT_OF_INERTIA'] * f_force

    #apply force to center
    wheel_forces = f_force.unsqueeze(3)*wheel_dirs_forward + p_force.unsqueeze(3)*wheel_dirs_side
    net_force = torch.sum(wheel_forces, dim = 2)
    net_torque = torch.sum(torch.cross(wheel_locations_world, torch.nn.functional.pad(wheel_forces, (0,1)), dim = 3)[..., 2], dim = 2)

    #net_torque
    ddx = 1/mod_par['M']*(net_force[:,:,0] + col_wrenches[:,:,0])
    ddy = 1/mod_par['M']*(net_force[:,:,1] + col_wrenches[:,:,1])
    ddtheta = 1/mod_par['I']*(net_torque + col_wrenches[:,:,2])
    acc = torch.cat((ddx.unsqueeze(2),ddy.unsqueeze(2),ddtheta.unsqueeze(2)), dim= 2)
    state[:, :, vn['S_X']:vn['S_THETA'] + 1] += sim_par['dt']*state[:,:,vn['S_DX']:vn['S_DTHETA']+1]
    state[:, :, vn['S_DX']:vn['S_DTHETA'] + 1] += sim_par['dt']*acc
    return state