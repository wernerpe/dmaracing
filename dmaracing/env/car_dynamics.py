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
    varnames['A_STEER'] = 10
    varnames['S_GAS'] = 11
    varnames['S_BREAK'] = 12 

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

@torch.jit.script
def state_derivative(state : torch.Tensor, actions : torch.Tensor, col_wrenches : torch.Tensor, par : Dict[str, float], num_agents : int, vn : Dict[str, int])-> torch.Tensor:
    #model used from here, can change this later... 
    # https://onlinelibrary.wiley.com/doi/pdf/10.1002/oca.2123
    # 
    # [x, y, theta, xdot, ydot, thetadot, delta]
    #blah wip
    
    alphaf = -torch.atan2(state[:, :, vn['S_DTHETA']]*par['lf'] + state[:, :, vn['S_DY']], state[:, :, vn['S_DX']] ) + state[:, :, vn['S_DELTA']]
    alphar = torch.atan2(state[:, :, vn['S_DTHETA']]*par['lr'] - state[:, :, vn['S_DY']], state[:, :, vn['S_DX']])
    Ffy = par['Df'] * torch.sin(par['Cf']*torch.atan(par['Bf']*alphaf)) 
    Fry = par['Dr'] * torch.sin(par['Cr']*torch.atan(par['Br']*alphar))  
    Frx = actions[:, :, vn['A_ACC']]

    dx = state[:, :, vn['S_DX']] * torch.cos(state[:, :, vn['S_THETA']]) - state[:, :, vn['S_DY']] * torch.sin(state[:, :, vn['S_THETA']]) 
    dy = state[:, :, vn['S_DX']] * torch.sin(state[:, :, vn['S_THETA']]) + state[:, :, vn['S_DY']] * torch.cos(state[:, :, vn['S_THETA']])
    dtheta = state[:, :, vn['S_DTHETA']]
    ddx = 1/par['m']*( Frx - Ffy*torch.sin(state[:, :, vn['S_DELTA']]) + par['m']*state[:, :, vn['S_DY']]*state[:, :, vn['S_DTHETA']]  \
                     +   torch.cos(state[:, :, vn['S_THETA']]) * col_wrenches[:, :, 0] + torch.sin(state[:, :, vn['S_THETA']])* col_wrenches[:, :, 1]  \
                     - 0.1  * (state[:, :, vn['S_DX']]>0) +  0.1  * (state[:, :, vn['S_DX']]<0 ))

    ddy = 1/par['m']*( Fry + Ffy*torch.cos(state[:, :, vn['S_DELTA']]) - par['m']*state[:, :, vn['S_DX']]*state[:, :, vn['S_DTHETA']]  \
                     - torch.sin(state[:, :, vn['S_THETA']]) * col_wrenches[:, :, 0] + torch.cos(state[:, :, vn['S_THETA']])* col_wrenches[:, :, 1] )

    ddtheta = 1/par['Iz']*(Ffy*par['lf']*torch.cos(state[:, :, vn['S_DELTA']]) + col_wrenches[:, :, 2])
    
    #box constraints on steering angle
    ddelta =  actions[:, :, vn['A_DDELTA']]

    dstate = torch.cat((dx.view(-1, num_agents, 1),
                        dy.view(-1, num_agents, 1),
                        dtheta.view(-1, num_agents, 1),
                        ddx.view(-1, num_agents, 1),
                        ddy.view(-1, num_agents, 1),
                        ddtheta.view(-1, num_agents, 1),
                        ddelta.view(-1, num_agents, 1)
                        ), dim=2)
    return dstate


"""
def step_cars(task, state : torch.Tensor, actions : torch.Tensor, col_wrenches : torch.Tensor, num_agents : int, mod_par : Dict[str, float],  sim_par : Dict[str, float], vn : Dict[str, int]) -> torch.Tensor:
    dstate = state_derivative(state, actions, col_wrenches, mod_par, num_agents , vn)
    
    #handle constraints
    is_addmissible = (state[:,:, vn['S_DX']] + sim_par['dt'] * actions[:,:, vn['A_ACC']] < 4.0) *(state[:,:, vn['S_DX']] + sim_par['dt'] * actions[:,:, vn['A_ACC']] > -2.0)
    dstate[:,:, vn['S_DX']] = is_addmissible * dstate[:,:, vn['S_DX']] + ~is_addmissible * (dstate[:,:, vn['S_DX']]- 1/mod_par['m']*actions[:, :, vn['A_ACC']])


    next_states = state + sim_par['dt'] * dstate  
    
    # clamp steering angle to constraints
    up = next_states[:, :, vn['S_DELTA']] > mod_par['max_steering_ang']
    low =  next_states[:, :, vn['S_DELTA']] < -mod_par['max_steering_ang']
    next_states[:, :, vn['S_DELTA']] = up * mod_par['max_steering_ang'] - low * mod_par['max_steering_ang'] + ~(up|low)*next_states[:, :, vn['S_DELTA']]

    #resolve collisions
    task.contact_wrenches = resolve_collsions(task.contact_wrenches,
                                              task.states,
                                              task.collision_pairs,
                                              mod_par['lf'],
                                              task.collision_verts,
                                              task.R,
                                              task.P_tot,
                                              task.D_tot,
                                              task.S_mat,
                                              task.Repf_mat,
                                              task.Ds,
                                              task.num_envs,
                                              task.zero_pad)
    return next_states
"""

def step_cars(state : torch.Tensor, 
              actions : torch.Tensor, 
              col_wrenches : torch.Tensor, 
              num_agents : int, 
              mod_par : Dict[str, float],  
              sim_par : Dict[str, float], 
              vn : Dict[str, int]) -> torch.Tensor:
    #set steering angle
    dir = torch.sign(actions[:, :, vn['A_STEER']] - state[:, :, vn['S_STEER']])
    val = torch.abs(actions[:, :, vn['A_STEER']] - state[:, :, vn['S_STEER']])

    state[:, :, vn['S_STEER']] = sim_par['dt']*dir * torch.min(50.0 * val, 3.0)
    torch.clamp(state[:, :, vn['S_STEER']], -np.pi/3, np.pi/3)

    #set gas 
    diff = state[:, :, vn['S_GAS']] - actions[:, :, vn['A_GAS']]
    state[:, :, vn['S_GAS']] += torch.clamp(diff, max=0.1)
    state[:, :, vn['S_GAS']] = torch.clamp(state[:, :, vn['S_GAS']], 0, 1)
    
    #set wheel speeds

    #set brake
    diff = state[:, :, vn['S_GAS']] - actions[:, :, vn['A_GAS']]
    state[:, :, vn['S_GAS']] += torch.clamp(diff, max=0.1)
    state[:, :, vn['S_GAS']] = torch.clamp(state[:, :, vn['S_GAS']], 0, 1)

    pass