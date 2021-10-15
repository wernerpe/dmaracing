import torch   
from typing import Dict
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
    varnames['S_DELTA'] = 6
    varnames['A_ACC'] = 0
    varnames['A_DDELTA'] = 1
    return varnames

def resolve_collsions(task):
    task.contact_wrenches[:,:,:] =0.0

    if len(task.collision_pairs):
        for colp in task.collision_pairs:

            idx_comp = torch.where(torch.norm(task.states[:, colp[0], 0:2] -  task.states[:, colp[1], 0:2], dim =1)<=1)[0]# 5.6*task.modelParameters['lf'])[0]
            
            if 1:#len(idx_comp):
                states_A = task.states[idx_comp, colp[0], 0:3]
                states_B = task.states[idx_comp, colp[1], 0:3]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] -states_A[:,:2]
                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], task.collision_verts, task.R[idx_comp,:,:])

                force_B, torque_A, torque_B = get_contact_wrenches(task.P_tot, 
                                                                task.D_tot, 
                                                                task.S_mat, 
                                                                task.Repf_mat,
                                                                task.Ds,
                                                                verts_tf,
                                                                task.num_envs,
                                                                task.zero_pad
                                                                )

                #rotate forces into global frame from frame A
                force_B = rotate_vec(force_B, states_A[:, 2], task.R[idx_comp,:,:])
                task.contact_wrenches[idx_comp, colp[0], :2] = -force_B
                task.contact_wrenches[idx_comp, colp[1], :2] = force_B
                task.contact_wrenches[idx_comp, colp[0], 2] = torque_A
                task.contact_wrenches[idx_comp, colp[1], 2] = torque_B

                #flip and check other direction
                states_A = task.states[:, colp[1], 0:3]
                states_B = task.states[:, colp[0], 0:3]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] - states_A[:,:2]
                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], task.collision_verts, task.R[idx_comp,:,:])

                force_B, torque_A, torque_B = get_contact_wrenches(task.P_tot, 
                                                                task.D_tot, 
                                                                task.S_mat, 
                                                                task.Repf_mat,
                                                                task.Ds,
                                                                verts_tf,
                                                                task.num_envs,
                                                                task.zero_pad
                                                                )

                #rotate forces into global frame from frame A
                force_B = rotate_vec(force_B, states_A[:, 2], task.R)
                task.contact_wrenches[idx_comp, colp[1], :2] += -force_B
                task.contact_wrenches[idx_comp, colp[0], :2] += force_B
                task.contact_wrenches[idx_comp, colp[1], 2] += torque_A
                task.contact_wrenches[idx_comp, colp[0], 2] += torque_B
    

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
                     +   torch.cos(state[:, :, vn['S_THETA']]) * col_wrenches[:, :, 0] + torch.sin(state[:, :, vn['S_THETA']])* col_wrenches[:, :, 1] )
                    #- 0.1  * (state[:, :, vn['S_DX']]>0) +  0.1  * (state[:, :, vn['S_DX']]<0)

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


#@torch.jit.script
def step_cars(task, state : torch.Tensor, actions : torch.Tensor, col_wrenches : torch.Tensor, num_agents : int, mod_par : Dict[str, float],  sim_par : Dict[str, float], vn : Dict[str, int]) -> torch.Tensor:
    dstate = state_derivative(state, actions, col_wrenches, mod_par, num_agents , vn)
    
    #handle constraints
    is_addmissible = (state[:,:, vn['S_DX']] + sim_par['dt'] * actions[:,:, vn['A_ACC']] < 4.0) *(state[:,:, vn['S_DX']] + sim_par['dt'] * actions[:,:, vn['A_ACC']] > -2.0)
    dstate[:,:, vn['S_DX']] = is_addmissible * dstate[:,:, vn['S_DX']] + ~is_addmissible * (dstate[:,:, vn['S_DX']]- 1/mod_par['m']*actions[:, :, vn['A_ACC']])


    next_states = state + sim_par['dt'] * dstate  
    
    # clamp steering angle to constraints
    up = next_states[:, :, vn['S_DELTA']] > np.pi/3
    low =  next_states[:, :, vn['S_DELTA']] < -np.pi/3
    next_states[:, :, vn['S_DELTA']] = up * np.pi/3 - low * np.pi/3 + ~(up|low)*next_states[:, :, vn['S_DELTA']]

    #resolve collisions
    resolve_collsions(task)
    return next_states
