import torch   
from enum import Enum
from typing import Dict
import numpy as np

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

def get_collision_pairs(num_agents):
    #naively check all collision pairs
    ls = np.arange(num_agents)
    pairs = [[a, b] for idx, a in enumerate(ls) for b in ls[idx + 1:]]
    return pairs

#only used for collsiion checking, only one car per env needed for pairwise checking
def get_car_vert_mat(w, l, num_envs, device):
    verts = torch.zeros((num_envs, 4, 2), device=device, dtype=torch.float, requires_grad=False)
    #top lft , ccw
    # 3 x-------x 0
    #   |       |
    # 2 X-------X 1
    #
    # ^ y 
    # ->x
    verts[:, 0, 0] = l/2
    verts[:, 0, 1] = w/2
    verts[:, 1, 0] = l/2
    verts[:, 1, 1] = -w/2
    verts[:, 2, 0] = -l/2
    verts[:, 2, 1] = -w/2
    verts[:, 3, 0] = -l/2
    verts[:, 3, 1] = w/2
    return verts

@torch.jit.script
def transform_col_verts(rel_trans_global : torch.Tensor, 
                        theta_A : torch.Tensor, 
                        theta_B : torch.Tensor, 
                        verts: torch.Tensor) ->torch.Tensor:
    '''
    (B collider vertex rep, A collidee poly rep)
    input: 
    rel_trans_global (num_envs, 2) :relative translation pB-pA in global frame 
    theta_A (num envs) : relative rotation of A to world
    theta_B (num envs) : relative rotation of B to world
    verts : vertex tensor (numenvs, 4 car vertices, 2 cords) 
    '''
    theta_rel = theta_B - theta_A
    R = torch.zeros((len(theta_rel), 2, 2), device = theta_rel.device)
    R[:, 0, 0 ] = torch.cos(theta_rel)
    R[:, 0, 1 ] = -torch.sin(theta_rel)
    R[:, 1, 0 ] = torch.sin(theta_rel)
    R[:, 1, 1 ] = torch.cos(theta_rel)
    verts_rot = torch.einsum('kij, klj->kli', R, verts)
    #rotate global translation into A frame
    R[:, 0, 0 ] = torch.cos(theta_A)
    R[:, 0, 1 ] = torch.sin(theta_A)
    R[:, 1, 0 ] = -torch.sin(theta_A)
    R[:, 1, 1 ] = torch.cos(theta_A)
    trans_rot = torch.einsum('kij, kj->ki', R, rel_trans_global)

    verts_rot_shift = verts_rot
    verts_rot_shift[:,0,:] += trans_rot[:]
    verts_rot_shift[:,1,:] += trans_rot[:]
    verts_rot_shift[:,2,:] += trans_rot[:]
    verts_rot_shift[:,3,:] += trans_rot[:]
    return verts_rot


def resolve_collsions():

    contact_forces = None
    return contact_forces

@torch.jit.script
def state_derivative(state : torch.Tensor, actions : torch.Tensor, par : Dict[str, float], num_agents : int, vn : Dict[str, int])-> torch.Tensor:
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
    ddx = 1/par['m']*( Frx - Ffy*torch.sin(state[:, :, vn['S_DELTA']]) + par['m']*state[:, :, vn['S_DY']]*state[:, :, vn['S_DTHETA']] )
    ddy = 1/par['m']*( Fry + Ffy*torch.cos(state[:, :, vn['S_DELTA']]) - par['m']*state[:, :, vn['S_DX']]*state[:, :, vn['S_DTHETA']] )
    ddtheta = 1/par['Iz']*(Ffy*par['lf']*torch.cos(state[:, :, vn['S_DELTA']]))
    
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
def step_cars(state : torch.Tensor, actions : torch.Tensor, num_agents : int, mod_par : Dict[str, float],  sim_par : Dict[str, float], vn : Dict[str, int]) -> torch.Tensor:
    dstate = state_derivative(state, actions, mod_par, num_agents , vn)
    
    #handle constraints
    is_addmissible = (state[:,:, vn['S_DX']] + sim_par['dt'] * actions[:,:, vn['A_ACC']] < 4.0) *(state[:,:, vn['S_DX']] + sim_par['dt'] * actions[:,:, vn['A_ACC']] > -2.0)
    dstate[:,:, vn['S_DX']] = is_addmissible * dstate[:,:, vn['S_DX']] + ~is_addmissible * (dstate[:,:, vn['S_DX']]- 1/mod_par['m']*actions[:, :, vn['A_ACC']])


    next_states = state + sim_par['dt'] * dstate  
    
    # clamp steering angle to constraints
    up = next_states[:, :, vn['S_DELTA']] > np.pi/2
    low =  next_states[:, :, vn['S_DELTA']] < -np.pi/2
    next_states[:, :, vn['S_DELTA']] = up * np.pi/2 - low * np.pi/2 + ~(up|low)*next_states[:, :, vn['S_DELTA']]

    #resolve collisions
    
    return next_states
