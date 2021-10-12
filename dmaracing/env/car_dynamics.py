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
    next_states[:, :, vn['S_DELTA']] = up * np.pi/2 + low * np.pi + ~(up|low)*next_states[:, :, vn['S_DELTA']]

    #resolve collisions
    
    return next_states
