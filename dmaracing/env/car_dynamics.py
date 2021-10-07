import torch   
from enum import Enum
from typing import Dict

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


def detect_collisison():
    NotImplementedError

def resolve_collsion():
    NotImplementedError

@torch.jit.script
def state_derivative(state : torch.Tensor, actions : torch.Tensor, par : Dict[str, float], vn : Dict[str, int])-> torch.Tensor:
    #https://onlinelibrary.wiley.com/doi/pdf/10.1002/oca.2123
    # 
    # [x, y, theta, xdot, ydot, thetadot, delta]
    #blah wip
    
    alphaf = -torch.atan2(state[:, :, vn['S_DTHETA']]*par['lf'] + state[:, :, vn['S_DY']], state[:, :, vn['S_DX']] ) + state[:, :, vn['S_DELTA']]
    alphar = torch.atan2(state[:, :, vn['S_DTHETA']]*par['lr'] - state[:, :, vn['S_DY']], state[:, :, vn['S_DX']])
    Ffy = par['Df'] * torch.sin(par['Cf']*torch.atan(par['Bf']*alphaf)) 
    Fry = par['Dr'] * torch.sin(par['Cr']*torch.atan(par['Br']*alphar))  
    Frx = actions[:, vn['A_ACC']]

    dx = state[:, :, vn['S_DX']] * torch.cos(state[:, :, vn['S_THETA']]) - state[:, :, vn['S_DY']] * torch.sin(state[:, :, vn['S_THETA']]) 
    dy = state[:, :, vn['S_DX']] * torch.sin(state[:, :, vn['S_THETA']]) + state[:, :, vn['S_DY']] * torch.cos(state[:, :, vn['S_THETA']])
    dtheta = state[:, :, vn['S_DTHETA']]
    ddx = 1/par['m']*( Frx - Ffy*torch.sin(state[:, :, vn['S_DELTA']]) + par['m']*state[:, :, vn['S_DY']]*state[:, :, vn['S_DTHETA']] )
    ddy = 1/par['m']*( Fry + Ffy*torch.cos(state[:, :, vn['S_DELTA']]) - par['m']*state[:, :, vn['S_DX']]*state[:, :, vn['S_DTHETA']] )
    ddtheta = 1/par['Iz']*(Ffy*par['lf']*torch.cos(state[:, :, vn['S_DELTA']]))
    ddelta = actions[:, :, vn['A_DDELTA']]

    dstate = torch.cat((dx, dy, dtheta, ddx, ddy, ddtheta, ddelta), dim=2)
    return dstate


@torch.jit.script
def step_cars(state : torch.Tensor, actions : torch.Tensor, mod_par : Dict[str, float], sim_par : Dict[str, float], variable_names : Dict[str, int]) -> torch.Tensor:
    next_states = state + sim_par['Ts'] * state_derivative(state, actions, mod_par, variable_names) 
    
    #resolve collisions
    return next_states
