import torch   
from enum import Enum


class state(Enum):
    #postition
    x = 0
    y = 1
    #heading
    theta = 2 
    #timederivatives
    dx = 3
    dy = 4
    dtheta = 5
    #steering angle
    delta =6

class state(Enum):
    #accelerator
    acc = 0
    #steering angle change
    ddelta =1

#statenames
s = state

def detect_collisison():
    NotImplementedError

def resolve_collsion():
    NotImplementedError

def state_derivative(sim)-> torch.Tensor:
    #https://onlinelibrary.wiley.com/doi/pdf/10.1002/oca.2123
    # 
    # [x, y, theta, xdot, ydot, thetadot, delta]
    #blah wip

    state = sim.state
    actions = sim.actions
    par = sim.modelParameters
    
    alphaf = -torch.arctan2(state[:, :, s.dtheta]*par['lf'] + state[:, :, s.dy], state[:, :, s.dx]) + state[:, :, s.delta]
    alphar = torch.arctan2(state[:, :, s.dtheta]*par['lr'] - state[:, :, s.dy], state[:, :, s.dx])
    Ffy = par['Df'] * torch.sin(par['Cf']*torch.arctan2(par['Bf']*alphaf)) 
    Fry = par['Dr'] * torch.sin(par['Cr']*torch.arctan2(par['Br']*alphar))  
    Frx = actions['acc']

    dx = state[:, :, s.dx] * torch.cos(state[:, :, s.theta]) - state[:, :, s.dy] * torch.sin(state[:, :, s.theta]) 
    dy = state[:, :, s.dx] * torch.sin(state[:, :, s.theta]) + state[:, :, s.dy] * torch.cos(state[:, :, s.theta])
    dtheta = state[:, :, s.dtheta]
    ddx = 1/par['m']*( Frx - Ffy*torch.sin(state[:, :, s.delta]) + par['m']*state[:, :, s.dy]*state[:, :, s.dtheta] )
    ddy = 1/par['m']*( Fry + Ffy*torch.cos(state[:, :, s.delta]) - par['m']*state[:, :, s.dx]*state[:, :, s.dtheta] )
    ddtheta = 1/par['Iz']*(Ffy*par['lf']*torch.cos(state[:, :, s.delta]))
    ddelta = actions[:, :, 'ddelta']

    dstate = torch.cat((dx, dy, dtheta, ddx, ddy, ddtheta, ddelta), dim=2)
    return dstate

def step_cars(sim) -> torch.Tensor:
    next_states = sim.state + sim.simulationParameters['Ts'] * state_derivative(sim) 
    #resolve collisions
    sim.state = next_states
    return next_states