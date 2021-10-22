import torch
import time
from dmaracing.env.car_dynamics import step_cars, get_varnames, set_dependent_params
from dmaracing.utils.helpers import getcfg
import os

path_cfg = os.getcwd() + '/cfg'
cfg, cfg_train = getcfg(path_cfg)
sim_par = cfg['sim']
mod_par = cfg['model']
num_envs = 10
num_agents = 5
vn = get_varnames() 
device = 'cuda:0'
state = torch.zeros((num_envs, num_agents, 12), device=device)
actions = torch.ones((num_envs, num_agents, 3), device=device)
actions[:,:, 2] = 0
R = torch.zeros((num_envs, num_agents, 2, 2), device=device)

wheel_locations = torch.zeros((4,2), device = device)
set_dependent_params(cfg['model'])
L = cfg['model']['L']
W = cfg['model']['W']
wheel_locations[0, 0] = L/2.0 
wheel_locations[0, 1] = W/2.0 
wheel_locations[1, 0] = L/2.0 
wheel_locations[1, 1] = -W/2.0 
wheel_locations[2, 0] = -L/2.0 
wheel_locations[2, 1] = -W/2.0 
wheel_locations[3, 0] = -L/2.0 
wheel_locations[3, 1] = W/2.0 

col_wrenches = torch.zeros((num_envs, num_agents, 3), device=device)


state = step_cars(state,actions, wheel_locations, R, col_wrenches, mod_par, sim_par, vn)
state = step_cars(state,actions, wheel_locations, R, col_wrenches, mod_par, sim_par, vn)

print('done')