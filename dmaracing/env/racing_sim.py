import torch
from torch._C import dtype
from dmaracing.env.car_dynamics import step_cars
from dmaracing.env.car_dynamics_utils import get_varnames, set_dependent_params, allocate_car_dynamics_tensors 
from dmaracing.env.viewer import Viewer
from typing import Tuple, Dict
import numpy as np
from dmaracing.utils.trackgen import get_track

class DmarEnv:
    def __init__(self, cfg, args) -> None:
        self.device = args.device
        self.headless = args.headless
        
        #variable names and indices
        self.vn = get_varnames() 
        self.modelParameters = cfg['model']
        set_dependent_params(self.modelParameters)
        self.simParameters = cfg['sim']
        self.num_states = self.simParameters['numStates']
        self.num_actions = self.simParameters['numActions']
        self.num_obs = self.simParameters['numObservations']       
        self.num_agents = self.simParameters['numAgents']
        self.num_envs = self.simParameters['numEnv']
        self.collide = self.simParameters['collide']

        self.track, self.tile_len = get_track(cfg)
    
        if not self.headless:
            self.viewer = Viewer(cfg, self.track)
        self.info = {}
        self.info['key'] = None

        #allocate tensors
        torch_zeros = lambda shape: torch.zeros(shape, device=self.device, dtype= torch.float, requires_grad=False)
        self.states = torch_zeros((self.num_envs, self.num_agents, self.num_states))
        self.contact_wrenches = torch_zeros((self.num_envs, self.num_agents, 3))
        self.actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.obs_buf = torch_zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch_zeros((self.num_envs, self.num_agents,))
        self.reset_buf = torch_zeros((self.num_envs, ))>1

        allocate_car_dynamics_tensors(self)

        env_ids = torch.arange(self.num_envs, dtype = torch.long)
        self.reset(env_ids)

        
    def observations(self,) -> None:
        pass
    
    def check_termination():
        pass

    def reset(self, env_ids) -> None:
        
        for agent in range(self.num_agents):
            tile_idx = 0 + -2*agent 
            x, y = self.track[1][tile_idx, :]
            offset_x = np.cos(self.track[3][tile_idx] + np.pi/2)* self.modelParameters['L'] + np.cos(self.track[3][tile_idx])* self.modelParameters['L']
            offset_y = np.sin(self.track[3][tile_idx] + np.pi/2)* self.modelParameters['L'] + np.sin(self.track[3][tile_idx])* self.modelParameters['L']
            self.states[env_ids, agent, self.vn['S_X']] = x + 0*offset_x
            self.states[env_ids, agent, self.vn['S_Y']] = y + 0*offset_y
            self.states[env_ids, agent, self.vn['S_THETA']] = self.track[3][tile_idx] + np.pi/2 
            #self.states[env_ids, agent, self.vn['S_THETA']+1:] = 0.0
            
        if not self.headless:
            self.render()

    def post_physics_step(self) -> None:
        if not self.headless:
            self.render()

    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]] :
        
        self.actions = actions.clone().to(self.device)
        self.states, self.contact_wrenches = step_cars(self.states, 
                                                       self.actions, 
                                                       self.wheel_locations, 
                                                       self.R, 
                                                       self.contact_wrenches, 
                                                       self.modelParameters, 
                                                       self.simParameters, 
                                                       self.vn,
                                                       self.collision_pairs,
                                                       self.collision_verts,
                                                       self.P_tot,
                                                       self.D_tot,
                                                       self.S_mat,
                                                       self.Repf_mat,
                                                       self.Ds,
                                                       self.num_envs,
                                                       self.zero_pad,
                                                       self.collide
                                                       )

        self.post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.info
    
    def render(self,) -> None:
        self.evnt = self.viewer.render(self.states[:,:,[0,1,2,10]])
        self.info['key'] = self.evnt
