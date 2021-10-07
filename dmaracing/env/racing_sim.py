import torch
from torch._C import dtype
from dmaracing.env.car_dynamics import *
from typing import Tuple
import numpy as np

class DmarEnv:
    def __init__(self, cfg, args) -> None:
        self.device = args.device

        #variable names and indices
        self.vn = get_varnames() 
        self.modelParameters = cfg['model']
        self.simParameters = cfg['sim']
        self.num_states = self.simParameters['numStates']
        self.num_actions = self.simParameters['numActions']
        self.num_obs = self.simParameters['numObservations']       
        self.num_agents = self.simParameters['numAgents']
        self.num_envs = self.simParameters['numEnv']

        self.info = {}

        #allocate tensors
        torch_zeros = lambda shape: torch.zeros(shape, device=self.device, dtype= torch.float, requires_grad=False)
        self.states = torch_zeros((self.num_envs, self.num_agents, self.num_states))
        self.actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.obs_buf = torch_zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch_zeros((self.num_envs, self.num_agents,))
        self.reset_buf =torch_zeros((self.num_envs, ))>1

        env_ids = torch.arange(self.num_envs, dtype = torch.long)
        self.reset(env_ids)

        

    def observations(self,) -> None:
        pass
    
    def check_termination():
        pass

    def reset(self, env_ids) -> None:
        self.states[env_ids, :, self.vn['S_X']] = 1.5*self.modelParameters['w']*torch.tile(torch.arange(self.num_agents, device=self.device, dtype= torch.float, requires_grad=False), (len(env_ids),1))
        self.states[env_ids, :, self.vn['S_Y']] = 0
        self.states[env_ids, :, self.vn['S_THETA']] = np.pi/2
        self.states[env_ids, :, self.vn['S_DX']:] = 0.0

    def post_physics_step(self) -> None:
        pass

    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]] :

        self.actions = actions.clone().to(self.device)
        self.states = step_cars(self.states, self.actions, self.num_agents, self.modelParameters, self.simParameters, self.vn)    
        self.post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.info
    
    def render(self, actions) -> None:
        pass
    