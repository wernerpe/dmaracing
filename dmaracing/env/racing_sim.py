import torch
from torch._C import dtype
from dmaracing.env.car_dynamics import *
from dmaracing.env.car_dynamics_utils import allocate_car_dynamics_tensors
from dmaracing.env.viewer import Viewer
from typing import Tuple
import numpy as np

class DmarEnv:
    def __init__(self, cfg, args) -> None:
        self.device = args.device
        self.headless = args.headless

        #variable names and indices
        self.vn = get_varnames() 
        self.modelParameters = cfg['model']
        self.simParameters = cfg['sim']
        self.num_states = self.simParameters['numStates']
        self.num_actions = self.simParameters['numActions']
        self.num_obs = self.simParameters['numObservations']       
        self.num_agents = self.simParameters['numAgents']
        self.num_envs = self.simParameters['numEnv']
        if not self.headless:
            self.viewer = Viewer(cfg)
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
        self.states[env_ids, :, self.vn['S_X']] = 1.5*self.modelParameters['w']*torch.tile(torch.arange(self.num_agents, device=self.device, dtype= torch.float, requires_grad=False), (len(env_ids),1))
        self.states[env_ids, :, self.vn['S_Y']] = 0
        self.states[env_ids, :, self.vn['S_THETA']] = np.pi/2
        self.states[env_ids, :, self.vn['S_DX']:] = 0.0
        if not self.headless:
            self.render()

    def post_physics_step(self) -> None:
        if not self.headless:
            self.render()

    #change this??? to make it jit
    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]] :
        
        self.actions = actions.clone().to(self.device)
        self.states = step_cars(self, self.states, self.actions, self.contact_wrenches, self.num_agents, self.modelParameters, self.simParameters, self.vn)    
        self.post_physics_step()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.info
    
    def render(self,) -> None:
        self.evnt = self.viewer.render(self.states[:,:,[0,1,2,6]])
        self.info['key'] = self.evnt
