import torch
from dmaracing.env.car_dynamics import *


class DmarEnv:
    def __init__(self, cfg, args) -> None:
        self.device = args.device
        self.names = get_varnames() 
        self.modelParameters = cfg['model']
        self.simParameters = cfg['sim']
        self.num_states = self.simParameters['numStates']
        self.num_actions = self.simParameters['numActions']
        self.num_obs = self.simParameters['numObservations']       
        self.num_agents = self.simParameters['numAgents']
        self.num_envs = self.simParameters['numEnv']

        #load params
        self.params = cfg
        
        #allocate tensors
        self.states = torch.zeros((self.num_envs, self.num_agents, self.num_agents))
        self.actions = torch.zeros((self.num_envs, self.num_agents, self.num_actions))
        self.obs_buf = torch.zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch.zeros((self.num_envs, self.num_agents,))

        env_ids = torch.arange(self.num_envs, dtype = torch.int32)
        self.reset(env_ids)

        

    def observations(self,) -> None:
        pass
    
    def reset(self, env_ids) -> None:
        pass
    
    def post_physics_step(self, env_ids) -> None:
        pass

    def step(self, actions) -> None:

        self.actions = actions.clone().to(self.device)
        self.states = step_cars(self.states, self.actions, self.modelParameters, self.simParameters, self.names)    
        self.post_physics_step()

    
    def render(self, actions) -> None:
        pass
    