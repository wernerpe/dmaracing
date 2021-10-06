import torch
from dmaracing.env.car_dynamics import *

num_env = 1000 
device = torch.cuda('cuda')
states = torch.zeros((num_env, 7), device = device, dtype = torch.float )



class DmarEnv:
    def __init__(self, args, cfg) -> None:
        self.device = args.device
        self.modelParameters = cfg['model']
        self.simParameters = cfg['sim']
        self.num_states = self.modelParameters['numStates']
        self.num_actions = self.modelParameters['numActions']
        self.num_obs = self.simParameters['numObservations']       
        self.num_agents = 1
        self.num_envs = 100

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
        self.actions =     


    
    def render(self, actions) -> None:
        pass
    