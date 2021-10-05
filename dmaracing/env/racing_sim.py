import torch

#states
#postion, heading, derivatives, current steering angle
#[x,y, theta, dx, dy, omega, alpha]
num_env = 1000 
device = torch.cuda('cuda')
states = torch.zeros((num_env, 7), device = device, dtype = torch.float )


import torch



class DmarEnv:
    def __init__(self) -> None:
        #load params
        #allocate tensors
        state
        pass

    def observations(self,) -> None:
        pass
    
    def reset(self, env_ids) -> None:
        pass
    
    def step(self, actions) -> None:
        pass
    
    def render(self, actions) -> None:
        pass
    