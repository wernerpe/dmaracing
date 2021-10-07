import torch
from torch._C import dtype
from dmaracing.env.racing_sim import DmarEnv
from dmaracing.utils.helpers import *
import os
import sys

def play():
    cfg['sim']['numEnv'] = 1
    env = DmarEnv(cfg, args)
    
    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], 2), device=args.device,  dtype= torch.float, requires_grad=False)
    actions[:, 0, 0] = 1.0
    for _ in range(10):
        obs, rew, dones, info = env.step(actions)
        print('1:', env.states[:,0,:])
        print('2:', env.states[:,1,:])

    print('up and running')
    
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0' 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train = getcfg(path_cfg)
    play()    
