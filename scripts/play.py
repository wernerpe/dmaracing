import torch
from torch._C import dtype
from dmaracing.env.racing_sim import DmarEnv
from dmaracing.utils.helpers import *
import os
import sys
import time

def play():
    cfg['sim']['numEnv'] = 100000
    env = DmarEnv(cfg, args)
    
    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], 2), device=args.device,  dtype= torch.float, requires_grad=False)
    actions[:, :, 0] = 0.01
    actions[:, :, 1] = 0.1
    num_trans = 0
    t0 = time.time()
    for idx in range(100):
        obs, rew, dones, info = env.step(actions)
        num_trans += 1
        #print('1:', env.states[0, 0, :3])
        #print('2:', env.states[:,1,-1])
    t1 = time.time()
    print('time', t1-t0)
    print('num trans', num_trans*cfg['sim']['numEnv']*cfg['sim']['numAgents'])
    print('up and running')
    
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = True 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train = getcfg(path_cfg)
    play()    
