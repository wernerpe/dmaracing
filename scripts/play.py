import torch
from torch._C import dtype
from dmaracing.env.racing_sim import DmarEnv
from dmaracing.utils.helpers import *
import os
import sys
import time
import numpy as np

def play():
    cfg['sim']['numEnv'] = 2000
    env = DmarEnv(cfg, args)
    
    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], 2), device=args.device,  dtype= torch.float, requires_grad=False)
    vel_cmd = 0.0
    steer_cmd = 0.0
    num_step = 400


    t0 = time.time()

    while True:
        actions[:,0, 0] = 2.0*(vel_cmd - env.states[:,0,env.vn['S_DX']])
        actions[:,0, 1] = 3.*(steer_cmd - env.states[:,0,env.vn['S_DELTA']]) 
        obs, rew, dones, info = env.step(actions)
        #print(info['key'])
        
        evt = info['key']

        if evt == 105:
            vel_cmd += 0.2
            print('vel_cmd', vel_cmd, env.states[0,0,env.vn['S_DX']])
        elif evt == 107:
            vel_cmd -= 0.2
            print('vel_cmd', vel_cmd, env.states[0,0,env.vn['S_DX']])
        elif evt == 106:
            steer_cmd += 0.4 * (steer_cmd < cfg['model']['max_steering_ang'])
            print('steer_cmd', steer_cmd, env.states[0,0,env.vn['S_DELTA']])
        elif evt == 108:
            steer_cmd -= 0.4 * (steer_cmd> -cfg['model']['max_steering_ang'])
            print('steer_cmd', steer_cmd, env.states[0,0,env.vn['S_DELTA']])
        
        #if ( idx%50  ) ==0:
        #    print(idx)
        #print('1:', env.states[0, 0, :3])
        #print('2:', env.states[:,1,-1])
    t1 = time.time()
    print('time', t1-t0)
    print('num trans', num_step*cfg['sim']['numEnv']*cfg['sim']['numAgents'])
    print('up and running')
    
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train = getcfg(path_cfg)
    play()    
