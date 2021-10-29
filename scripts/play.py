import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
import os

def play():
    cfg['sim']['numEnv'] = 2000
    cfg['sim']['numAgents'] = 5
    env = DmarEnv(cfg, args)
    obs = env.reset_all()

    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    vel_cmd = 0.0
    steer_cmd = 0.0
    brk_cmd = 0.0
      
    while True:
        actions[0 ,0, 0] = steer_cmd 
        actions[0 ,0, 1] = vel_cmd
        actions[0 ,0, 2] = brk_cmd
        
        obs, rew, dones, info = env.step(actions)
        
        evt = info['key']
        if evt == 105:
            vel_cmd += 0.1
            brk_cmd = 0
            print('vel_cmd', vel_cmd, env.states[0,0,env.vn['S_DX']])
        elif evt == 107:
            vel_cmd = 0.1
            brk_cmd = 0.9
            print('vel_cmd', vel_cmd, env.states[0,0,env.vn['S_DX']])
        elif evt == 106:
            steer_cmd += 0.4 * (steer_cmd < 1)
        elif evt == 108:
            steer_cmd -= 0.4 * (steer_cmd> -1)
       
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    play()    
