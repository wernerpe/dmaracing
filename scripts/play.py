import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
import os

def play():
    cfg['sim']['numEnv'] = 10
    env = DmarEnv(cfg, args)
    obs = env.obs_buf[:,0,:]
    dir, model = get_run(logdir, run = -1, chkpt=-1)
    cfg_train['learn']['resume'] = model
    ppo = get_ppo(args, env, cfg_train, dir)
    policy = ppo.actor_critic.act_inference
   
    while True:
        actions = policy(obs)
        print(actions[0,:])
        obs, rew, dones, info = env.step(actions)
        print(obs)
        
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    args.test = True
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    play()    
