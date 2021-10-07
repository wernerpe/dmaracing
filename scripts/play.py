from dmaracing.env.racing_sim import DmarEnv
from dmaracing.utils.helpers import *
import os
import sys

def play():
    cfg['sim']['numenv'] = 10
    env = DmarEnv(cfg, args)
    env.reset()
    print('up and running')
    
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0' 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train = getcfg(path_cfg)
    play()    
