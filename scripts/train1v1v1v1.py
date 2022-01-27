from dmaracing.utils.rl_helpers import get_mappo_runner
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os

def train():
    env = DmarEnv(cfg, args)
    runner = get_mappo_runner(env, cfg_train, logdir, args.device, cfg['sim']['numAgents'])
    runner.learn(cfg_train['runner']['max_iterations'], init_at_random_ep_len=True)

if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    cfg['sim']['numAgents'] = 4
    cfg['sim']['collide'] = 1
    #cfg['track']['num_tracks'] = 2
    set_dependent_cfg_entries(cfg)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logdir = logdir+'/'+timestamp

    train()