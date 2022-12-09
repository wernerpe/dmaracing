from dmaracing.utils.rl_helpers import get_mappo_runner
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
import torch 
import sys

def train():
    env = DmarEnv(cfg, args)
    runner = get_mappo_runner(env, cfg_train, logdir, args.device)
    
    if INIT_FROM_CHKPT:
        #load active policies
        model_paths = []
        for run, chkpt in zip(runs, chkpts):
            dir, modelnr = get_run(logdir_root, run = run, chkpt=chkpt)
            model_paths.append("{}/model_{}.pt".format(dir, modelnr))
            print("Loading model" + model_paths[-1])

        runner.load_multi_path(model_paths, load_optimizer=True)
        
        #populate adversary buffer
        adv_model_paths = []
        for run, chkpt in zip(adv_runs, adv_chkpts):
            dir, modelnr = get_run(logdir_root, run = run, chkpt=chkpt)
            adv_model_paths.append("{}/model_{}.pt".format(dir, modelnr))
            print("Loading model" + adv_model_paths[-1])

        runner.populate_adversary_buffer(adv_model_paths)

    runner.learn(cfg_train['runner']['max_iterations'], init_at_random_ep_len=True)

if __name__ == "__main__":
    
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    print('[DMAR TRAIN] Available gpus:', available_gpus)
    print(torch.cuda.is_available())
    args = CmdLineArguments()
    args.parse(sys.argv[1:])
    args.headless = True
    args.device = 'cuda:0'
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix_train='_1v1')
    cfg['sim']['numAgents'] = 2
    cfg['sim']['collide'] = 1 
    args.override_cfg_with_args(cfg, cfg_train)
    if not args.headless:
        cfg['viewer']['logEvery'] = -1
    #cfg_train['runner']['experiment_name'] = '1v1_supercloud'
    #cfg['track']['num_tracks'] = 2
    set_dependent_cfg_entries(cfg, cfg_train)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logdir = logdir_root+'/'+timestamp+'_col_'+str(cfg['sim']['collide'])+'_ar_'+str(cfg['learn']['actionRateRewardScale'])+'_rr_'+str(cfg['learn']['rankRewardScale'])

    cfg["logdir"] = logdir

    INIT_FROM_CHKPT = False
    #active policies
    runs = ['22_09_06_13_02_48_col_0']
    chkpts = [-1]
    ##policies to populate adversary buffer
    adv_runs = ['22_09_06_13_02_48_col_0'] * 5
    adv_chkpts = [500]*5

    train()