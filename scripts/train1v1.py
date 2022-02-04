from dmaracing.utils.rl_helpers import get_mappo_runner
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os

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
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = True 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir_root = getcfg(path_cfg)
    cfg['sim']['numAgents'] = 2
    cfg['sim']['collide'] = 1
    cfg_train['runner']['experiment_name'] = '1v1_supercloud'
    #cfg['track']['num_tracks'] = 2
    set_dependent_cfg_entries(cfg)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logdir = logdir_root+'/'+timestamp

    INIT_FROM_CHKPT = False
    #active policies
    runs = ['22_02_01_18_57_26', '22_02_01_18_57_26']
    chkpts = [-1, -1]
    ##policies to populate adversary buffer
    adv_runs = ['22_02_01_18_57_26'] * 5
    adv_chkpts = [1000, 1500, 2000, 2200, 2500]

    train()