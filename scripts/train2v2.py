from dmaracing.utils.rl_helpers import get_mappo_runner
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os

def train():
    env = DmarEnv(cfg, args)
    runner = get_mappo_runner(env, cfg_train, logdir, args.device, cfg['sim']['numAgents'])

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
    cfg['sim']['numAgents'] = 4
    cfg['sim']['collide'] = 1
    #cfg['viewer']['logEvery'] = -1
    #cfg['sim']['numEnv'] = 16
    #cfg['track']['num_tracks'] = 2
    cfg_train['policy']['teamsize'] = 2
    cfg_train['policy']['numteams'] = 2
    cfg_train['runner']['policy_class_name'] = 'MultiTeamCMAAC' #MAActorCritic 
    cfg_train['runner']['algorithm_class_name'] = 'JRMAPPO' #IMAPPO 

    set_dependent_cfg_entries(cfg)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logdir = logdir_root +'/'+ cfg_train['runner']['algorithm_class_name'] + '_' + timestamp
    cfg["logdir"] = logdir
    INIT_FROM_CHKPT = False
    #active policies
    runs = ['22_02_03_21_09_18']*4
    chkpts = [-1] * 4
    ##policies to populate adversary buffer
    adv_runs = ['22_02_03_21_09_18'] * 10
    adv_chkpts = [1000, 1500, 2000, 2200, 2500, 5000, 10000, 11000, 12000, 10500]

    train()