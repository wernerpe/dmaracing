from dmaracing.utils.rl_helpers import get_mappo_runner
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless = True  # False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir_root = getcfg(path_cfg)
    #cfg['sim']['numAgents'] = 4
    #cfg['sim']['collide'] = 1
    if not args.headless:
        cfg['viewer']['logEvery'] = -1
    cfg['sim']['numEnv'] = 16
    cfg['track']['num_tracks'] = 2
    #cfg_train['policy']['teamsize'] = 2
    #cfg_train['policy']['numteams'] = 2
    cfg_train['runner']['policy_class_name'] = 'MAActorCritic'  # 'MultiTeamCMAAC' #MAActorCritic 
    cfg_train['runner']['algorithm_class_name'] = 'IMAPPO'  # 'JRMAPPO' #IMAPPO 
    #cfg_train['runner']['num_steps_per_env'] = 32
    #cfg_train['runner']['population_update_interval'] = 5

    args.override_cfg_with_args(cfg, cfg_train)
    set_dependent_cfg_entries(cfg, cfg_train)

    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    # logdir = logdir_root +'/'+timestamp+'_no_dist_to_go_' + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    logdir = logdir_root +'/'+timestamp + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    cfg["logdir"] = logdir
    INIT_FROM_CHKPT = False
    #active policies
    runs = ['JRMAPPO22_05_16_11_41_14']*2
    chkpts = [-1] * 4
    ##policies to populate adversary buffer
    adv_runs = ['JRMAPPO22_05_16_11_41_14'] * 3
    adv_chkpts = [1000, 500, 200]

    train()