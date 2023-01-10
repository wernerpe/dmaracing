from dmaracing.utils.rl_helpers import get_bilevel_ppo_runner, get_bima_ppo_runner
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def train():
    env = DmarEnvBilevel(cfg, args)
    # runner = get_bilevel_ppo_runner(env, cfg_train, logdir, args.device)
    runner = get_bima_ppo_runner(env, cfg_train, logdir, args.device)
    if INIT_FROM_CHKPT:
        #load active policies
        model_paths_hl, model_paths_ll = [], []
        for run_hl, chkpt_hl, run_ll, chkpt_ll in zip(runs_hl, chkpts_hl, runs_ll, chkpts_ll):
            dir_hl, modelnr_hl = get_run(logdir_root, run=run_hl + '/hl_model', chkpt=chkpt_hl)
            model_paths_hl.append("{}/model_{}.pt".format(dir_hl, modelnr_hl))
            print("Loading HL model" + model_paths_hl[-1])
            dir_ll, modelnr_ll = get_run(logdir_root, run=run_ll + '/ll_model', chkpt=chkpt_ll)
            model_paths_ll.append("{}/model_{}.pt".format(dir_ll, modelnr_ll))
            print("Loading LL model" + model_paths_ll[-1])
        runner.load_multi_path(model_paths_hl, model_paths_ll, load_optimizer=True)
        
        #populate adversary buffer
        adv_model_paths_hl, adv_model_paths_ll = [], []
        for run_hl, chkpt_hl, run_ll, chkpt_ll in zip(runs_hl, chkpts_hl, runs_ll, chkpts_ll):

            dir_hl, modelnr_hl = get_run(logdir_root, run=run_hl + '/hl_model', chkpt=chkpt_hl)
            adv_model_paths_hl.append("{}/model_{}.pt".format(dir_hl, modelnr_hl))
            print("Loading HL model" + adv_model_paths_hl[-1])
            dir_ll, modelnr_ll = get_run(logdir_root, run=run_ll + '/ll_model', chkpt=chkpt_ll)
            adv_model_paths_ll.append("{}/model_{}.pt".format(dir_ll, modelnr_ll))
            print("Loading LL model" + adv_model_paths_ll[-1])

        runner.populate_adversary_buffer(adv_model_paths_hl, adv_model_paths_ll)

    runner.learn(cfg_train['runner']['max_iterations'], init_at_random_ep_len=False)

if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = True 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='_bilevel', postfix_train='_bilevel')
    cfg['sim']['numAgents'] = 4
    cfg['sim']['collide'] = 1
    
    #cfg['sim']['numEnv'] = 16
    #cfg['track']['num_tracks'] = 2

    # cfg_train['runner']['policy_class_hl_name'] = 'BilevelActorCritic'
    # cfg_train['runner']['algorithm_class_hl_name'] = 'BilevelPPO'
    # cfg_train['runner']['policy_class_ll_name'] = 'ActorCritic'
    # cfg_train['runner']['algorithm_class_ll_name'] = 'PPO'

    cfg_train['policy']['numteams'] = 4
    cfg_train['policy']['teamsize'] = 1

    cfg['teams'] = dict()
    cfg['teams']['numteams'] = cfg_train['policy']['numteams']
    cfg['teams']['teamsize'] = cfg_train['policy']['teamsize']
    
    set_dependent_cfg_entries(cfg, cfg_train)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logdir = logdir_root +'/'+timestamp
    
    cfg["logdir"] = logdir
    cfg["viewer"]["logEvery"] = 10  # 10  #-1

    # cfg_train['runner']['iter_per_hl'] = 10
    # cfg_train['runner']['iter_per_hl'] = 50

    INIT_FROM_CHKPT = False
    #active policies
    runs_hl = ['22_12_14_18_03_08']*4
    chkpts_hl = [850, 500, 100, 0]  # [-1] * 4
    runs_ll = ['22_12_14_18_03_08']*4
    chkpts_ll = [850, 500, 100, 0]  # [-1] * 4
    # ##policies to populate adversary buffer
    # adv_runs_hl = ['22_12_14_10_20_10'] * 10
    # adv_chkpts_hl = [1000, 1500, 2000, 2200, 2500, 5000, 10000, 11000, 12000, 10500]
    # adv_runs_ll = ['22_12_14_10_20_10'] * 10
    # adv_chkpts_ll = [1000, 1500, 2000, 2200, 2500, 5000, 10000, 11000, 12000, 10500]

    train()