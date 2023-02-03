from dmaracing.utils.rl_helpers import get_bima_ppo_runner
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train():
    env = DmarEnvBilevel(cfg, args)
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
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless = True  # False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='_bilevel', postfix_train='_bilevel')
    #cfg['sim']['numAgents'] = 4
    #cfg['sim']['collide'] = 1
    if not args.headless:
        cfg['viewer']['logEvery'] = -1
    # cfg['sim']['numEnv'] = 16
    # cfg['track']['num_tracks'] = 2
    #cfg_train['policy']['teamsize'] = 2
    #cfg_train['policy']['numteams'] = 2
    # cfg_train['runner']['policy_class_name'] = 'MultiTeamCMAAC' #MAActorCritic 
    # cfg_train['runner']['algorithm_class_name'] = 'JRMAPPO' #IMAPPO 
    #cfg_train['runner']['num_steps_per_env'] = 32
    #cfg_train['runner']['population_update_interval'] = 5

    cfg_train['policy']['numteams'] = 2
    cfg_train['policy']['teamsize'] = 2
    cfg['learn']['agent_dropout_prob'] = 0.2  # 0.0
    # cfg['viewer']['logEvery'] = 1

    cfg['teams'] = dict()
    cfg['teams']['numteams'] = cfg_train['policy']['numteams']
    cfg['teams']['teamsize'] = cfg_train['policy']['teamsize']

    args.override_cfg_with_args(cfg, cfg_train)
    set_dependent_cfg_entries(cfg, cfg_train)

    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    # logdir = logdir_root +'/'+timestamp+'_no_dist_to_go_' + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    # logdir = logdir_root +'/'+timestamp + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    logdir = logdir_root +'/'+timestamp + '_bilevel_2v2'
    cfg["logdir"] = logdir


    INIT_FROM_CHKPT = False
    #active policies
    runs_hl = ['23_01_10_08_21_36_bilevel_2v2']*2
    chkpts_hl = [500, 500]
    runs_ll = ['23_01_10_08_21_36_bilevel_2v2']*2
    chkpts_ll = [500, 500]
    ##policies to populate adversary buffer
    adv_runs = ['23_01_10_08_21_36_bilevel_2v2'] * 3
    adv_chkpts = [500, 500, 500]

    train()