from dmaracing.utils.rl_helpers import get_bima_ppo_runner
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def eval():
    env = DmarEnvBilevel(cfg, args)
    runner = get_bima_ppo_runner(env, cfg_train, logdir, args.device)

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

    policy_eval_hl = runner.get_inference_policy_hl(device=env.device)
    policy_eval_ll = runner.get_inference_policy_ll(device=env.device)

    run_eval(env, policy_eval_hl, policy_eval_ll)


def run_eval(env, policy_hl, policy_ll):

    num_episodes = 5

    max_hl_steps = env.max_episode_length // env.dt_hl

    for _ in range(num_episodes):

        obs, _ = env.reset()

        already_done = env.reset_buf > 1000

        for i_hl in range(max_hl_steps):
            actions_hl_raw = policy_hl(obs)
            # reward_ep_ll = torch.zeros(self.env.num_envs, self.num_agents, 1, dtype=torch.float, device=self.device)
            for i_ll in range(env.dt_hl):
                actions_hl = env.project_into_track_frame(actions_hl_raw)
                obs_ll = torch.concat((obs, actions_hl), dim=-1)
                actions_ll = policy_ll(obs_ll)

                obs, privileged_obs, rewards, dones, infos = env.step(actions_ll)

            already_done |= dones
            if ~torch.any(~already_done):
                break


if __name__ == "__main__":
    args = CmdLineArguments()
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless = True  # False 


    # ### Run information
    exp_name = 'tri_single_blr_hierarchical'
    timestamp ='23_01_30_07_55_49_bilevel_2v2'
    checkpoint = 1000
    #active policies
    runs_hl = [timestamp]*2
    chkpts_hl = [checkpoint, checkpoint]
    runs_ll = [timestamp]*2
    chkpts_ll = [checkpoint, checkpoint]
    ##policies to populate adversary buffer
    adv_runs = [timestamp] * 3
    adv_chkpts = [checkpoint, checkpoint, checkpoint]


    path_cfg = os.getcwd() + '/logs/' + exp_name + '/' + runs_hl[0]
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='', postfix_train='')
    cfg['viewer']['logEvery'] = 1
    cfg['sim']['numEnv'] = 1

    # cfg_train['policy']['numteams'] = 2
    # cfg_train['policy']['teamsize'] = 2
    cfg['learn']['agent_dropout_prob'] = 0.0
    # cfg['learn']['resetrand'] = [0.05, 0.05, 0.0, 0.0, 0.0,  0.0, 0.0]
    cfg["sim"]["reset_timeout_only"] = True

    cfg['teams'] = dict()
    cfg['teams']['numteams'] = cfg_train['policy']['numteams']
    cfg['teams']['teamsize'] = cfg_train['policy']['teamsize']

    args.override_cfg_with_args(cfg, cfg_train)
    set_dependent_cfg_entries(cfg, cfg_train)

    # logdir = logdir_root +'/'+timestamp+'_no_dist_to_go_' + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    # logdir = logdir_root +'/'+timestamp + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    logdir = logdir_root +'/eval/'+timestamp + '_bilevel_2v2'
    cfg["logdir"] = logdir

    eval()