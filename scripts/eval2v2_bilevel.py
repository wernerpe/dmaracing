from dmaracing.utils.rl_helpers import get_bima_ppo_runner
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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


    save_dir = 'logs/saved_models/'+timestamp
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir+'/hl_model')
        os.mkdir(save_dir+'/ll_model')
    shutil.copy(logdir_root+'/'+timestamp+'/cfg.yml', save_dir+'/cfg.yml' )
    shutil.copy(logdir_root+'/'+timestamp+'/cfg_train.yml', save_dir+'/cfg_train.yml' )
    
    # ### Save jit models to original folder
    policy_hl_jit = torch.jit.script(runner.alg_hl.actor_critic.teamacs[0].ac.actor.to('cpu'))
    policy_hl_jit.save(save_dir+ "/hl_model/jit_model_" +str(modelnr_hl)+".pt")

    policy_ll_jit = torch.jit.script(runner.alg_ll.actor_critic.teamacs[0].ac.actor.to('cpu'))
    policy_ll_jit.save(save_dir + "/ll_model/jit_model_" +str(modelnr_ll)+".pt")

    
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

                # actions_ll = actions_ll.view((*actions_ll.shape[:-1], 2, 2))

                obs, privileged_obs, rewards, dones, infos = env.step(actions_ll)
                evt = env.viewer_events
                if evt == 121:
                    print("env ", env.viewer.env_idx_render, " reset")
                    env.episode_length_buf[env.viewer.env_idx_render] = 1e9
            already_done |= dones
            if ~torch.any(~already_done):
                break


if __name__ == "__main__":
    args = CmdLineArguments()
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless =  True 


    # ### Run information
    exp_name = 'tri_multiagent_blr_hierarchical'  # 'tri_single_blr_hierarchical'
    timestamp = '23_02_15_10_03_56_bilevel_2v2'  # '23_01_31_14_30_58_bilevel_2v2'  # '23_01_31_11_54_24_bilevel_2v2'
    checkpoint = 1000  # 500  # 1300
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
    #cfg['learn']['agent_dropout_prob'] = 0.0
    # cfg['learn']['resetrand'] = [0.05, 0.05, 0.0, 0.0, 0.0,  0.0, 0.0]
    cfg["sim"]["reset_timeout_only"] = True

    cfg['teams'] = dict()
    cfg['teams']['numteams'] = cfg_train['policy']['numteams']
    cfg['teams']['teamsize'] = cfg_train['policy']['teamsize']

    args.override_cfg_with_args(cfg, cfg_train)
    set_dependent_cfg_entries(cfg, cfg_train)

    # logdir = logdir_root +'/'+timestamp+'_no_dist_to_go_' + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    # logdir = logdir_root +'/'+timestamp + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
    logdir = logdir_root +'/eval/'+timestamp
    cfg["logdir"] = logdir

    eval()