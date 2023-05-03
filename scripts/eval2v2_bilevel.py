from dmaracing.utils.rl_helpers import get_bima_ppo_runner
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
import shutil
import numpy as np
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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


    # save_dir = 'logs/saved_models/'+timestamp
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    #     os.mkdir(save_dir+'/hl_model')
    #     os.mkdir(save_dir+'/ll_model')
    # shutil.copy(logdir_root+'/'+timestamp+'/cfg.yml', save_dir+'/cfg.yml' )
    # shutil.copy(logdir_root+'/'+timestamp+'/cfg_train.yml', save_dir+'/cfg_train.yml' )
    
    # # ### Save jit models to original folder
    # if hasattr(runner.alg_hl.actor_critic.teamacs[0].ac, 'actor'):
    #     policy_hl_jit = torch.jit.script(runner.alg_hl.actor_critic.teamacs[0].ac.actor.to('cpu'))
    #     policy_hl_jit.save(save_dir+ "/hl_model/jit_model_" +str(modelnr_hl)+".pt")
    # else:
    #     policy_hl_jit = torch.jit.script(runner.alg_hl.actor_critic.teamacs[0].ac.critic.to('cpu'))
    #     policy_hl_jit.save(save_dir+ "/hl_model/jit_critic_model_" +str(modelnr_hl)+".pt")

    # policy_ll_jit = torch.jit.script(runner.alg_ll.actor_critic.teamacs[0].ac.actor.to('cpu'))
    # policy_ll_jit.save(save_dir + "/ll_model/jit_model_" +str(modelnr_ll)+".pt")

    # policy_ll_jit = torch.jit.script(runner.alg_ll.actor_critic.teamacs[0].ac.actor.to('cpu'))
    # policy_ll_jit.save(save_dir + "/ll_model/jit_model_" +str(modelnr_ll)+".pt")
    # exit()
    
    # #populate adversary buffer
    # adv_model_paths_hl, adv_model_paths_ll = [], []
    # for run_hl, chkpt_hl, run_ll, chkpt_ll in zip(runs_hl, chkpts_hl, runs_ll, chkpts_ll):

    #     dir_hl, modelnr_hl = get_run(logdir_root, run=run_hl + '/hl_model', chkpt=chkpt_hl)
    #     adv_model_paths_hl.append("{}/model_{}.pt".format(dir_hl, modelnr_hl))
    #     print("Loading HL model" + adv_model_paths_hl[-1])
    #     dir_ll, modelnr_ll = get_run(logdir_root, run=run_ll + '/ll_model', chkpt=chkpt_ll)
    #     adv_model_paths_ll.append("{}/model_{}.pt".format(dir_ll, modelnr_ll))
    #     print("Loading LL model" + adv_model_paths_ll[-1])

    # runner.populate_adversary_buffer(adv_model_paths_hl, adv_model_paths_ll)

    policy_eval_hl = runner.get_inference_policy_hl(device=env.device)
    policy_eval_ll = runner.get_inference_policy_ll(device=env.device)

    # run_eval(env, policy_eval_hl, policy_eval_ll)
    # run_eval_sensitivity(env, policy_eval_hl, policy_eval_ll)
    return run_eval_winrate(env, policy_eval_hl, policy_eval_ll)


def run_eval(env, policy_hl, policy_ll):

    num_episodes = 5

    max_hl_steps = env.max_episode_length // env.dt_hl
    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']
    for _ in range(num_episodes):

        obs, _ = env.reset()

        already_done = env.reset_buf > 1000

        for i_hl in range(max_hl_steps):
            actions_hl_raw = policy_hl(obs)
            # reward_ep_ll = torch.zeros(self.env.num_envs, self.num_agents, 1, dtype=torch.float, device=self.device)
            for i_ll in range(env.dt_hl):
                actions_hl = env.project_into_track_frame(actions_hl_raw)
                t1 = time.time()
                obs_ll = torch.concat((obs, actions_hl), dim=-1)
                actions_ll = policy_ll(obs_ll)

                # actions_ll = actions_ll.view((*actions_ll.shape[:-1], 2, 2))

                obs, privileged_obs, rewards, dones, infos = env.step(actions_ll)
                viewermsg = [ (f"""{'vbax:':>{10}}{' '}{env.states[0,0,3].item():.2f}"""),
                             (f"""{'vcmd:':>{10}}{' '}{env.cfg['learn']['actionscale'][1]*actions_ll[0,0,1].item() + env.cfg['learn']['defaultactions'][1]:.2f}"""),
                             ]
                env.viewer.clear_string()
                for msg in viewermsg:
                    env.viewer.add_string(msg)
                env.viewer.x_offset = int(-env.viewer.width/env.viewer.scale_x*env.states[env.viewer.env_idx_render, 0, 0])
                env.viewer.y_offset = int(env.viewer.height/env.viewer.scale_y*env.states[env.viewer.env_idx_render, 0, 1])
                env.viewer.draw_track()
                evt = env.viewer_events
                if evt == 121:
                    print("env ", env.viewer.env_idx_render, " reset")
                    env.episode_length_buf[env.viewer.env_idx_render] = 1e9
                t2 = time.time()
                #print('dt ', t2-t1)
                realtime = t2-t1-time_per_step
                if realtime < 0:
                    time.sleep(-realtime)
            already_done |= dones
            if ~torch.any(~already_done):
                break
    

def run_eval_winrate(env, policy_hl, policy_ll):

    num_episodes = 10  # 10
    teamranks_all = 0
    wincounts_all = 0

    max_hl_steps = env.max_episode_length // env.dt_hl

    for ep in range(num_episodes):

        total_steps = 0

        obs, _ = env.reset()
        env.log_episode = True
        env._global_step = ep

        # env.dyn_model.max_vel_vec = 3.0 + 0.0*env.dyn_model.max_vel_vec
        # env.dyn_model.max_vel_vec[:, 2:] = 2.0

        already_done = env.reset_buf > 1000

        for i_hl in range(max_hl_steps):
            actions_hl_raw = policy_hl(obs)
            # reward_ep_ll = torch.zeros(self.env.num_envs, self.num_agents, 1, dtype=torch.float, device=self.device)
            for i_ll in range(env.dt_hl):
                actions_hl = env.project_into_track_frame(actions_hl_raw)
                obs_ll = torch.concat((obs, actions_hl), dim=-1)
                actions_ll = policy_ll(obs_ll)

                # actions_ll = actions_ll.view((*actions_ll.shape[:-1], 2, 2))
                prev_teamrank = obs[:, 0, -22]

                obs, privileged_obs, rewards, dones, infos = env.step(actions_ll)

                total_steps += 1

                # # Viewer
                # viewermsg = [ (f"""{'vby:':>{10}}{' '}{10.0*obs[0,0,1].item():.2f}"""),]
                # env.viewer.clear_string()
                # for msg in viewermsg:
                #     env.viewer.add_string(msg)
                
                already_done |= dones
                if ~torch.any(~already_done):
                    break
        teamranks_all += prev_teamrank.sum().cpu().item()  # obs[:, 0, -22].sum().cpu().item()
        wincounts_all += (prev_teamrank==0).sum().cpu().item()

        winrate = np.round(wincounts_all / (obs.shape[0] * (ep+1)), 3)

        print('Completed ' + str(obs.shape[0] * (ep+1)) + '/' + str(obs.shape[0] * num_episodes) + ' episodes: avg team rank = ' + str(np.round(teamranks_all / (obs.shape[0] * (ep+1)), 3)))
        print('Completed ' + str(obs.shape[0] * (ep+1)) + '/' + str(obs.shape[0] * num_episodes) + ' episodes: avg win rate = ' + str(winrate))

    return winrate


def run_eval_sensitivity(env, policy_hl, policy_ll):


    max_steps = 60

    obs, _ = env.reset()

    already_done = env.reset_buf > 1000

    for i_hl in range(max_steps//env.dt_hl+1):
        actions_hl_raw = policy_hl(obs)
        # reward_ep_ll = torch.zeros(self.env.num_envs, self.num_agents, 1, dtype=torch.float, device=self.device)
        for i_ll in range(env.dt_hl):
            if i_hl*env.dt_hl+i_ll>max_steps:
                break
            actions_hl = env.project_into_track_frame(actions_hl_raw)
            obs_ll = torch.concat((obs, actions_hl), dim=-1)
            actions_ll = policy_ll(obs_ll)

            # actions_ll = actions_ll.view((*actions_ll.shape[:-1], 2, 2))

            obs, privileged_obs, rewards, dones, infos = env.step(actions_ll)
    
    ado_obs_ids = [
        {'name': 'Prog', 'id': -7, 'min': -3.0 * 0.10, 'max': +3.0 * 0.10},
        {'name': 'Cerr', 'id': -6, 'min': -0.7 * 0.25, 'max': +0.7 * 0.25},
        {'name': 'RotS', 'id': -5, 'min': -1.0 * 1.00, 'max': +1.0 * 1.00},
        # 'Rotation Cos': -4,
        {'name': 'VelX', 'id': -3, 'min': -3.0 * 0.10, 'max': +3.0 * 0.10},
        {'name': 'VelY', 'id': -2, 'min': -3.0 * 0.10, 'max': +3.0 * 0.10},
        {'name': 'VelA', 'id': -1, 'min': -3.0 * 0.10, 'max': +3.0 * 0.10},
    ]
    num_ado = 3
    steps = 50

    for elem in ado_obs_ids:
        start = num_ado * elem['id']

        vals = torch.linspace(elem['min'], elem['max'], steps)

        for idx in range(num_ado):

            actions_ll_all = []
            for val in vals:
                observation = obs[:, 0].clone().unsqueeze(dim=1)
                observation[..., start+idx] = val

                if elem['name']=='RotS':
                    observation[..., start+idx+num_ado] = torch.cos(torch.asin(val))

                observation = observation.repeat(1, 4, 1)

                actions_hl_raw = policy_hl(observation)
                actions_hl = env.project_into_track_frame(actions_hl_raw)
                observation_ll = torch.concat((observation, actions_hl), dim=-1)
                actions_ll = policy_ll(observation_ll)

                actions_ll_all.append(actions_ll[:, 0])

            actions_ll_all = torch.stack(actions_ll_all, dim=-1)
            actions_ll_std = actions_ll_all.std(dim=-1)
            actions_ll_rng = actions_ll_all.amax(dim=-1) - actions_ll_all.amin(dim=-1)
            actions_ll_mean = actions_ll_all.mean(dim=-1)

            act_ll_std_mean = np.round(actions_ll_std.mean(dim=0).cpu().numpy(), 3)
            act_ll_rng_mean = np.round(actions_ll_rng.mean(dim=0).cpu().numpy(), 3)
            act_ll_mean_mean = np.round(actions_ll_mean.mean(dim=0).cpu().numpy(), 3)
            print(elem['name'] + ' ' + str(idx) + ':     mean std = ' + str(act_ll_std_mean) + '   ;   mean range = ' + str(act_ll_rng_mean) + '   ;   mean mean = ' + str(act_ll_mean_mean))

            pass

    pass
            


if __name__ == "__main__":
    args = CmdLineArguments()
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless =  True 


    checkpoints_to_play = [1200]

    winrate_results = []
    for checkpoint_opp in checkpoints_to_play:

      # ### Run information
      exp_name = 'tri_2v2_vhc_rear'  # 'tri_single_blr_hierarchical'
      timestamp = '23_04_30_19_35_04_bilevel_2v2'  #'23_03_23_11_34_55_bilevel_2v2'  # '23_03_20_19_06_44_bilevel_2v2'  # '23_02_21_17_16_07_bilevel_2v2'  # '23_01_31_14_30_58_bilevel_2v2'  # '23_01_31_11_54_24_bilevel_2v2'
      checkpoint = 1200  # 500  # 1300
      #active policies
      runs_hl = [timestamp, '23_04_30_19_35_04_bilevel_2v2']  # '23_02_22_21_18_03_bilevel_2v2'
      chkpts_hl = [checkpoint, checkpoint_opp]
      runs_ll = [timestamp, '23_04_30_19_35_04_bilevel_2v2']
      chkpts_ll = [checkpoint, checkpoint_opp]
      ##policies to populate adversary buffer
      adv_runs = ['23_04_30_19_35_04_bilevel_2v2']
      adv_chkpts = [checkpoint]


      path_cfg = os.getcwd() + '/logs/' + exp_name + '/' + runs_hl[0]
      cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='', postfix_train='')
      cfg['viewer']['logEvery'] = 1  # -1
      cfg['sim']['numEnv'] = 1  # 100

      # cfg_train['policy']['numteams'] = 2
      # cfg_train['policy']['teamsize'] = 2
      # cfg['learn']['agent_dropout_prob'] = 0.0
      cfg['learn']['agent_dropout_prob_val_ini'] = 0.0
      cfg['learn']['agent_dropout_prob_val_end'] = 0.0
      cfg['learn']['steer_ado_with_PPC'] = True  # True  # False
      cfg['learn']['ppc_prob_val_ini'] = 1.0  # 0.0
      cfg['learn']['ppc_prob_val_end'] = 1.0  # 0.0
      cfg['learn']['resetrand'] = [0.05, 0.05, 0.5, 2.0, 0.01,  0.5, 0.0]  # [0.05, 0.05, 0.2, 0.1, 0.01, 0.1, 0.0]
      cfg["sim"]["reset_timeout_only"] = True
      cfg["sim"]["filtercollisionstoego"] = False
      # cfg['model']['max_vel'] = 3.5
      cfg['model']['vm_noise_scale_ego'] = 0.1
      cfg['model']['vm_noise_scale_ado'] = 0.1
      cfg["track"]["track_half_width"] = 0.6  # 0.63
      cfg["learn"]["timeout"] = 30.0  # 16.0

      cfg['teams'] = dict()
      cfg['teams']['numteams'] = cfg_train['policy']['numteams']
      cfg['teams']['teamsize'] = cfg_train['policy']['teamsize']

      if not "centralized_value_hl" in cfg_train["runner"]:
          cfg_train["runner"]["centralized_value_hl"] = True
          cfg_train["runner"]["centralized_value_ll"] = False

      args.override_cfg_with_args(cfg, cfg_train)
      set_dependent_cfg_entries(cfg, cfg_train)

      # logdir = logdir_root +'/'+timestamp+'_no_dist_to_go_' + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
      # logdir = logdir_root +'/'+timestamp + cfg_train['runner']['algorithm_class_name']+'_'+str(cfg_train['runner']['num_steps_per_env'])
      logdir = logdir_root +'/eval/'+timestamp
      cfg["logdir"] = logdir

      winrate_result = eval()

      winrate_results.append(winrate_result)

    pass
