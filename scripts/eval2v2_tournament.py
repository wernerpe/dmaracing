from dmaracing.utils.rl_helpers import get_bima_ppo_runner
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def policy_loader(runner, agent_paths, agent_checkpoints,  device):
    runs_hl = agent_paths
    runs_ll = agent_paths
    chkpts_hl = agent_checkpoints 
    chkpts_ll = agent_checkpoints 
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
    
    policy_eval_hl = runner.get_inference_policy_hl(device=device)
    policy_eval_ll = runner.get_inference_policy_ll(device=device)

    return [policy_eval_hl, policy_eval_ll]

def get_matches(paths_agents):
    matchups = []
    for idx1, ag1 in enumerate(paths_agents[:-1]):
        for ag2 in paths_agents[idx1+1:]:
            if ag1 != ag2:
                matchups.append([ag1, ag2])
    return matchups


def run_eval(env:DmarEnvBilevel, lumped_policy):

    lumped_pol_hl = lumped_policy[0]
    lumped_pol_ll = lumped_policy[1]

    obs = env.reset_eval()
    already_done = env.reset_buf > 1000
    eval_ep_terminal_ranks = 0.*env.ranks
    eval_ep_duration = 0.*already_done
    
    for ev_it in range(env.max_episode_length+1):
        if ev_it % env.cfg['learn']['episode_length_ll'] ==0:
            hl_actions = lumped_pol_hl(obs)
        hl_act_proj = env.project_into_track_frame(hl_actions)
        obs_ll = torch.cat((obs, hl_act_proj), dim = -1)
        actions = lumped_pol_ll(obs_ll)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        eval_ep_duration += ev_it*dones*(~already_done)

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
    exp_name = 'tri_multiagent_blr_hierarchical'  # 'tri_single_blr_hierarchical'
    timestamp = '23_02_02_11_14_46_bilevel_2v2'  # '23_01_31_14_30_58_bilevel_2v2'  # '23_01_31_11_54_24_bilevel_2v2'
    checkpoint = 500  # 500  # 1300
    
     
    #agents 
    agent_names = ['bi_cen', 'bi_dec', 'll_cent', 'll_dec']
    agent_paths = ['23_02_02_11_14_46_bilevel_2v2']*4
    agent_checkpoints = [200]*4
    
    
    path_cfg = os.getcwd() + '/logs/' + exp_name + '/' + agent_paths[0]
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='', postfix_train='')
    cfg['viewer']['logEvery'] = -1
    cfg['sim']['numEnv'] = 1

    cfg['learn']['agent_dropout_prob'] = 0.0
    cfg["sim"]["reset_timeout_only"] = True

    cfg['teams'] = dict()
    cfg['teams']['numteams'] = cfg_train['policy']['numteams']
    cfg['teams']['teamsize'] = cfg_train['policy']['teamsize']

    args.override_cfg_with_args(cfg, cfg_train)
    set_dependent_cfg_entries(cfg, cfg_train)

    logdir = logdir_root +'/eval/'+timestamp
    cfg["logdir"] = logdir

    env = DmarEnvBilevel(cfg, args)
    
    matches = get_matches(agent_paths)
    
    for match in matches:
        #get policies
        runner = get_bima_ppo_runner(env, cfg_train, logdir, args.device)
        policy_bundle = policy_loader(runner, match)
        eval_matchup()
    
    do_tournament(env, policy_bundels)

    eval()