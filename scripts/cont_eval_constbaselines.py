

from math import comb
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
#from dmaracing.utils.rating_helpers import compute_winprob_i
from dmaracing.utils.rl_helpers import get_mappo_runner
import trueskill
import sys
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import date, datetime
import time 
from dmaracing.controllers.purepursuit import PPController

def run_eval(policy, env :DmarEnv, FP_controller = None):
    obs, _ = env.reset()
        
    already_done = env.reset_buf > 1000
    eval_ep_rewards_tot = 0.*already_done
    eval_ep_rewards_team = 0.*already_done
    eval_ep_duration = 0.*already_done
    eval_ep_terminal_ranks = 0.*env.ranks
    
    #stat_tracking
    stats_num_overtakes_per_race = 0*env.stats_overtakes_last_race
    stats_num_collisions_per_race = 0*env.stats_overtakes_last_race
    stats_ego_car_leaves_track_fraction = 0*env.stats_overtakes_last_race
    stats_ado_car_leaves_track_fraction = 0*env.stats_overtakes_last_race
    stats_win_from_behind_fraction = 0*env.stats_overtakes_last_race
    stats_avg_lead_time_per_race = 0*env.stats_overtakes_last_race
    stats_percentage_sa_laptime_last_race = 0*env.stats_percentage_sa_laptime_last_race
    for ev_it in range(env.max_episode_length+1):
        actions = policy(obs)
        if FP_controller is not None:
            actions[:,1, :] = FP_controller.step()[:,1,:]
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        eval_ep_duration += ev_it*dones*(~already_done)
        eval_ep_terminal_ranks += env.ranks[:, :] * (~already_done)*dones
        eval_ep_rewards_tot += torch.sum(rewards[:,0,:], dim = 1).view(-1,1)*(~already_done)
        eval_ep_rewards_team += (rewards[:,0,1]).view(-1,1)*(~already_done)

        add_flag = (dones*(~already_done)).view(-1) 
        stats_num_overtakes_per_race += env.stats_overtakes_last_race * add_flag 
        stats_num_collisions_per_race += env.stats_num_collisions_last_race * add_flag 
        stats_ego_car_leaves_track_fraction += env.stats_ego_left_track_last_race * add_flag 
        stats_ado_car_leaves_track_fraction += env.stats_ado_left_track_last_race * add_flag 
        stats_win_from_behind_fraction += env.stats_win_from_behind_last_race * add_flag 
        stats_avg_lead_time_per_race += env.stats_lead_time_last_race * add_flag 
        stats_percentage_sa_laptime_last_race += env.stats_percentage_sa_laptime_last_race* add_flag 
  
        already_done |= dones
        if ~torch.any(~already_done):
            break
    
    stats_num_overtakes_per_race = torch.mean(stats_num_overtakes_per_race).item()
    stats_num_collisions_per_race = torch.mean(stats_num_collisions_per_race).item()
    stats_ego_car_leaves_track_fraction = torch.sum((stats_ego_car_leaves_track_fraction>(0.2/env.dt))).item()/env.num_envs
    stats_ado_car_leaves_track_fraction = torch.sum((stats_ado_car_leaves_track_fraction>(0.2/env.dt))).item()/env.num_envs
    num_start_from_behind = torch.sum(1.0*(stats_win_from_behind_fraction != 0)).item()
    stats_win_from_behind_fraction = torch.sum((stats_win_from_behind_fraction== 1)).item()/(num_start_from_behind + 1e-6)
    stats_avg_lead_time_per_race = torch.mean(stats_avg_lead_time_per_race).item()/(env.max_episode_length * env.dt)
    idx = torch.where(stats_percentage_sa_laptime_last_race< 10)[0]
    stats_avg_percentage_sa_laptime = torch.mean(stats_percentage_sa_laptime_last_race[idx]).item()
    stats = {'num_overtakes': stats_num_overtakes_per_race,
             'num_collisions': stats_num_collisions_per_race,
             'ego_offtrack': stats_ego_car_leaves_track_fraction,
             'ado_offtrack': stats_ado_car_leaves_track_fraction,
             'win_from_behind': stats_win_from_behind_fraction,
             'avg_time_lead_frac_race': stats_avg_lead_time_per_race,
             'percentage_sa_laptime': stats_avg_percentage_sa_laptime
             }
    return [eval_ep_duration, eval_ep_terminal_ranks, eval_ep_rewards_tot, stats]

def load_polices_det(combos, runner, env, it, opponents):
    #round robin policy loading    
    checkpoints = combos[it]
    model_paths = []
    for modelnr in checkpoints:
        if modelnr == 'ppc':
            model_paths.append(model_paths[-1])
        elif modelnr in opponents:
            model_paths.append("{}/{}.pt".format(logdir_adversaries, modelnr))
        else:    
            model_paths.append("{}/model_{}.pt".format(logdir, modelnr))
        print("Loading model" + model_paths[-1])
    policy_infos = runner.load_multi_path(model_paths)
    lumped_policy = runner.get_inference_policy(device=env.device)
    return lumped_policy, checkpoints

def updateratings(ratings, checkpoints, results, max_duration):
    ratings_match = [ratings[pt] for pt in checkpoints]
    duration, ranks, rewards, stats = results
    duration, ranks, rewards = duration.cpu().numpy(), ranks.cpu().numpy(), rewards.cpu().numpy()
    for dur, rank in zip(duration, ranks):
        update_ratio = 1.0 #dur/max_duration
        new_ratings = trueskill.rate(ratings_match, rank)
        for it, (old, new) in enumerate(zip(ratings_match, new_ratings)):
            mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
            sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
            ratings_match[it] = (trueskill.Rating(mu, sigma),)
    for idx, pt in enumerate(checkpoints):
        ratings[pt] = ratings_match[idx]
    return ratings, stats

def Log(logfile, ratings, stats, num_matches, checkpoints, it):
    log = (f"\033[1m EVAL: {it} \033[0m"
            f"""\n{'#' * 20}\n""")
    with open(logfile,'a') as fd:
        for key in ratings.keys():
            rat = ratings[key]
            mu = rat[0].mu
            sigma = rat[0].sigma
            # writer.add_scalar('EVAL/ratings_mean', mu, key)
            # writer.add_scalar('EVAL/ratings_sigma', sigma, key)
            # writer.add_scalar('EVAL/num_matches', num_matches[key], key)
            if key in checkpoints:
                stats_string = ''
                is_first = (checkpoints.index(key) == 0)
                
                for value in stats.values():
                    stats_string += str(value)+ ', ' if is_first else '-1, ' 
                log += (f"""{key:>{5}}{', m:'}{mu:.2f}{', s:'}{sigma:.2f}{', n:'}{num_matches[key]}\n""")
                line = str(-10 if isinstance(key, str) else key)+', '+str(mu) + ', ' + str(sigma) + ', ' + str(num_matches[key])+ ', ' + stats_string + '\n'
                
                fd.write(line)
            
        for key, val in stats.items():
            log += (f"""{key:>{5}}{': '}{val:.2f}\n""")   
    print(log)

def continuous_eval():
    #cfg['track']['num_tracks'] = 2
    env = DmarEnv(cfg, args)
    runner = get_mappo_runner(env, cfg_train, logdir_root, env.device, cfg['sim']['numAgents'])
    ppc = PPController(env,
                       lookahead_dist=1.5,
                       maxvel=2.5,
                       k_steer=1.0,
                       k_gas=2.0)
    #writer = SummaryWriter(log_dir=logdir, flush_secs=10)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logfile = logdir + '/continuouseval_'+timestamp+'.csv'
    ratings = {}
    num_matches = {}
    it = 0
    runs = [int(r[6:-3]) for r in os.listdir(logdir) if '.pt' in r]
    runs.sort()
    ops = ['ppc', 'rl_fast', 'sa_rl_slow', 'sa_rl_fast']
    num_its= len(runs)*len(ops)
    combos = []
    for ag0 in range(len(runs)):
        for ag1 in ops:
            combos.append([runs[ag0], ag1])
    perm = np.random.permutation(len(combos))
    combos = [combos[perm[idx]] for idx in range(len(perm))] 
    
    t_start = time.time()
    it_times = []

    for it in range(num_its):
        t_0 = time.time()
        lumped_policy, checkpoints = load_polices_det(combos, runner, env, it, ops)
        #lumped_policy, checkpoints = load_polices(logdir, runner, env)
        #get ratings of agents in eval loop
        #ratings_eval = []
        for chkpt in checkpoints:
            if chkpt not in ratings.keys():
                ratings[chkpt] = (trueskill.Rating(mu = 0),)
                num_matches[chkpt] = 1
            num_matches[chkpt] += 1
            #ratings_eval.append(ratings[chkpt])
        results = run_eval(lumped_policy, env, ppc if 'ppc' in checkpoints[1] else None)
        ratings, stats = updateratings(ratings, checkpoints, results, env.max_episode_length)
        Log(logfile, ratings, stats, num_matches, checkpoints, it)
        it +=1
        t_1 = time.time()
        it_times.append(t_1-t_0)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(t_1-t_start))
        rem = np.mean(it_times) * (num_its - it)
        rem = time.strftime("%H:%M:%S",time.gmtime(rem))
        print('it : '+str(it)+'/'+str(num_its)+'\n elapsed: '+ elapsed + '\n remaining: ' + rem)

if __name__ == "__main__":
    csv_key = ['checkpoint', 
                'mu', 
                'sigma', 
                'nmatchups', 
                'overtakes', 
                'collisions', 
                'ego offtrack', 
                'ado offtrack', 
                'win from behind', 
                'fraction of race led',
                'fraction of sa performance']
    print('eval key for csv')
    for k in csv_key:
        print(k)

    args = CmdLineArguments()
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless = True
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='_1v1')  # True)
    cfg['sim']['numEnv'] = 2048
    cfg['sim']['numAgents'] = 2
    cfg['sim']['collide'] = 1
    experiment_path = '22_11_02_09_06_07_col_1_ar_0.1_rr_1.0'
    if not args.headless:
        cfg['viewer']['logEvery'] = -1
    logdir = logdir_root +'/'+ experiment_path
    logdir_adversaries = os.getcwd()+'/logs/eval_adversaries'
    #load experiment cfgs
    pth_cfg = logdir + '/cfg.yml' 
    pth_cfg_train = logdir + '/cfg_train.yml'
    with open(pth_cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    with open(pth_cfg_train, 'r') as stream:
        cfg_train = yaml.safe_load(stream)
    
    cfg["logdir"] = logdir
    cfg['viewer']['logEvery'] = -1
    cfg['test'] = True #disable length randomizations only use mean of blr but still randomly draw maxvel vecs
    cfg['learn']['resetgrid'] = True
    #cfg['learn']['timeout'] = 20
    #cfg['learn']['resetrand'] = [0.0, 0.0, 0., 0.0, 0.0,  0., 0.0]
    cfg['learn']['agent_dropout_prob'] = 0.0
    cfg['learn']['obs_noise_lvl'] = 0.0 if cfg['test'] else cfg['learn']['obs_noise_lvl']
    cfg['trackRaceStatistics'] = True
    set_dependent_cfg_entries(cfg, cfg_train)

    continuous_eval()