

import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
#from dmaracing.utils.rating_helpers import compute_winprob_i
from dmaracing.utils.rl_helpers import get_mappo_runner
import sys
import trueskill
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import date, datetime

def run_eval(policy, env):
    obs, _ = env.reset()
        
    already_done = env.reset_buf > 1000
    eval_ep_rewards_tot = 0.*already_done
    eval_ep_rewards_team = 0.*already_done
    eval_ep_duration = 0.*already_done
    eval_ep_terminal_ranks = 0.*env.ranks
        
    for ev_it in range(env.max_episode_length+1):
        actions = policy(obs)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        eval_ep_duration += ev_it*dones*(~already_done)
        eval_ep_terminal_ranks += env.ranks[:, :] * (~already_done)*dones
        eval_ep_rewards_tot += torch.sum(rewards[:,0,:], dim = 1).view(-1,1)*(~already_done)
        eval_ep_rewards_team += (rewards[:,0,1]).view(-1,1)*(~already_done)
        already_done |= dones
        if ~torch.any(~already_done):
            break
    return [eval_ep_duration, eval_ep_terminal_ranks, eval_ep_rewards_tot]
        
def load_polices(logdir, runner, env):
    lumped_policy = None
    runs = [int(r[6:-3]) for r in os.listdir(logdir)[::10] if '.pt' in r]  
    runs.sort()

    #random choice    
    checkpoints = [np.random.choice(runs) for _ in range(env.num_agents)]
    model_paths = []
    for modelnr in checkpoints:
        model_paths.append("{}/model_{}.pt".format(logdir, modelnr))
        print("Loading model" + model_paths[-1])
    policy_infos = runner.load_multi_path(model_paths)
    lumped_policy = runner.get_inference_policy(device=env.device)
    return lumped_policy, checkpoints

def updateratings(ratings, checkpoints, results, max_duration):
    ratings_match = [ratings[pt] for pt in checkpoints]
    duration, ranks, rewards = results
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
    return ratings

def Log(logfile, ratings, num_matches, checkpoints, it):
    log = (f"\033[1m EVAL: {it} \033[0m"
            f"""\n{'#' * 20}\n""")
    with open(logfile,'a') as fd:
        for key in ratings.keys():
            rat = ratings[key]
            mu = rat[0].mu
            sigma = rat[0].sigma
            log += (f"""{key:>{5}}{', m:'}{mu:.2f}{', s:'}{sigma:.2f}{', n:'}{num_matches[key]}\n""")
            # writer.add_scalar('EVAL/ratings_mean', mu, key)
            # writer.add_scalar('EVAL/ratings_sigma', sigma, key)
            # writer.add_scalar('EVAL/num_matches', num_matches[key], key)
            line = str(key)+', '+str(mu) + ', ' + str(sigma) + ', ' + str(num_matches[key])+'\n'
            fd.write(line)   
    print(log)

def continuous_eval():
    cfg['sim']['numEnv'] = 128
    #cfg['track']['num_tracks'] = 2
    env = DmarEnv(cfg, args)
    runner = get_mappo_runner(env, cfg_train, logdir_root, env.device, cfg['sim']['numAgents'])
    #writer = SummaryWriter(log_dir=logdir, flush_secs=10)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logfile = logdir + '/continuouseval_'+timestamp+'.csv'
    ratings = {}
    num_matches = {}
    it = 0
    while True:
        lumped_policy, checkpoints = load_polices(logdir, runner, env)
        #get ratings of agents in eval loop
        #ratings_eval = []
        for chkpt in checkpoints:
            if chkpt not in ratings.keys():
                ratings[chkpt] = (trueskill.Rating(mu = 0),)
                num_matches[chkpt] = 1
            num_matches[chkpt] += 1
            #ratings_eval.append(ratings[chkpt])
        results = run_eval(lumped_policy, env)
        ratings = updateratings(ratings, checkpoints, results, env.max_episode_length)
        Log(logfile, ratings, num_matches, checkpoints, it)
        it +=1

if __name__ == "__main__":
    args = CmdLineArguments()
    args.parse(sys.argv[1:])
    args.device = 'cuda:0'
    args.headless = False
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir_root = getcfg(path_cfg, postfix='_1v1')  # True)
    cfg['sim']['numAgents'] = 2
    cfg['sim']['collide'] = 1
    experiment_path = '22_08_23_11_49_39'
    if not args.headless:
        cfg['viewer']['logEvery'] = -1
    logdir = logdir_root +'/'+ experiment_path
    
    #load experiment cfgs
    pth_cfg = logdir + '/cfg.yml' 
    pth_cfg_train = logdir + '/cfg_train.yml'

    with open(pth_cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    with open(pth_cfg_train, 'r') as stream:
        cfg_train = yaml.safe_load(stream)
    
    cfg["logdir"] = logdir
    cfg['viewer']['logEvery'] = -1
    cfg['test'] = True
    continuous_eval()