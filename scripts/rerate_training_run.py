from math import log
from tabnanny import check
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rating_helpers import compute_winprob_i
from dmaracing.utils.rl_helpers import get_mappo_runner
import numpy as np
import os
import trueskill
import matplotlib.pyplot as plt

def evaluate_training_run(logdir, run, downsampling = 10, steps = 10):
    env = DmarEnv(cfg, args)
    model_dir, checkpointsr = get_all_chckpts_run(logdir, run)
    checkpoints = np.array([500, 5000, 20000, 50000])#np.array(checkpointsr[::14] + [checkpointsr[-1]]) 
    ratings = {}
    for chckpt in checkpoints:
        ratings[chckpt] = (trueskill.Rating(mu=0),)
    
    for step in range(steps):
        #select agent checkpoints
        ags = np.random.choice(len(checkpoints), 4)
        print('step: ', step+1,' / ', steps)
        print('Loading training checkpoints:', checkpoints[ags].tolist())
        ratings = evaluate_agents(ratings, checkpoints[ags].tolist(), run, env)
        print(ratings)
    #plot result
    fig, ax = plt.subplots(nrows=1, ncols=1)
    mus = []
    sigmas = []
    checkpoint_plot = []
    for key in ratings.keys():
        rat = ratings[key]
        mu = rat[0].mu
        sigma = rat[0].sigma
        if mu>0:
            mus.append(mu)
            sigmas.append(sigma)
            checkpoint_plot.append(key)
    #ax.clear()
    ax.plot(np.array(checkpoint_plot), np.array(mus))
    #ax.legend()
    ax.set_ylabel('trueskill')
    ax.set_xlabel('training step')
    plt.show()    
    print('done')

def evaluate_agents(ratings, agent_checkpoints, run, env):
    active_ratings = [ratings[checkpt] for checkpt in agent_checkpoints]
    #load policies
    model_paths = []
    modelnrs = []
    for run, chkpt in zip([run]*4, agent_checkpoints):
        dir, modelnr = get_run(logdir, run = run, chkpt=chkpt)
        modelnrs.append(modelnr)
        model_paths.append("{}/model_{}.pt".format(dir, modelnr))
        print("Loading model" + model_paths[-1])
    runner = get_mappo_runner(env, cfg_train, logdir, env.device, cfg['sim']['numAgents'])
    
    policy_infos = runner.load_multi_path(model_paths)
    policy = runner.get_inference_policy(device=env.device)
    active_ratings = race(env, policy, active_ratings, dur = 1202)
    for it, checkpt in enumerate(agent_checkpoints):
        ratings[checkpt] = active_ratings[it] 
    return ratings

def race(env, policy, active_ratings, dur = 1000):
    env.reset()
    obs = env.obs_buf
    for step in range(dur):
        actions = policy(obs)
        obs, _, rew, dones, info = env.step(actions)
        if 'ranking' in info:
            ranks = info['ranking'][0].cpu().numpy()
            update_ratio = info['ranking'][1]
            new_ratings = trueskill.rate(active_ratings, ranks)
            for old, new, it in zip(active_ratings, new_ratings, range(len(active_ratings))):
                mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
                sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
                active_ratings[it] = (trueskill.Rating(mu, sigma),)
        if step%400 == 0:
            env.episode_length_buf[:] = 1e9
    return active_ratings

if __name__ == "__main__":
    args = CmdLineArguments()
    SOUND = False
    args.device = 'cuda:0'
    args.headless = True 
    #args.test = True
    path_cfg = os.getcwd() + '/cfg'

    cfg, cfg_train, logdir = getcfg(path_cfg)
    cfg['sim']['numEnv'] = 4096
    cfg['sim']['numAgents'] = 4
    cfg['sim']['collide'] = 1
    
    cfg['learn']['resetgrid'] = True
    cfg['learn']['reset_tile_rand'] = 40
    cfg['viewer']['logEvery'] = -1
    cfg['track']['seed'] = 5
    cfg['track']['num_tracks'] = 30
    cfg['viewer']['multiagent'] = True

    set_dependent_cfg_entries(cfg)
    evaluate_training_run(logdir, run = -1, downsampling = 80, steps = 10)  

    #onlyfiles = [f for f in os.listdir(logdir) if os.isfile(os.join(logdir, f))]
    print('done')
    #play()    
