from asyncio import runners
from math import log
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rl_helpers import get_mappo_runner
import numpy as np
import os
import time
import matplotlib.pyplot as plt
#import trueskill
from scipy.stats import norm

def play():
    chkpts = [-1, 1250]
    
    cfg['sim']['numEnv'] = 2
    cfg['sim']['numAgents'] = 2
    cfg['learn']['timeout'] = 40
    cfg['learn']['offtrack_reset'] = 3.0
    cfg['learn']['reset_tile_rand'] = 5
    cfg['sim']['test_mode'] = True
    
    cfg['track']['seed'] = 12
    cfg['track']['num_tracks'] = 1

    #cfg['track']['CHECKPOINTS'] = 3
    #cfg['track']['TRACK_RAD'] = 800
    cfg['viewer']['multiagent'] = True
    np.random.seed(0)
    env = DmarEnv(cfg, args)
    #env.viewer.mark_env(0)
    obs = env.obs_buf
    model_paths = []
    modelnrs = []
    for chkpt in chkpts:
        dir, modelnr = get_run(logdir, run = "22_01_19_19_26_15", chkpt=chkpt)
        modelnrs.append(modelnr)
        model_paths.append("{}/model_{}.pt".format(dir, modelnr))
        print("Loading model" + model_paths[-1])
    runner = get_mappo_runner(env, cfg_train, logdir, device = env.device)
    
    policy_infos = runner.load_multi_path(model_paths)
    policy = runner.get_inference_policy(device = env.device)
    value_fn = runner.get_value_functions(device = env.device)
    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']

    #match statistics
    #num_races = 0
    #num_agent_0_wins = 0
    #skill_ag0 = [policy_infos[0]['trueskill']['mu'], policy_infos[0]['trueskill']['sigma']]
    #skill_ag1 = [policy_infos[1]['trueskill']['mu'], policy_infos[1]['trueskill']['sigma']]
    #
    ##predicted win percentage
    ##ratings = [trueskill.Rating(mu=skill_ag0), trueskill.Rating(mu=skill_ag1)]
    #print("matchup trueskill: ag0: ", skill_ag0,', ag1: ', skill_ag0)
    #mu_match = skill_ag0[0] - skill_ag1[0]
    #var_match = skill_ag0[1]**2 + skill_ag1[1]**2
    #win_prob = 1 - norm.cdf((0-mu_match)/(np.sqrt(2*var_match)))
    #print("win probability agent 0: ", win_prob)
    
    #define track starting tile for env 0 
    #tile_idx = 128 #tile_idx_env[:, agent]  
    #set agent 1 off to side, vay agent 0 along centerline
    tile_idx_1 = 80
    #tile_idx = (torch.rand((1,1), device=env.device) * env.active_track_tile_counts[0].view(-1,1)).to(dtype=torch.long)
    startpos = env.active_centerlines[0, tile_idx_1, :]
    angle = env.active_alphas[0, tile_idx_1]
    env.states[0, 1,env.vn['S_X']] = startpos[0]
    env.states[0, 1,env.vn['S_Y']] = startpos[1] - 2.0         
    env.states[0, 1,env.vn['S_THETA']] = angle.item()

    values_pairs = []
    for idx in np.arange(-10, 11):
        tile_idx = tile_idx_1 + idx#(torch.rand((1,1), device=env.device) * 0.22*env.active_track_tile_counts[0].view(-1,1)+30).to(dtype=torch.long)
        startpos = env.active_centerlines[0, tile_idx, :]
        angle = env.active_alphas[0, tile_idx]
        env.states[0,0,env.vn['S_X']] = startpos[0]
        env.states[0,0,env.vn['S_Y']] = startpos[1]          
        env.states[0,0,env.vn['S_THETA']] = angle.item()

        
        dists = torch.norm(env.states[:, :, 0:2].unsqueeze(2)-env.active_centerlines.unsqueeze(1), dim=3)
        sort = torch.sort(dists, dim = 2)
        env.dist_active_track_tile = sort[0][:,:, 0]
        env.active_track_tile = sort[1][:,:, 0]

        env.compute_observations()
        observations = env.obs_buf
        values = value_fn(observations)
        values_pairs.append([idx, values[0, :,0].detach().cpu().numpy()])
        print(values)

        lookahead_markers = env.lookahead + torch.tile(env.states[:,:,0:2].unsqueeze(2), (1,1,env.horizon, 1))
        pts = lookahead_markers[env.viewer.env_idx_render,0,:,:].cpu().numpy()
        #print(lookahead_markers[0,0,0:5,:])
        env.viewer.clear_markers()
        env.viewer.add_point(pts, 5,(5,10,222))
        env.render()
        time.sleep(0.01)

    pos = np.array([entry[0] for entry in values_pairs])
    values_0 = np.array([entry[1][0] for entry in values_pairs])
    values_1 = np.array([entry[1][1] for entry in values_pairs])

    fig, ax= plt.subplots(nrows = 1, ncols = 1)
    ax.plot(pos, values_0, c = 'b', label = 'value prediction agent0')
    ax.plot(pos, values_1, c = 'r', label = 'value prediction agent1')
    ax.set_title('varying starting position of agent 0')
    ax.set_xlabel('position [tile]')
    ax.set_ylabel('value fn prediction')
    ax.legend()
    

if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    args.test = True
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    play()    
