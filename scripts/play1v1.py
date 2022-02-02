from math import log
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rl_helpers import get_mappo_runner
import numpy as np
import os
import time
#import trueskill
from scipy.stats import norm

def play():
    env = DmarEnv(cfg, args)
    #env.viewer.mark_env(0)
    obs = env.obs_buf
    model_paths = []
    modelnrs = []
    for run, chkpt in zip(runs, chkpts):
        dir, modelnr = get_run(logdir, run = run, chkpt=chkpt)
        modelnrs.append(modelnr)
        model_paths.append("{}/model_{}.pt".format(dir, modelnr))
        print("Loading model" + model_paths[-1])
    runner = get_mappo_runner(env, cfg_train, logdir, device = env.device)
    
    policy_infos = runner.load_multi_path(model_paths)
    policy = runner.get_inference_policy(device=env.device)
    
    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']

    num_races = 0
    num_agent_0_wins = 0
    skill_ag0 = [policy_infos[0]['trueskill']['mu'], policy_infos[0]['trueskill']['sigma']]
    skill_ag1 = [policy_infos[1]['trueskill']['mu'], policy_infos[1]['trueskill']['sigma']]
    #predicted win percentage
    #ratings = [trueskill.Rating(mu=skill_ag0), trueskill.Rating(mu=skill_ag1)]
    print("matchup trueskill: ag0: ", skill_ag0,', ag1: ', skill_ag0)
    mu_match = skill_ag0[0] - skill_ag1[0]
    var_match = skill_ag0[1]**2 + skill_ag1[1]**2
    win_prob = 1 - norm.cdf((0-mu_match)/(np.sqrt(2*var_match)))
    print("win probability agent 0: ", win_prob)
    idx = 0 
    while True:
        t1 = time.time()
        actions = policy(obs)
        #actions[:,1:,1] *=0.9
        obs,_, rew, dones, info = env.step(actions)

        
        dones_idx = torch.unique(torch.where(dones)[0])
        if len(dones_idx):
            num_races += len(dones_idx)
            num_agent_0_wins += torch.sum(info['ranking'][:,1], dim = 0).item()
        if idx %300 ==0:
            print("wins_0 / races: ", num_agent_0_wins, '/', num_races, '=', num_agent_0_wins*1.0/(num_races+0.001))
        obsnp = obs[:,0,:].cpu().numpy()
        rewnp = rew[:,0].cpu().numpy()
        cont = env.conturing_err.cpu().numpy()
        act = actions[:,0,:].cpu().detach().numpy()
        states = env.states.cpu().numpy()
        om_mean = np.mean(states[env.viewer.env_idx_render,0, env.vn['S_W0']:env.vn['S_W3'] +1 ])

        viewermsg = [(f"""{'p0 '+str(modelnrs[0])}{' ts: '}{policy_infos[0]['trueskill']['mu']:.1f}"""),
                     (f"""{'p1 '+str(modelnrs[1])}{' ts: '}{policy_infos[1]['trueskill']['mu']:.1f}"""),
                     (f"""{'Win prob p0 : ':>{10}}{win_prob:.3f}"""),
                     (f"""{'rewards:':>{10}}{' '}{100*rewnp[env.viewer.env_idx_render]:.2f}"""   ),
                     (f"""{'velocity x:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 0]:.2f}"""),
                     #(f"""{'velocity y:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 1]:.2f}"""),
                     #(f"""{'ang vel:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 2]:.2f}"""),
                     #(f"""{'steer:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_STEER']]:.2f}"""),
                     #(f"""{'gas:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_GAS']]:.2f}"""),
                     #(f"""{'brake:':>{10}}{' '}{act[env.viewer.env_idx_render, env.vn['A_BRAKE']]:.2f}"""),
                     (f"""{'om_mean:':>{10}}{' '}{om_mean:.2f}"""),
                     (f"""{'collision:':>{10}}{' '}{env.is_collision[0,0].item():.2f}"""),
                     (f"""{'rank ag 0 :':>{10}}{' '}{1+env.ranks[env.viewer.env_idx_render, 0].item():.2f}"""),
                     (f"""{'laps ag 0 :':>{10}}{' '}{env.lap_counter[env.viewer.env_idx_render, 0].item():.2f}"""),
                     (f"""{'step :':>{10}}{' '}{env.episode_length_buf[env.viewer.env_idx_render].item():.2f}""")]

        #env.viewer.clear_markers()
        
        #closest_point_marker = env.interpolated_centers[env.viewer.env_idx_render, 0, :, :].cpu().numpy()
        #env.viewer.add_point(closest_point_marker, 2,(222,10,0), 2)

        env.viewer.clear_string()
        for msg in viewermsg:
            env.viewer.add_string(msg)

        idx +=1
        
        evt = env.viewer_events

        if evt == 121:
            print("env ", env.viewer.env_idx_render, " reset")
            env.episode_length_buf[env.viewer.env_idx_render] = 1e9
        
        #t2 = time.time()
        #realtime = t2-t1-time_per_step
        #if realtime < 0:
        #     time.sleep(-realtime)

if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    args.test = True
    path_cfg = os.getcwd() + '/cfg'

    cfg, cfg_train, logdir = getcfg(path_cfg)

    chkpts = [-1, -1]
    runs = [-1, -2]
    cfg['sim']['numEnv'] = 1
    cfg['sim']['numAgents'] = 2
    cfg['learn']['timeout'] = 300
    cfg['learn']['offtrack_reset'] = 4.0
    cfg['learn']['reset_tile_rand'] = 5
    cfg['sim']['test_mode'] = True
    
    cfg['track']['seed'] = 12
    cfg['track']['num_tracks'] = 20

    #cfg['track']['CHECKPOINTS'] = 3
    #cfg['track']['TRACK_RAD'] = 800
    cfg['viewer']['multiagent'] = True

    set_dependent_cfg_entries(cfg)

    play()    
