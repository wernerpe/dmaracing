from math import log
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rating_helpers import compute_winprob_i
from dmaracing.utils.rl_helpers import get_mappo_runner
import numpy as np
import os
import time
import playsound
import threading
import trueskill
import matplotlib.pyplot as plt


def play():
    start = [0]*4
    ratings = [(trueskill.Rating(mu = s, sigma=1.0),) for s in start]

    env = DmarEnv(cfg, args)
    obs = env.obs_buf
    model_paths = []
    modelnrs = []
    for run, chkpt in zip(runs, chkpts):
        dir, modelnr = get_run(logdir, run = run, chkpt=chkpt)
        modelnrs.append(modelnr)
        model_paths.append("{}/model_{}.pt".format(dir, modelnr))
        print("Loading model" + model_paths[-1])
    runner = get_mappo_runner(env, cfg_train, logdir, env.device, cfg['sim']['numAgents'])
    
    policy_infos = runner.load_multi_path(model_paths)
    policy = runner.get_inference_policy(device=env.device)
    attention_head = runner.alg.actor_critic.ac1.actor._encoder._network
    num_ego_obs = runner.alg.actor_critic.ac1.actor._encoder.num_ego_obs
    num_ado_obs = runner.alg.actor_critic.ac1.actor._encoder.num_ado_obs
    num_agents = runner.alg.actor_critic.ac1.actor._encoder.num_agents
    attention_tensor = torch.zeros((env.num_envs, env.num_agents-1))

    def compute_linear_attention(obs, attention_tensor):
        obs_ego = obs[..., :num_ego_obs]
        obs_ado = obs[..., num_ego_obs:num_ego_obs+(num_agents-1)*num_ado_obs]

        for ado_id in range(num_agents-1):
            ado_ag_obs = obs_ado[..., ado_id::(num_agents-1)]
            attention_tensor[:, ado_id]= attention_head(torch.cat((obs_ego, ado_ag_obs), dim=-1).detach())
        return attention_tensor

    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']
    
    ag = 0

    skill_ag = [[p_infos['trueskill']['mu'], p_infos['trueskill']['sigma']] for p_infos in policy_infos]
    mu = [skill[0] for skill in skill_ag]
    sigma = [skill[1] for skill in skill_ag]
    for idx in range(env.num_agents):
        win_prob = compute_winprob_i(mu, sigma, idx)
        print("win probability agent ",str(idx)," : ", win_prob)
    idx = 0 
    rank_old = env.ranks[env.viewer.env_idx_render, 0].item()
    playedlasttime = False
    lastvel = 0
    mus = []
    fig, ax = plt.subplots(nrows=1, ncols=1)

    #while True:
    for idx in range(30000):
        
        t1 = time.time()
        obs = obs
        actions = policy(obs)
        obs, _, rew, dones, info = env.step(actions)
        attention_tensor = compute_linear_attention(obs[:,ag,:], attention_tensor)
        if 'ranking' in info:
            ranks = info['ranking'][0].cpu().numpy()#torch.mean(1.0*env.ranks, dim= 0).cpu().numpy()
            update_ratio = info['ranking'][1]
            #for rkidx in range(ranks.shape[0]):
            #rank = ranks[rkidx, :]
            new_ratings = trueskill.rate(ratings, ranks)
            for old, new, it in zip(ratings, new_ratings, range(len(ratings))):
                mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
                sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
                ratings[it] = (trueskill.Rating(mu, sigma),)
            #ratings = []
            #ratings = new_ratings
            mus.append([r[0].mu for r in ratings])
            sigs = [r[0].sigma for r in ratings]
            print('rating mus', mus[-1])
            print('rating sigma', sigs)
            ax.clear()
            ax.plot(np.array(mus), label = ['0', '1','2','3'])
            ax.legend()
            plt.draw()
            #plt.show(block = False)

        obsnp = obs[:,ag,:].cpu().numpy()
        act = actions[:,ag,:].cpu().detach().numpy()
        states = env.states.cpu().numpy()
        om_mean = np.mean(states[env.viewer.env_idx_render,ag, env.vn['S_W0']:env.vn['S_W3'] +1 ])
        rank = env.ranks[env.viewer.env_idx_render, ag].item() +1
        ranks = env.ranks[env.viewer.env_idx_render, :] +1
        vel = env.vels_body[env.viewer.env_idx_render,ag, 0].item() 
        acc = vel - lastvel
        #lastvel = vel
        
        
        #env.viewer.add_lines()
        ado_pattern = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
        #ado_obs_ag = obsnp[env.viewer.env_idx_render, 35:52]
        ado_msg_attention = [(f"""{'att ' + str(ado_idx):>{10}}{' '}{attention_tensor[env.viewer.env_idx_render, idx].item():.2f}""") if env.active_agents[env.viewer.env_idx_render, ado_idx] else (f"""{'att ' + str(ado_idx):>{10}}{' '}{'deactivated'}""") for idx, ado_idx in enumerate(ado_pattern[ag])]
        #ado_msg_cont = [(f"""{'ado cont ag' + str(ado_idx):>{10}}{' '}{ado_obs_ag[3+idx]:.2f}""") for idx, ado_idx in enumerate(ado_pattern[ag])]



        viewermsg = [#(f"""{'relative proggress a1-a0:':>{10}}{' '}{relprogress.item():.2f}"""),
                     #(f"""{'relative contouringerr a1-a0:':>{10}}{' '}{relcontouringerr.item():.2f}"""),
                     #(f"""{'p0 '+str(modelnrs[0])}{' ts: '}{policy_infos[0]['trueskill']['mu']:.1f}"""), 
                     #(f"""{'p1 '+str(modelnrs[1])}{' ts: '}{policy_infos[1]['trueskill']['mu']:.1f}"""),
                     #(f"""{'p2 '+str(modelnrs[2])}{' ts: '}{policy_infos[2]['trueskill']['mu']:.1f}"""),
                     #(f"""{'p3 '+str(modelnrs[3])}{' ts: '}{policy_infos[3]['trueskill']['mu']:.1f}"""),
                     (f"""{'att 0 '+str(modelnrs[3])}{' ts: '}{policy_infos[3]['trueskill']['mu']:.1f}"""),
                     (f"""{'att 1 '+str(modelnrs[3])}{' ts: '}{policy_infos[3]['trueskill']['mu']:.1f}"""),
                     (f"""{'att 2 '+str(modelnrs[3])}{' ts: '}{policy_infos[3]['trueskill']['mu']:.1f}"""),
                     
                     (f"""{'rank :':>{10}}{' '}{ranks[ag]:.2f}""")
                     ]

        viewermsg = viewermsg + ado_msg_attention
        if SOUND:
            diff =  rank-rank_old
            col = torch.any(env.is_collision[env.viewer.env_idx_render])
            if col and idx%47 ==0:
                threading.Thread(target=playsound.playsound, args=('dmaracing/utils/audio/collisions.mp3',), daemon=True).start()
            if col and idx%45 == 0:
                threading.Thread(target=playsound.playsound, args=('dmaracing/utils/audio/oof.mp3',), daemon=True).start()  
            if diff>0 and not playedlasttime:
                threading.Thread(target=playsound.playsound, args=('dmaracing/utils/audio/overtaking.mp3',), daemon=True).start()
                playedlasttime = True
                
            elif playedlasttime:
                playedlasttime = False
            #idx += 1
        
        
        env.viewer.x_offset = int(-env.viewer.width/env.viewer.scale_x*env.states[env.viewer.env_idx_render, ag, 0])
        env.viewer.y_offset = int(env.viewer.height/env.viewer.scale_y*env.states[env.viewer.env_idx_render, ag, 1])
        env.viewer.draw_track()
        env.viewer.clear_string()
        env.viewer.clear_markers()
        env.viewer.clear_lines()
        
        #self_centerline_pos = env.interpolated_centers[env.viewer.env_idx_render, ag, 0, :].cpu().numpy()
        #env.viewer.add_point(self_centerline_pos.reshape(1,2), 2,(222,10,0), 2)
        #endpoints = np.array([env.states[env.viewer].reshape(1,2), 
        #                      env.states[env.viewer.env_idx_render, ag, 0:2].cpu().numpy().reshape(1,2)])
        #env.viewer.add_lines(endpoints=endpoints.squeeze(), color = (0,0,255), thickness=2)
   
        for ado_idx, ado_ag in enumerate(ado_pattern[ag]):
            if ado_ag != ag:
                endpoints = np.array([states[env.viewer.env_idx_render,ag,0:2],
                                      states[env.viewer.env_idx_render,ado_ag,0:2]])
                if attention_tensor[env.viewer.env_idx_render, ado_idx].item() >= 0.01 and env.active_agents[env.viewer.env_idx_render, ado_ag]:
                    env.viewer.add_lines(endpoints=endpoints.squeeze(), color = (255,0,0), thickness= int(8*attention_tensor[env.viewer.env_idx_render, ado_idx].item()))
            
        for msg in viewermsg:
            env.viewer.add_string(msg)

        evt = env.viewer_events

        if evt == 105:
            ag = (ag + 1) % env.num_agents
        if evt == 117:
            ag = (ag - 1) % env.num_agents
            
        if evt == 121:
            print("env ", env.viewer.env_idx_render, " reset")
            env.episode_length_buf[env.viewer.env_idx_render] = 1e9
        if evt == 112:
            plt.show(block = False)
            print('paused')
        #if idx%100 ==0:
        #    env.episode_length_buf[:] = 1e9
    plt.show()    

if __name__ == "__main__":
    args = CmdLineArguments()
    SOUND = False
    args.device = 'cuda:0'
    args.headless = False 
    #args.test = True
    path_cfg = os.getcwd() + '/cfg'

    cfg, cfg_train, logdir = getcfg(path_cfg)

    chkpts = [-1]*4
    runs = [-1]*4
    cfg['sim']['numEnv'] = 1 #500
    cfg['sim']['numAgents'] = 4
    cfg['sim']['collide'] = 1
    
    cfg['learn']['timeout'] = 100
    cfg['learn']['offtrack_reset'] = 5.0
    #cfg['learn']['reset_tile_rand'] = 20
    #cfg['sim']['test_mode'] = True
    cfg['learn']['resetgrid'] = True
    cfg['learn']['reset_tile_rand'] = 40
    cfg['viewer']['logEvery'] = -1
    cfg['track']['seed'] = 5
    cfg['track']['num_tracks'] = 30
    cfg['viewer']['multiagent'] = True
    cfg['learn']['agent_dropout_prob'] = 0.3

    set_dependent_cfg_entries(cfg)

    play()    
