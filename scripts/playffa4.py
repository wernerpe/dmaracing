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
    
    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']
    
    ag = 0

    num_races = 0
    num_agent_0_wins = 0
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
    all_results = []
    all_race_times = []
    env_idx_ag0_win = []
    env_perm = np.random.permutation(4)    
    env_inv_perm = np.argsort(env_perm)
    print(env_inv_perm)
    mus = []
    fig, ax = plt.subplots(nrows=1, ncols=1)

    #while True:
    for idx in range(30000):
        #if idx%500 == 0:
        #    perm = np.random.permutation(env.num_agents)    
        #    inv_perm = np.argsort(perm)
        #    env_perm = torch.tensor(perm, dtype=torch.long, device=env.device)
        #    inv_env_perm = torch.tensor(inv_perm, dtype=torch.long, device=env.device)
    
        t1 = time.time()
        obs = obs#[:,env_perm,:]
        actions = policy(obs)#[:,env_inv_perm,:]
        #actions[:,0,1] *=0.0

        obs, _, rew, dones, info = env.step(actions)
        #if 'ranking' in info.keys():
            # avg = torch.mean(1.0*info['ranking'], dim = 0) 
            # all_results += [1.0*info['ranking']]#[:, env_inv_perm]]
            # all_race_times += [info['percentage_max_episode_length']]
            # #idx_ag0_win = torch.where((1.0*info['ranking'])[:, 0] == 0)
            # #env_idx_ag0_win += [torch.where(dones)[0][idx_ag0_win]]
            # ranks = (1.0*info['ranking']).cpu().numpy()#torch.mean(1.0*info['ranking'], dim = 0).cpu().numpy()
            # #agent_of_rank = np.argsort(ranks)
            # dones_idx = torch.unique(torch.where(dones)[0])
            ##for rnkidx in range(ranks):
            #ranks_final = 0*agent_of_rank
            #for it, envag in enumerate(agent_of_rank[1:]):
            #    #avg(env (rank i))- avg(env (rank i-1))>eps 
            #    if ranks[envag]- ranks[agent_of_rank[it]]  > 0.2:
            #        ranks_final[envag] = ranks_final[agent_of_rank[it]] + 1 
            #    else:
            #        ranks_final[envag] = ranks_final[agent_of_rank[it]]
            #ranks_final = (ranks_final).tolist()
            
            
            #update_ratio = 1.0*len(dones_idx)/len(dones)
            #new_ratings = trueskill.rate(ratings, ranks)
            #
            #for old, new, it in zip(ratings, new_ratings, range(len(ratings))):
            #    mu = (1-update_ratio)*old[0].mu + update_ratio*new[0].mu
            #    sigma = (1-update_ratio)*old[0].sigma + update_ratio*new[0].sigma
            #    ratings[it] = (trueskill.Rating(mu, sigma),)

        dones_idx = torch.unique(torch.where(dones)[0])
        #if len(dones_idx):
        #    num_races += len(dones_idx)
        #    num_agent_0_wins +=len(torch.where(info['ranking'][:,0] == 0))

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
            #if len(all_results):
                #allidx_ag0win = np.concatenate(tuple([r.cpu().numpy().reshape(-1,) for r in env_idx_ag0_win]), axis = 0) 
                #print('ag0 win idx: ', np.mean(allidx_ag0win))
                #res = np.concatenate(tuple([(r*perc.view(-1,1)).cpu().numpy().reshape(-1,4) for perc, r in zip(all_race_times, all_results)]), axis = 0)
                #tot_time_percentage = 
                #print(len(res), np.mean(res, axis = 0))
                #print('overall', np.mean(res))
            #mus.append([r[0].mu for r in ratings])
            print('rating mus', mus[-1])
            print('rating sigma', sigs)
            ax.clear()
            ax.plot(np.array(mus), label = ['0', '1','2','3'])
            ax.legend()
            plt.draw()
            #plt.show(block = False)

        #if idx %300 ==0:
        #    print("wins_0 / races: ", num_agent_0_wins, '/', num_races, '=', num_agent_0_wins*1.0/(num_races+0.001))
        obsnp = obs[:,ag,:].cpu().numpy()
        #rewnp = rew[:,0].cpu().numpy()
        #cont = env.contouring_err.cpu().numpy()
        act = actions[:,ag,:].cpu().detach().numpy()
        states = env.states.cpu().numpy()
        #relprogress = env.progress_other[env.viewer.env_idx_render, ag, 0]
        #relcontouringerr = env.contouring_err_other[env.viewer.env_idx_render, ag, 0]
        om_mean = np.mean(states[env.viewer.env_idx_render,ag, env.vn['S_W0']:env.vn['S_W3'] +1 ])
        #step = env.episode_length_buf[env.viewer.env_idx_render].item()
        #time_sim = cfg['sim']['dt']*cfg['sim']['decimation']*step
        rank = env.ranks[env.viewer.env_idx_render, ag].item() +1
        ranks = env.ranks[env.viewer.env_idx_render, :] +1
        vel = env.vels_body[env.viewer.env_idx_render,ag, 0].item() 
        acc = vel - lastvel
        #lastvel = vel
        #print(states[env.viewer.env_idx_render,ag, env.vn['S_W0']:env.vn['S_W3'] +1 ])

        ##check ado observations
        
        ado_prog = []
        ado_cont = []
        #env.viewer.add_lines()
        ado_pattern = [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
        ado_obs_ag = obsnp[env.viewer.env_idx_render, 35:52]
        ado_msg_prog = [(f"""{'ado prog ag' + str(ado_idx):>{10}}{' '}{ado_obs_ag[idx]:.2f}""") for idx, ado_idx in enumerate(ado_pattern[ag])]
        ado_msg_cont = [(f"""{'ado cont ag' + str(ado_idx):>{10}}{' '}{ado_obs_ag[3+idx]:.2f}""") for idx, ado_idx in enumerate(ado_pattern[ag])]



        viewermsg = [#(f"""{'relative proggress a1-a0:':>{10}}{' '}{relprogress.item():.2f}"""),
                     #(f"""{'relative contouringerr a1-a0:':>{10}}{' '}{relcontouringerr.item():.2f}"""),
                     (f"""{'p0 '+str(modelnrs[0])}{' ts: '}{policy_infos[0]['trueskill']['mu']:.1f}"""), 
                     (f"""{'p1 '+str(modelnrs[1])}{' ts: '}{policy_infos[1]['trueskill']['mu']:.1f}"""),
                     (f"""{'p2 '+str(modelnrs[2])}{' ts: '}{policy_infos[2]['trueskill']['mu']:.1f}"""),
                     (f"""{'p3 '+str(modelnrs[3])}{' ts: '}{policy_infos[3]['trueskill']['mu']:.1f}"""),
                     (f"""{'rank :':>{10}}{' '}{ranks[ag]:.2f}""")
                     ]

        viewermsg = viewermsg+ ado_msg_prog + ado_msg_cont
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
        #endpoints = np.array([self_centerline_pos.reshape(1,2), 
        #                      env.states[env.viewer.env_idx_render, ag, 0:2].cpu().numpy().reshape(1,2)])
        #env.viewer.add_lines(endpoints=endpoints.squeeze(), color = (0,0,255), thickness=2)
        
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

    chkpts = [-1, 20000, 10000, 1000]
    runs = [-1]*4#[-3, -3, -3, -3]
    cfg['sim']['numEnv'] = 1 #500
    cfg['sim']['numAgents'] = 4
    cfg['sim']['collide'] = 1
    
    #cfg['learn']['timeout'] = 10
    #cfg['learn']['offtrack_reset'] = 5.0
    #cfg['learn']['reset_tile_rand'] = 20
    #cfg['sim']['test_mode'] = True
    cfg['learn']['resetgrid'] = True
    cfg['learn']['reset_tile_rand'] = 40
    cfg['viewer']['logEvery'] = -1
    cfg['track']['seed'] = 5
    cfg['track']['num_tracks'] = 30
    cfg['viewer']['multiagent'] = True

    set_dependent_cfg_entries(cfg)

    play()    
