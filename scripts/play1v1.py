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
import matplotlib.pyplot as plt
from dmaracing.controllers.purepursuit import PPController

def play():
    env = DmarEnv(cfg, args)
    ppc = PPController(env,
                       lookahead_dist=1.5,
                       maxvel=2.5,
                       k_steer=1.0,
                       k_gas=2.0)

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
    
    if SAVE:
        policy_jit = torch.jit.script(runner.alg.actor_critic.ac1.actor.to('cpu'))
        policy_jit.save("logs/saved_models/" + dir[15:] + "_" +str(modelnr)+".pt")
        print("Done saving")
        exit()

    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']

    num_races = 0
    num_agent_0_wins = 0
    #skill_ag0 = [policy_infos[0]['trueskill']['mu'], policy_infos[0]['trueskill']['sigma']]
    #skill_ag1 = [policy_infos[1]['trueskill']['mu'], policy_infos[1]['trueskill']['sigma']]
    #predicted win percentage
    #ratings = [trueskill.Rating(mu=skill_ag0), trueskill.Rating(mu=skill_ag1)]
    #print("matchup trueskill: ag0: ", skill_ag0,', ag1: ', skill_ag0)
    # mu_match = skill_ag0[0] - skill_ag1[0]
    # var_match = skill_ag0[1]**2 + skill_ag1[1]**2
    # win_prob = 1 - norm.cdf((0-mu_match)/(np.sqrt(2*var_match)))
    # print("win probability agent 0: ", win_prob, "var match: ", var_match)
    idx = 0 
    obs_past = []
    act_past = []
    ag = 0
    #for idx in range(150):
    while True:
        t1 = time.time()
        actions = policy(obs)
        if USE_PPC:
            actions[:,1,:] = ppc.step()[:,1,:]
        # actions[0, :,:] = 0
        # actions[0, :,1] = -0.5
        #obs_past.append(obs[0,...].detach().cpu().numpy())
        #act_past.append(actions[0,...].detach().cpu().numpy())

        #actions[:,:,1] =0.5
        obs, _, rew, dones, info = env.step(actions)
        dones_idx = torch.unique(torch.where(dones)[0])
        if len(dones_idx):
            num_races += len(dones_idx)
            #num_agent_0_wins += torch.sum(info['ranking'][:,1], dim = 0).item()
        if idx %300 ==0:
            print("wins_0 / races: ", num_agent_0_wins, '/', num_races, '=', num_agent_0_wins*1.0/(num_races+0.001))
        obsnp = obs[:,ag,:].cpu().numpy()
        rewnp = rew[:,ag].cpu().numpy()
        cont = env.contouring_err.cpu().numpy()
        act = actions[:,ag,:].cpu().detach().numpy()
        states = env.states.cpu().numpy()
        #om_mean = np.mean(states[env.viewer.env_idx_render,0, env.vn['S_W0']:env.vn['S_W3'] +1 ])
        mv1 = env.dyn_model.dynamics_integrator.dyn_model.max_vel_vec[env.viewer.env_idx_render,1].item() if not USE_PPC else ppc.maxvel
        viewermsg = [(f"""{'ag: '+str(ag)}"""),
                     #(f"""{'p1 '+str(modelnrs[1])}{' ts: '}{policy_infos[1]['trueskill']['mu']:.1f}"""),
                     #(f"""{'Win prob p0 : ':>{10}}{win_prob:.3f}"""),
                     (f"""{'rewards:':>{10}}{' '}{100*rewnp[env.viewer.env_idx_render, 0]:.2f}"""   ),
                     (f"""{'velocity:':>{10}}{' '}{np.linalg.norm(obsnp[env.viewer.env_idx_render, 0:2]*10):.2f}"""),
                     (f"""{'maxvel 0:':>{10}}{' '}{env.dyn_model.dynamics_integrator.dyn_model.max_vel_vec[env.viewer.env_idx_render,0].item():.2f}"""),
                     (f"""{'maxvel 1:':>{10}}{' '}{mv1:.2f}"""),
                     #(f"""{'velocity y:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 1]:.2f}"""),
                     #(f"""{'ang vel:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 2]:.2f}"""),
                     (f"""{'steer car:':>{10}}{' '}{0.3*act[env.viewer.env_idx_render, 0]:.2f}"""),
                     #(f"""{'gas raw 0:':>{10}}{' '}{actions[env.viewer.env_idx_render, 0, env.vn['A_GAS']].item():.2f}"""),
                     (f"""{'gas inp 0:':>{10}}{' '}{env.action_scales[1] * actions[env.viewer.env_idx_render, ag, 1] + env.default_actions[1]:.2f}"""),
                     #(f"""{'gas 1:':>{10}}{' '}{actions[env.viewer.env_idx_render, 1, env.vn['A_GAS']].item():.2f}"""),
                     #(f"""{'brake:':>{10}}{' '}{act[env.viewer.env_idx_render, env.vn['A_BRAKE']]:.2f}"""),
                     #(f"""{'om_mean:':>{10}}{' '}{om_mean:.2f}"""),
                     (f"""{'collision:':>{10}}{' '}{env.is_collision[env.viewer.env_idx_render, ag].item():.2f}"""),
                     (f"""{'rank ag 0 :':>{10}}{' '}{env.ranks[env.viewer.env_idx_render, ag].item():.2f}"""),
                     (f"""{'laps ag 0 :':>{10}}{' '}{env.lap_counter[env.viewer.env_idx_render, ag].item():.2f}"""),
                     (f"""{'step :':>{10}}{' '}{env.episode_length_buf[env.viewer.env_idx_render].item():.2f}"""),
                     (f"""{'gasoffset:':>{10}}{' '}{env.dyn_model.integration_function.dyn_model.gas_noise[env.viewer.env_idx_render, ag].item():.2f}"""   )]

        env.viewer.x_offset = int(-env.viewer.width/env.viewer.scale_x*env.states[env.viewer.env_idx_render, ag, 0])
        env.viewer.y_offset = int(env.viewer.height/env.viewer.scale_y*env.states[env.viewer.env_idx_render, ag, 1])
        env.viewer.draw_track()
        if USE_PPC:
            env.viewer.clear_markers()
            env.viewer.add_point(ppc.lookahead_points[env.viewer.env_idx_render, 1, :,:].cpu().numpy(), 4,(0,0,255), 2)

        env.viewer.clear_string()
        for msg in viewermsg:
            env.viewer.add_string(msg)

        idx +=1
        evt = env.viewer_events

        if evt == 121:
            print("env ", env.viewer.env_idx_render, " reset")
            env.episode_length_buf[env.viewer.env_idx_render] = 1e9
        
        if evt == 105:
            ag = 1
        if evt == 111:
            ag = 0
        
        t2 = time.time()
        realtime = t2-t1-time_per_step
        if realtime < 0:
            time.sleep(-realtime)
    
    fig = plt.figure()
    plt.plot(np.array(obs_past)[:,0,:])
    plt.show()


if __name__ == "__main__":
    SAVE = False
    USE_PPC = False
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    args.test = True
    path_cfg = os.getcwd() + '/cfg'

    cfg, cfg_train, logdir = getcfg(path_cfg, postfix='_1v1')

    chkpts = [-1, -1]
    runs = [-1, -1] #['22_09_02_10_40_34_col_0_ar_0.4_rr_0.0', '22_09_02_10_40_34_col_0_ar_0.4_rr_0.0']
    cfg['sim']['numEnv'] = 4
    cfg['sim']['numAgents'] = 2
    cfg['learn']['timeout'] = 300
    #cfg['learn']['offtrack_reset'] = 4.0
    cfg['learn']['resetgrid'] = False
    cfg['learn']['resetrand'] = [0.0, 0.0, 0., 0.0, 0.0,  0., 0.0]
    cfg['sim']['collide'] = 1
    
    cfg['test'] = args.test
    cfg['learn']['obs_noise_lvl'] = 0.0 if args.test else cfg['learn']['obs_noise_lvl']

    if not args.headless:
        cfg['viewer']['logEvery'] = -1
    cfg["logdir"] = logdir
    set_dependent_cfg_entries(cfg, cfg_train)

    play()    
