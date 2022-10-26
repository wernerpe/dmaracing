from math import log
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rl_helpers import get_ppo_runner
import numpy as np
import os
import time
from datetime import date, datetime

def play():
    chkpt = -1
    
    env = DmarEnv(cfg, args)
    names = print(env.track_names)
    #env.viewer.mark_env(0)
    obs = env.obs_buf

    dir, model = get_run(logdir, run = -1, chkpt=chkpt)
    chkpt = model
    runner = get_ppo_runner(env, cfg_train, logdir, device = env.device)
    model_path = "{}/model_{}.pt".format(dir, model)
    print("Loading model" + model_path)
    runner.load(model_path)

    policy = runner.get_inference_policy(device=env.device)
    
    lap_times = {}
    for idx, n in enumerate(env.track_names):
        lap_times[idx] = []

    obs = env.obs_buf.clone().view(-1, env.num_obs)
    steps = 0
    while True:
        #t1 = time.time()
        actions = policy(obs)
        #actions[:, 1] = 2.0
        obs,_, rew, dones, info = env.step(actions)
        steps +=1
        if len(info):
            track_ids =info['proginfo'][0].cpu().numpy().reshape(-1)
            time_raced = (info['proginfo'][1].cpu().numpy()*env.dt*env.decimation).reshape(-1)
            progress = info['proginfo'][3].cpu().numpy().reshape(-1)
            lap_lengths = info['proginfo'][5].cpu().numpy().reshape(-1)
            maxvels  = info['proginfo'][6].cpu().numpy().reshape(-1)
            is_max_ep_time = (np.abs(info['proginfo'][1].cpu().numpy() - env.max_episode_length) < 2).reshape(-1)
            num_laps_float = progress/lap_lengths
            avg_laptime_est = time_raced/num_laps_float 
            #env_ids = info['proginfo'][6]
            print('writing results to csv')
            for idt, mv, avg_laptime_est, max_ep_time in zip(track_ids, maxvels, avg_laptime_est, is_max_ep_time):
                lap_times[idt].append([mv, avg_laptime_est]) 
                if max_ep_time:
                    with open(logfile,'a') as fd:
                        line = str(idt)+', '+ str(mv) + ', ' + str(avg_laptime_est) +'\n'
                        #print(env.track_names[idt],  ', '+ str(mv),  ', '+str(avg_laptime_est))
                        fd.write(line)

        obsnp = obs.cpu().numpy()
        rewnp = rew.cpu().numpy()
        #cont = env.contuoring_err.cpu().numpy()
        act = env.actions.cpu().detach().numpy()
        #steer_commands.append(act[0,0,0])
        states = env.states.cpu().numpy()
        #om_mean = np.mean(states[env.viewer.env_idx_render,0, env.vn['S_W0']:env.vn['S_W3'] +1 ])

        viewermsg = [(f"""{'policy chckpt '+str(chkpt)}"""),
                     #(f"""{'rewards:':>{10}}{' '}{100*rewnp[env.viewer.env_idx_render]:.2f}"""   ),
                     #(f"""{'velocity x:':>{10}}{' '}{10*obsnp[env.viewer.env_idx_render, 0]:.2f}"""),
                     #(f"""{'velocity y:':>{10}}{' '}{10*obsnp[env.viewer.env_idx_render, 1]:.2f}"""),
                     (f"""{'velocity:':>{10}}{' '}{np.linalg.norm(obsnp[env.viewer.env_idx_render, 0:2]*10):.2f}"""),
                     #(f"""{'ang vel:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 2]:.2f}"""),
                     #(f"""{'trackprogress:':>{10}}{' '}{env.track_progress[env.viewer.env_idx_render,0].item():.2f}"""),
                     (f"""{'maxvel:':>{10}}{' '}{env.dyn_model.dynamics_integrator.dyn_model.max_vel_vec[env.viewer.env_idx_render,0].item():.2f}"""),                     
                     (f"""{'progress:':>{10}}{' '}{env.track_progress[env.viewer.env_idx_render,0].item():.2f}"""),                     
                     
                     #(f"""{'steer:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_STEER']]:.2f}"""),
                     #(f"""{'gas:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_GAS']]:.2f}"""),
                    # (f"""{'brake:':>{10}}{' '}{act[env.viewer.env_idx_render, 0, env.vn['A_BRAKE']]:.2f}"""),
                     #(f"""{'cont err:':>{10}}{' '}{cont[env.viewer.env_idx_render, 0]:.2f}"""),
                     #(f"""{'omega mean:':>{10}}{' '}{om_mean:.2f}"""),
                     ]
        env.viewer.x_offset = int(-env.viewer.width/env.viewer.scale_x*env.states[env.viewer.env_idx_render, 0, 0])
        env.viewer.y_offset = int(env.viewer.height/env.viewer.scale_y*env.states[env.viewer.env_idx_render, 0, 1])
        env.viewer.draw_track()
        #env.viewer.clear_markers()
        #closest_point_marker = env.interpolated_centers[env.viewer.env_idx_render, 0, :, :].cpu().numpy()
        #env.viewer.add_point(closest_point_marker, 2,(222,10,0), 2)
        #print(env.is_on_track[env.viewer.env_idx_render])
        env.viewer.clear_string()
        for msg in viewermsg:
            env.viewer.add_string(msg)
        
        evt = env.viewer_events
        if evt == 121:
            print("env ", env.viewer.env_idx_render, " reset")
            env.episode_length_buf[env.viewer.env_idx_render] = 1e9
        
        #t2 = time.time()
        #realtime = t2-t1-time_per_step
        
        # if realtime < 0:
        #      time.sleep(-realtime)

    # steer = np.array(steer_commands)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(steer)

if __name__ == "__main__":
    SAVE = False
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False
    args.test = True
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    cfg["logdir"] = logdir
    cfg_train['runner']['policy_class_name'] = 'ActorCritic'
    cfg_train['runner']['algorithm_class_name'] = 'PPO'
    cfg['learn']['resetrand'] = [0.0, 0.0, 0., 0.0, 0.0,  0., 0.0]
    #cfg['sim']['collide'] = 0
    cfg['sim']['numEnv'] = 2048
    cfg['sim']['numAgents'] = 1
    cfg['learn']['timeout'] = 20
    cfg["viewer"]["logEvery"] = -1
    cfg['trackRaceStatistics'] = True

    cfg['test'] = args.test
    set_dependent_cfg_entries(cfg, cfg_train)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logfile = logdir + '/laptimes_'+timestamp+'.csv'
    play()    
