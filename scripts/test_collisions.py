from turtle import pos
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.env.car_dynamics_utils import get_collision_pairs, get_collision_pairs2
from dmaracing.utils.helpers import *
import os
import numpy as np
import time
positions = np.array([
                       #[[0, 0, np.pi/3], [4, 0, 0], [4, 2.2, 0], [0, 2.2, 0]],
                       #[[0, 0, np.pi/3], [2.45, 0.2, -np.pi/6], [7, 2.2, 0], [0, 2.2, -np.pi/6]],
                       #[[0, 0, np.pi/2], [4, 0, 0], [4, 2.2, 0], [0, 2.2, -np.pi/6]],
                       [[0, 0, np.pi/3], [2.5, 0.8, 0], [7, 2.2, 0], [0, 2.2, -np.pi/6]],
                     ])


def set_config(env, positions, colpair = 0):
    if colpair == 0:
        env.collision_pairs = get_collision_pairs(env.num_agents)
        print('colp0')
    else:
        env.collision_pairs = get_collision_pairs2(env.num_agents)
        print('colp1')
    env.states[:] = 0.0
    env.contact_wrenches[:] = 0.0
    env.shove[:] = 0.0
    for ag in range(env.num_agents):
        env.states[0,ag, 0] = positions[ag,0] + 0.0*(np.random.rand()-0.5)
        env.states[0,ag, 1] = positions[ag,1] + 0.0*(np.random.rand()-0.5)
        env.states[0,ag, 2] = positions[ag,2] + 0.0*(np.random.rand()-0.5)

def run():
    cfg['sim']['numEnv'] = 1
    cfg['sim']['numAgents'] = 4
    cfg['sim']['decimation'] = 1
    
    cfg['track']['num_tracks'] = 2

    cfg['viewer']['multiagent'] = True
    cfg['learn']['defaultactions'] = [0,0,0]
    cfg['learn']['actionscale'] = [1,1,1]
    cfg['learn']['offtrack_reset'] = 100
    cfg['learn']['timeout'] = 100
    cfg['model']['OFFTRACK_FRICTION_SCALE'] = 1
    cfg['model']['drag_reduction'] = 1.0
    
    set_dependent_cfg_entries(cfg)
    
    env = DmarEnv(cfg, args)
    obs = env.obs_buf

    #actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    conf_idx = 0
    contactpairidx = 0
    while True:
        set_config(env, positions[conf_idx], contactpairidx % 2)

        obs, _, rew, dones, info = env.step(actions)
        #enter analysis loop
        while True:
            time.sleep(0.1)
            print(env.contact_wrenches)
            #print(env.shove)
            #print(torch.sum(env.shove[:,:,0:2]))
            #print(torch.sum(env.contact_wrenches[:,:,0:2]))
            tau_sum = torch.sum(env.contact_wrenches[:,:,2])
            ri_x = env.states[:,:,0]
            ri_y = env.states[:,:,1]
            fi_x = env.contact_wrenches[:,:,0]
            fi_y = env.contact_wrenches[:,:,1]
            ri_cross_fi_sum = torch.sum(ri_x*fi_y-ri_y*fi_x)
            #print('Change of Angular momentum')
            #print(ri_cross_fi_sum+tau_sum)
            env.render()
        
            evt = env.viewer_events
            if evt == 112:
                print('next_config')
                break
        conf_idx += 1
        contactpairidx +=1
        if conf_idx == positions.shape[0]:
            conf_idx = 0
        
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    cfg["viewer"]["logEvery"] = -1
    cfg["track"]['OFFTRACK_FRICTION_SCALE'] = 1.0
    run()    
