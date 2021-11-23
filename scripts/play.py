from math import log
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rl_helpers import get_ppo_runner
import numpy as np
import os
import time

def play():
    chkpt = -1
    cfg['sim']['numEnv'] = 10
    cfg['sim']['numAgents'] = 1
    cfg['learn']['timeout'] = 300
    cfg['learn']['offtrack_reset'] = 3.0
    cfg['track']['seed'] = 31
    cfg['track']['num_tracks'] = 1

    cfg['track']['CHECKPOINTS'] = 12
    cfg['track']['TRACK_RAD'] = 700
    cfg['viewer']['multiagent'] = False

    env = DmarEnv(cfg, args)
    #env.viewer.mark_env(0)
    obs = env.obs_buf[:,0,:]

    dir, model = get_run(logdir, run = -1, chkpt=chkpt)
    chkpt = model
    runner = get_ppo_runner(env, cfg_train, logdir, device = env.device)
    model_path = "{}/model_{}.pt".format(dir, model)
    print("Loading model" + model_path)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=env.device)
    
    time_per_step = cfg['sim']['dt']*cfg['sim']['decimation']

    while True:
        t1 = time.time()
        actions = policy(obs)
        obs,_, rew, dones, info = env.step(actions)
        obsnp = obs.cpu().numpy()
        rewnp = rew.cpu().numpy()
        cont = env.conturing_err.cpu().numpy()
        act = actions.cpu().detach().numpy()
        states = env.states.cpu().numpy()
        om_mean = np.mean(states[env.viewer.env_idx_render,0, env.vn['S_W0']:env.vn['S_W3'] +1 ])

        viewermsg = [(f"""{'policy chckpt '+str(chkpt)}"""),
                     (f"""{'rewards:':>{10}}{' '}{100*rewnp[env.viewer.env_idx_render]:.2f}"""   ),
                     (f"""{'velocity x:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 0]:.2f}"""),
                     (f"""{'velocity y:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 1]:.2f}"""),
                     (f"""{'ang vel:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 2]:.2f}"""),
                     (f"""{'steer:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_STEER']]:.2f}"""),
                     (f"""{'gas:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_GAS']]:.2f}"""),
                     (f"""{'brake:':>{10}}{' '}{act[env.viewer.env_idx_render, env.vn['A_BRAKE']]:.2f}"""),
                     (f"""{'cont err:':>{10}}{' '}{cont[env.viewer.env_idx_render, 0]:.2f}"""),
                     (f"""{'omega mean:':>{10}}{' '}{om_mean:.2f}"""),
                     ]
        
        env.viewer.clear_string()
        for msg in viewermsg:
            env.viewer.add_string(msg)
        t2 = time.time()
        realtime = t2-t1-time_per_step
        #if realtime < 0:
        #     time.sleep(-realtime)

if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    args.test = True
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    play()    
