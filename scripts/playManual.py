import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
import os
import numpy as np

def play():
    cfg['sim']['numEnv'] = 4
    cfg['sim']['numAgents'] = 1
    cfg['track']['num_tracks'] = 3
    cfg['viewer']['multiagent'] = True
    cfg['learn']['defaultactions'] = [0,0,0]
    cfg['learn']['actionscale'] = [1,1,1]
    cfg['learn']['offtrack_reset'] = 100
    cfg['learn']['timeout'] = 100
    cfg['model']['OFFTRACK_FRICTION_SCALE'] = 1
    
    
    env = DmarEnv(cfg, args)
    obs = env.obs_buf

    #actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    vel_cmd = 0.0
    steer_cmd = 0.0
    brk_cmd = 0.0
      
    while True:
        actions[0 , 0, 0] = steer_cmd 
        actions[0 , 0, 1] = vel_cmd
        actions[0 , 0, 2] = brk_cmd
        
        obs, _, rew, dones, info = env.step(actions)
        obsnp = obs[:,:].cpu().numpy()
        rewnp = rew[:].cpu().numpy()
        cont = env.conturing_err.cpu().numpy()
        act = actions[:,0,:].cpu().detach().numpy()
        states = env.states.cpu().numpy()
        om_mean = np.mean(states[env.viewer.env_idx_render,0, env.vn['S_W0']:env.vn['S_W3'] +1 ])

        viewermsg = [
                     (f"""{'rewards:':>{10}}{' '}{100*rewnp[env.viewer.env_idx_render]:.2f}"""   ),
                     (f"""{'velocity x:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 0]:.2f}"""),
                     (f"""{'velocity y:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 1]:.2f}"""),
                     (f"""{'ang vel:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 2]:.2f}"""),
                     (f"""{'steer:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_STEER']]:.2f}"""),
                     (f"""{'gas:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_GAS']]:.2f}"""),
                     (f"""{'brake:':>{10}}{' '}{act[env.viewer.env_idx_render, env.vn['A_BRAKE']]:.2f}"""),
                     (f"""{'cont err:':>{10}}{' '}{cont[env.viewer.env_idx_render, 0]:.2f}"""),
                     (f"""{'omega mean:':>{10}}{' '}{om_mean:.2f}"""),
                     (f"""{'omega mean:':>{10}}{' '}{om_mean:.2f}"""),
                     (f"""{'lap:':>{10}}{' '}{env.lap_counter[0, 0]:.2f}"""),
                     ]
        
        env.viewer.clear_string()
        for msg in viewermsg:
            env.viewer.add_string(msg)

        evt = env.viewer_events
        if evt == 105:
            vel_cmd += 0.1
            brk_cmd = 0
            print('vel_cmd', vel_cmd, env.states[0,0,env.vn['S_DX']])
        elif evt == 107:
            vel_cmd = 0.0
            brk_cmd += 0.1
            print('vel_cmd', vel_cmd, env.states[0,0,env.vn['S_DX']])
        elif evt == 106:
            steer_cmd += 0.4 * (steer_cmd < 1)
        elif evt == 108:
            steer_cmd -= 0.4 * (steer_cmd> -1)
       
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    play()    
