import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
import os
import numpy as np

def play():
    cfg['sim']['numEnv'] = 32
    #cfg['sim']['numAgents'] = 4
    #cfg['sim']['decimation'] = 4
    
    #cfg['track']['num_tracks'] = 3
    #cfg['track']['num_tracks'] = 3
    #cfg['viewer']['multiagent'] = True
    cfg['learn']['defaultactions'] = [0,0,0]
    cfg['learn']['actionscale'] = [1,1,1]
    #cfg['learn']['resetrand'] = [0.0]*7
    cfg['learn']['reset_tile_rand'] = 5
    #cfg['learn']['resetgrid'] = True
    
    #cfg['learn']['offtrack_reset'] = 10
    #cfg['learn']['timeout'] = 100
    #cfg['model']['OFFTRACK_FRICTION_SCALE'] = 1
    #cfg['model']['drag_reduction'] = 1.0
    
    set_dependent_cfg_entries(cfg, cfg_train)
    
    env = DmarEnv(cfg, args)
    obs = env.obs_buf

    #actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    vel_cmd = 0.0
    steer_cmd = 0.0
    brk_cmd = 0.0
    lastvel = 0
    ag = 0
    #env.states[:] = 0.0
    #for idx in range(env.num_agents):
    #    env.states[0, idx, 0] = idx*10.0
    #    env.states[0, idx, 5] = 10.0
        
    print('###########################')
    while True:
        
        actions[0 , ag, 0] = steer_cmd 
        actions[0 , ag, 1] = vel_cmd
        actions[0 , ag, 2] = brk_cmd
        
        obs, _, rew, dones, info = env.step(actions)
        #if 'ranking' in info:
        #    print('info', info['ranking'])
        obsnp = obs[:,0,:].cpu().numpy()
        rewnp = rew[:, 0,0].cpu().numpy()
        cont = env.contouring_err.cpu().numpy()
        act = actions[:,ag,:].cpu().detach().numpy()
        states = env.states.cpu().numpy()
        om_mean = np.mean(states[env.viewer.env_idx_render,ag, env.vn['S_W0']:env.vn['S_W3'] +1 ])

        idx_veloth = 39
        vel_other = obsnp[env.viewer.env_idx_render, idx_veloth:idx_veloth+2]
        #print(states[env.viewer.env_idx_render,0, env.vn['S_W0']:env.vn['S_W3'] +1 ])
       
        #print(env.active_agents[env.viewer.env_idx_render])
        viewermsg = [
                     (f"""{'rewards:':>{10}}{' '}{100*rewnp[env.viewer.env_idx_render]:.2f}"""   ),
                     #(f"""{'velocity x:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 0]:.2f}"""),
                     #(f"""{'velocity y:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 1]:.2f}"""),
                     #(f"""{'ang vel:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 2]:.2f}"""),
                     #(f"""{'steer:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_STEER']]:.2f}"""),
                     #(f"""{'gas:':>{10}}{' '}{states[env.viewer.env_idx_render, 0, env.vn['S_GAS']]:.2f}"""),
                     (f"""{'gas state:':>{10}}{' '}{states[env.viewer.env_idx_render, ag, env.vn['S_GAS']]:.2f}"""),
                     (f"""{'gas act:':>{10}}{' '}{act[env.viewer.env_idx_render, env.vn['A_GAS']]:.2f}"""),
                     #(f"""{'brake:':>{10}}{' '}{act[env.viewer.env_idx_render, env.vn['A_BRAKE']]:.2f}"""),
                     #(f"""{'cont err:':>{10}}{' '}{cont[env.viewer.env_idx_render, 0]:.2f}"""),
                     (f"""{'omega mean:':>{10}}{' '}{om_mean:.2f}"""),
                     #(f"""{'omega mean:':>{10}}{' '}{om_mean:.2f}"""),
                     #(f"""{'velother x:':>{10}}{' '}{vel_other[0]:.2f}"""),
                     #(f"""{'velother y:':>{10}}{' '}{vel_other[1]:.2f}"""),                     
                     (f"""{'lap:':>{10}}{' '}{env.lap_counter[0, ag]:.2f}"""),
                     (f"""{'rank ag 0 :':>{10}}{' '}{1+env.ranks[env.viewer.env_idx_render, ag].item():.2f}"""),
                     #(f"""{'trank ag 0 :':>{10}}{' '}{1+env.teamranks[env.viewer.env_idx_render, ag].item():.2f}"""),
                     ]
        #print(env.progress_other[0,0,:])
        env.viewer.x_offset = int(-env.viewer.width/env.viewer.scale_x*env.states[env.viewer.env_idx_render, ag, 0])
        env.viewer.y_offset = int(env.viewer.height/env.viewer.scale_y*env.states[env.viewer.env_idx_render, ag, 1])
        env.viewer.draw_track()
       
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
            #brk_cmd += 0.1
            print('vel_cmd', vel_cmd, env.states[0,0,env.vn['S_DX']])
        elif evt == 106:
            steer_cmd += 0.4 * (steer_cmd < 1)
        elif evt == 108:
            steer_cmd -= 0.4 * (steer_cmd> -1)
        elif evt == 121:
            print("env ", env.viewer.env_idx_render, " reset")
            env.episode_length_buf[env.viewer.env_idx_render] = 1e9
        if evt == 110:
            ag = (ag + 1) % env.num_agents
        if evt == 109:
            ag = (ag - 1) % env.num_agents
        
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg, postfix='_straight_line')
    cfg["viewer"]["logEvery"] = -1
    cfg["track"]['OFFTRACK_FRICTION_SCALE'] = 1.0
    cfg["learn"]['defaultactions'] = [0.,0.,0.0]
    cfg["learn"]['offtrack_reset'] = 5
    cfg["learn"]['timeout'] = 20.0
    
    play()    
