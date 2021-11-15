import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
import os

def play():
    cfg['sim']['numEnv'] = 4
    cfg['sim']['numAgents'] = 1
    cfg['viewer']['multiagent'] = True
    cfg['learn']['defaultactions'] = [0,0,0]
    cfg['learn']['actionscale'] = [1,1,1]
    env = DmarEnv(cfg, args)
    obs = env.obs_buf

    #actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numAgents'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    actions = torch.zeros((cfg['sim']['numEnv'], cfg['sim']['numActions']), device=args.device,  dtype= torch.float, requires_grad=False)
    vel_cmd = 0.0
    steer_cmd = 0.0
    brk_cmd = 0.0
      
    while True:
        actions[0 , 0] = steer_cmd 
        actions[0 , 1] = vel_cmd
        actions[0 , 2] = brk_cmd
        
        obs, _, rew, dones, info = env.step(actions)
        obsnp = obs.cpu().numpy()
        rewnp = rew.cpu().numpy()
        #print("-------------------------------")
        #print("rewards      :", rew[0])
        #print("velocity     :", obs[0, :2])
        #print("ang velocity :", obs[0, 2])
        #print("steer        :", obs[0, 3])
        #print("gas          :", obs[0, 4])
        
        viewermsg = [(f"""{'rewards:':>{10}}{' '}{rewnp[0]:.2f}"""   ),
                     (f"""{'velocity x:':>{10}}{' '}{obsnp[0, 0]:.2f}"""),
                     (f"""{'velocity y:':>{10}}{' '}{obsnp[0, 1]:.2f}"""),
                     (f"""{'ang vel:':>{10}}{' '}{obsnp[0, 2]:.2f}"""),
                     (f"""{'steer:':>{10}}{' '}{obsnp[0, 3]:.2f}"""),
                     (f"""{'gas:':>{10}}{' '}{obsnp[0, 4]:.2f}""")
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
            vel_cmd = 0.1
            brk_cmd = 0.9
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
