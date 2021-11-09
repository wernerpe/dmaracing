import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
import os

def play():
    chkpt = 240
    cfg['sim']['numEnv'] = 30
    env = DmarEnv(cfg, args)
    obs = env.obs_buf[:,0,:]
    dir, model = get_run(logdir, run = -1, chkpt=chkpt)
    cfg_train['learn']['resume'] = model
    ppo = get_ppo(args, env, cfg_train, dir)
    policy = ppo.actor_critic.act_inference
   
    while True:
        actions = policy(obs)
        #print(actions[0,:])
        obs, rew, dones, info = env.step(actions)
        obsnp = obs.cpu().numpy()
        rewnp = rew.cpu().numpy()
        #print("-------------------------------")
        #print("rewards      :", rew[0])
        #print("velocity     :", obs[0, :2])
        #print("ang velocity :", obs[0, 2])
        #print("steer        :", obs[0, 3])
        #print("gas          :", obs[0, 4])
        
        viewermsg = [(f"""{'policy chckpt '+str(chkpt)+', env 0'}"""),
                     (f"""{'rewards:':>{10}}{' '}{rewnp[0]:.2f}"""   ),
                     (f"""{'velocity x:':>{10}}{' '}{obsnp[0, 0]:.2f}"""),
                     (f"""{'velocity y:':>{10}}{' '}{obsnp[0, 1]:.2f}"""),
                     (f"""{'ang vel:':>{10}}{' '}{obsnp[0, 2]:.2f}"""),
                     (f"""{'steer:':>{10}}{' '}{obsnp[0, 3]:.2f}"""),
                     (f"""{'gas:':>{10}}{' '}{obsnp[0, 4]:.2f}""")
                     ]
        
        env.viewer.clear_string()
        for msg in viewermsg:
            env.viewer.add_string(msg)
        #print(obs)
        
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    args.test = True
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    play()    
