from math import log
import torch
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from dmaracing.utils.rl_helpers import get_ppo_runner
import os

def play():
    chkpt = -1
    cfg['sim']['numEnv'] = 100
    cfg['sim']['numAgents'] = 1
    cfg['learn']['timeout'] = 30
    cfg['learn']['offtrack_reset'] = 3.0
    cfg['track']['seed'] = 11
    cfg['track']['CHECKPOINTS'] = 10
    cfg['track']['TRACK_RAD'] = 700
    cfg['viewer']['multiagent'] = True

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
    
    while True:
        actions = policy(obs)
        obs,_, rew, dones, info = env.step(actions)
        obsnp = obs.cpu().numpy()
        rewnp = rew.cpu().numpy()
        cont = env.conturing_err.cpu().numpy()

        viewermsg = [(f"""{'policy chckpt '+str(chkpt)}"""),
                     (f"""{'rewards:':>{10}}{' '}{100*rewnp[env.viewer.env_idx_render]:.2f}"""   ),
                     (f"""{'velocity x:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 0]:.2f}"""),
                     (f"""{'velocity y:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 1]:.2f}"""),
                     (f"""{'ang vel:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 2]:.2f}"""),
                     (f"""{'steer:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 3]:.2f}"""),
                     (f"""{'gas:':>{10}}{' '}{obsnp[env.viewer.env_idx_render, 4]:.2f}"""),
                     (f"""{'cont err:':>{10}}{' '}{cont[env.viewer.env_idx_render, 0]:.2f}""")
                     ]
        
        env.viewer.clear_string()
        for msg in viewermsg:
            env.viewer.add_string(msg)
        
if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    args.test = True
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    play()    
