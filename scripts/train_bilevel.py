from dmaracing.utils.rl_helpers import get_ppo_runner, get_hierarchical_ppo_runner, get_bilevel_ppo_runner
from dmaracing.env.dmar import DmarEnv
from dmaracing.env.dmar_bilevel import DmarEnvBilevel
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os

def train():
    # env = DmarEnv(cfg, args)
    env = DmarEnvBilevel(cfg, args)
    # runner = get_ppo_runner(env, cfg_train, logdir, args.device)
    # runner = get_hierarchical_ppo_runner(env, cfg_train, logdir, args.device)
    
    runner = get_bilevel_ppo_runner(env, cfg_train, logdir, args.device)

    # runner.learn(cfg_train['runner']['max_iterations'], init_at_random_ep_len=True)
    runner.learn(cfg_train['runner']['max_iterations'], init_at_random_ep_len=False)

if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = True  # False 
    path_cfg = os.getcwd() + '/cfg'
    # cfg, cfg_train, logdir = getcfg(path_cfg)
    cfg, cfg_train, logdir = getcfg(path_cfg, postfix='_bilevel', postfix_train='_bilevel')
    cfg['sim']['numAgents'] = 1
    cfg['sim']['collide'] = 0
    #cfg['track']['num_tracks'] = 2
    # cfg_train['runner']['policy_class_name'] = 'ActorCritic'
    # cfg_train['runner']['algorithm_class_name'] = 'PPO'
    # cfg_train['runner']['policy_class_name'] = 'HierarchicalActorCritic'
    # cfg_train['runner']['algorithm_class_name'] = 'HierarchicalPPO'
    cfg_train['runner']['policy_class_hl_name'] = 'BilevelActorCritic'
    cfg_train['runner']['algorithm_class_hl_name'] = 'BilevelPPO'
    cfg_train['runner']['policy_class_ll_name'] = 'ActorCritic'
    cfg_train['runner']['algorithm_class_ll_name'] = 'PPO'
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logdir = logdir+'/'+timestamp
    cfg["logdir"] = logdir
    cfg["viewer"]["logEvery"] = 5  #-1
    print("logevery")
    print(cfg["viewer"]["logEvery"])
    #cfg['learn']['offtrack_reset'] = 0.3
    set_dependent_cfg_entries(cfg, cfg_train)
    train()