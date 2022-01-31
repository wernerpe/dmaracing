import yaml
import os
import torch

class CmdLineArguments:
    pass


def set_dependent_cfg_entries(cfg):
    numObservations = cfg['sim']['numConstantObservations']
    numObservations += cfg['learn']['horizon']*2 #lookaheadhorizon
    numObservations += (2+1+1+3) * (cfg['sim']['numAgents']-1)
    cfg['sim']['numObservations'] = numObservations

def getcfg(path):
    pth_cfg = path+'/cfg.yml'
    pth_cfg_train = path+'/cfg_train.yml'
    
    with open(pth_cfg, 'r') as stream:
        cfg = yaml.safe_load(stream)
    with open(pth_cfg_train, 'r') as stream:
        cfg_train = yaml.safe_load(stream)
    logdir = 'logs/'+cfg_train['runner']['experiment_name']
    
    #update parameter dependent observation counts
    set_dependent_cfg_entries(cfg)
    return cfg, cfg_train, logdir


def  get_run(logdir, run, chkpt):
    
    if type(run) == str:
        model_dir = logdir + '/'+ run
    else:
        runs = os.listdir(logdir)
        runs.sort()
        runstr = runs[run]
        model_dir = logdir+'/'+ runstr

    if chkpt == -1:
        models = os.listdir(model_dir)
        models = [i for i in models if 'model' in i]
        nrs = [int(model[6:-3]) for model in models]
        nrs.sort()
        modelnr = int(nrs[-1])
    else:
        modelnr = chkpt

    return model_dir, modelnr

def get_ppo(args, env, cfg_train, logdir):
    learn_cfg = cfg_train['learn']
    chkpt = learn_cfg["resume"]
    
    ppo = PPO(vec_env=env,
              actor_critic_class=ActorCritic,
              num_transitions_per_env=learn_cfg["nsteps"],
              num_learning_epochs=learn_cfg["noptepochs"],
              num_mini_batches=learn_cfg["nminibatches"],
              clip_param=learn_cfg["cliprange"],
              gamma=learn_cfg["gamma"],
              lam=learn_cfg["lam"],
              init_noise_std=learn_cfg.get("init_noise_std", 0.3),
              value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
              entropy_coef=learn_cfg["ent_coef"],
              learning_rate=learn_cfg["optim_stepsize"],
              max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
              use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
              schedule=learn_cfg.get("schedule", "fixed"),
              desired_kl=learn_cfg.get("desired_kl", None),
              model_cfg=cfg_train["policy"],
              device=env.rl_device,
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=args.test,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    if args.test:
        print("Loading model from {}/model_{}.pt".format(logdir, chkpt))
        ppo.test()
    return ppo


def rand(min, max, shape, device):
    r = torch.rand(shape, device=device, dtype = torch.float, requires_grad=False)
    dist = max-min
    return dist*r + min