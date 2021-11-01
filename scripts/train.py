from rl_pytorch.ppo import PPO, ActorCritic
from dmaracing.env.dmar import DmarEnv
from dmaracing.utils.helpers import *
from datetime import date, datetime
import os

def train():
    env = DmarEnv(cfg, args)
    learn_cfg = cfg_train['learn']

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
              is_testing=False,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )
    print('ppo created')
    ppo.run(cfg_train['learn']['max_iterations'],cfg_train['learn']['save_interval'])


if __name__ == "__main__":
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train, logdir = getcfg(path_cfg)
    now = datetime.now()
    timestamp = now.strftime("%y_%m_%d_%H_%M_%S")
    logdir = logdir+'/'+timestamp
    train()