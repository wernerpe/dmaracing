#from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

def get_ppo_runner(env, train_cfg, log_dir, device):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device = device)
    return runner
