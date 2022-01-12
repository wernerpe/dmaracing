#from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.runners.multi_agent_on_policy_runner import MAOnPolicyRunner

def get_ppo_runner(env, train_cfg, log_dir, device):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device = device)
    return runner

def get_mappo_runner(env, train_cfg, log_dir, device):
    runner = MAOnPolicyRunner(env, train_cfg, log_dir, device = device)
    return runner
