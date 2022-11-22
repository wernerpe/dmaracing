from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.runners.multi_agent_on_policy_runner import MAOnPolicyRunner
from rsl_rl.runners.hierarchical_on_policy_runner import HierarchicalOnPolicyRunner

def get_ppo_runner(env, train_cfg, log_dir, device):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device = device)
    return runner

def get_mappo_runner(env, train_cfg, log_dir, device, num_agents = 2):
    runner = MAOnPolicyRunner(env, train_cfg, log_dir, device, num_agents)
    return runner

def get_hierarchical_ppo_runner(env, train_cfg, log_dir, device):
    runner = HierarchicalOnPolicyRunner(env, train_cfg, log_dir, device = device)
    return runner
