from rl_games.common import vecenv, env_configurations
from rl_games.common import algo_observer
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner
from rl_games import algos_torch
from dmaracing.env.racing_sim import DmarEnv
from dmaracing.utils.helpers import *
import os

def create_env(cfg, args):
    env = DmarEnv(cfg, args)
    return env

class RLGDmarEnv(vecenv.IVecEnv):
    def __init__(self, cfg, args):
        self.env = DmarEnv(cfg, args)
        self.info = {}

    def step(self, action):
        obs, reward, done, self.info = self.env.step(action)
        return obs, reward, done, self.info
        
    def reset(self):
        self.obs = self.env.reset()
        return self.obs

    def get_env_info(self):
        return self.info

class RLGDmarAlgoObserver(AlgoObserver):
    def __init__(self):
        super().__init__()


#deploy training here
if __name__ == "__main__":
    vecenv.register('Dmar', lambda cfg, num_actors, **kwargs : RLGDmarEnv())
    print('uauauuaua this interface is giving me a stroke')
    args = CmdLineArguments()
    args.device = 'cuda:0'
    args.headless = False 
    path_cfg = os.getcwd() + '/cfg'
    cfg, cfg_train = getcfg(path_cfg)
    algo_observer = DmarAlgoObserver()
    runner = Runner(algo_observer)
    runner.load(cfg_train)
    runner.run(vargs)