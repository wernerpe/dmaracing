from gym import spaces
import numpy as np

class BaseEnv:
    def __init__(self, env) -> None:
        self.env = env
        self.obs_space = spaces.Box(np.ones(self.num_obs)*-np.Inf, np.ones(self.num_obs)*np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states)*-np.Inf, np.ones(self.num_states)*np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions)*-np.Inf, np.ones(self.num_actions)*np.Inf)

    @property
    def observation_space(self):
        return self.obs_space
    
    @property
    def action_space(self):
        return self.obs_space
    
    @property
    def observation_space(self):
        return self.obs_space
    