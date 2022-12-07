import torch

class PPController:
    def __init__(self,
                 env , 
                 lookahead_dist = 0.2, #[m]
                 maxvel = 1.0,
                 k_steer = 1.0,
                 k_gas = 1.0,
                 ):
        self.maxvel = maxvel
        self.env = env
        self.k_s = k_steer
        self.k_g = k_gas
        self.lookahead_dist = lookahead_dist
        self._actions = torch.zeros_like(env.actions)
        self.ori_vec = torch.zeros_like(env.actions)
        self._int_lh = int(self.lookahead_dist/self.env.track_poly_spacing)

    def step(self):
        lookahead_locs = self.get_lookahead()
        steer_error = self.get_steer_error(lookahead_locs)
        gas_error = self.get_vel_error(steer_error)
        self._actions[..., 0] = self.k_s * steer_error
        self._actions[..., 1] = self.k_g * gas_error    
        return self._actions

    def get_lookahead(self,):
        idx = (self.env.active_track_tile + self._int_lh).view(-1, self.env.num_agents, 1)
        idx = torch.remainder(idx, self.env.active_track_tile_counts.view(-1, 1, 1))
        centers = self.env.active_centerlines[:, idx, :]
        centers = centers[self.env.all_envs, self.env.all_envs, ...]
        angles_at_centers = self.env.active_alphas[:, idx].clone()
        angles_at_centers = angles_at_centers[self.env.all_envs, self.env.all_envs, ...]
        self.trackdir_lookahead = torch.stack((torch.cos(angles_at_centers), torch.sin(angles_at_centers)), dim=3)
        self.lookahead_points = centers + self.trackdir_lookahead * self.env.sub_tile_progress.view(
            self.env.num_envs, self.env.num_agents, 1, 1
        )
        return self.lookahead_points[:, :, 0, :]

    def get_steer_error(self, lookahead_loc):
        yaw = self.env.states[..., 2]
        loc = self.env.states[..., 0:2]
        self.ori_vec[..., 0] = torch.cos(yaw) 
        self.ori_vec[..., 1] = torch.sin(yaw) 
        pt = lookahead_loc - loc
        lookahead_dir = pt/(1e-6+torch.norm(pt , dim = -1).view(-1, self.env.num_agents, 1))
        ang = torch.arccos(0.995*torch.einsum('eac, eac -> ea', self.ori_vec, lookahead_dir))
        sign = torch.sign(self.ori_vec[...,0]*lookahead_dir[..., 1] - self.ori_vec[...,1]*lookahead_dir[..., 0]) 

        return ang*sign

    def get_vel_error(self, ang_error):
        vel = torch.norm(self.env.states[..., 3:5], dim =-1) 
        return torch.clip((1-0.4*torch.abs(ang_error)), max =1, min = 0.2)*self.env.dyn_model.dynamics_integrator.dyn_model.max_vel_vec.squeeze()-vel 
        #return torch.clip((1-0.4*torch.abs(ang_error)), max =1, min = 0.2)*self.maxvel-vel 
