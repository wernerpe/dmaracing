import torch
from torch._C import dtype
from gym import spaces
from dmaracing.env.car_dynamics import step_cars
from dmaracing.env.car_dynamics_utils import get_varnames, set_dependent_params, allocate_car_dynamics_tensors 
from dmaracing.env.viewer import Viewer
from typing import Tuple, Dict
import numpy as np
from dmaracing.utils.trackgen import get_track

class DmarEnv():
    def __init__(self, cfg, args) -> None:
        self.device = args.device
        self.rl_device = args.device
        self.headless = args.headless



        #variable names and indices
        self.vn = get_varnames() 
        self.modelParameters = cfg['model']
        set_dependent_params(self.modelParameters)
        self.simParameters = cfg['sim']
        self.num_states = 0
        self.num_internal_states = self.simParameters['numStates']
        self.num_actions = self.simParameters['numActions']
        self.num_obs = self.simParameters['numObservations']       
        self.num_agents = self.simParameters['numAgents']
        self.num_envs = self.simParameters['numEnv']
        self.collide = self.simParameters['collide']

        #gym stuff
        self.obs_space = spaces.Box(np.ones(self.num_obs)*-np.Inf, np.ones(self.num_obs)*np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_internal_states)*-np.Inf, np.ones(self.num_internal_states)*np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions)*-np.Inf, np.ones(self.num_actions)*np.Inf)

        self.track, self.tile_len, self.track_num_tiles = get_track(cfg, self.device)
        print("track loaded with ", self.track_num_tiles, " tiles")
        self.centerline = torch.tensor(self.track[1], dtype=torch.float, device=self.device, requires_grad=False)

        if not self.headless:
            self.viewer = Viewer(cfg, self.track)
        self.info = []
        
        #allocate env tensors
        torch_zeros = lambda shape: torch.zeros(shape, device=self.device, dtype= torch.float, requires_grad=False)
        self.states = torch_zeros((self.num_envs, self.num_agents, self.num_internal_states))
        self.contact_wrenches = torch_zeros((self.num_envs, self.num_agents, 3))
        self.actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.obs_buf = torch_zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch_zeros((self.num_envs, self.num_agents,))
        self.reset_buf = torch_zeros((self.num_envs, self.num_agents))>1


        allocate_car_dynamics_tensors(self)

        #other tensor
        self.active_track_tile = torch.zeros((self.num_envs, self.num_agents), dtype = torch.long, device=self.device, requires_grad=False)
        self.old_active_track_tile = torch.zeros_like(self.active_track_tile)
        #self.step(self.actions)
        self.reset()
        self.step(self.actions[:,0,:])
        self.viewer.center_cam(self.states)
        if not self.headless:
            self.render()

               
    def compute_observations(self,) -> None:
        #get points along centerline
        #get track boundary polys for every point
        #do stuff...
        ori = self.states[:,:,2].unsqueeze(2)
        vels = self.states[:,:,3:6]
        tile_idx = torch.remainder(self.active_track_tile.unsqueeze(2) + torch.arange(10, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0), self.track_num_tiles)
        lookahead = (self.centerline[tile_idx, :] - torch.tile(self.states[:,:,0:2].unsqueeze(2), (1,1,10, 1))).view(self.num_envs, self.num_agents, -1)
        self.obs_buf = torch.cat((ori, 
                                  vels, 
                                  0.1*lookahead), 
                                  dim=2)
        

    def compute_rewards(self,) -> None:
        
 
        rew_prog = self.active_track_tile - self.old_active_track_tile
        linecrossing_forward = torch.where(rew_prog < -self.track_num_tiles + 5)[0]
        linecrossing_backward = torch.where(rew_prog > self.track_num_tiles - 5)[0]
        
        #edge cases
        rew_prog[linecrossing_forward] = 1.0
        rew_prog[linecrossing_backward] = -1.0

        rew_prog *= 1 #self.rew_scales['progress']

        rew_on_track = 1.0*self.is_on_track # * self.rew_scales['progress']
        self.rew_buf = rew_prog + rew_on_track 

    def check_termination(self) -> None:
        self.reset_buf = ~self.is_on_track
        

    def reset(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, dtype = torch.long)
        self.reset_envs(env_ids)
        self.post_physics_step()
        return self.obs_buf[:, 0, :]

    def reset_envs(self, env_ids) -> None:
        
        for agent in range(self.num_agents):
            tile_idx = torch.randint(0, self.track_num_tiles, (len(env_ids),))  
            startpos = torch.tensor(self.track[1][tile_idx, :], device=self.device, dtype=torch.float).view(len(env_ids), 2)
            offset_x = np.cos(self.track[3][tile_idx] + np.pi/2)* self.modelParameters['L'] + np.cos(self.track[3][tile_idx])* self.modelParameters['L']
            offset_y = np.sin(self.track[3][tile_idx] + np.pi/2)* self.modelParameters['L'] + np.sin(self.track[3][tile_idx])* self.modelParameters['L']
            self.states[env_ids, agent, self.vn['S_X']] = startpos[:,0] #+ 0*offset_x
            self.states[env_ids, agent, self.vn['S_Y']] = startpos[:,1] #+ 0*offset_y
            self.states[env_ids, agent, self.vn['S_THETA']] = torch.tensor(self.track[3][tile_idx], device=self.device, dtype=torch.float) + np.pi/2 
            self.states[env_ids, agent, self.vn['S_THETA']+1:] = 0.0
        
        dists = torch.norm(self.states[env_ids,:, 0:2].unsqueeze(2)-self.centerline.unsqueeze(0).unsqueeze(0), dim=3)
        self.old_active_track_tile[env_ids, :] = torch.sort(dists, dim = 2)[1][:,:, 0]
    
    def step(self, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]] : 
        actions = actions.clone().to(self.device)
        if actions.requires_grad:
           actions = actions.detach()   
        self.actions[:,0,:] = actions
        self.actions[:,:,1] += 1.0
        self.actions[:,:,2] *= 0.1

        #print(self.actions[0,0,:])
        self.simulate()
        self.post_physics_step()
        return self.obs_buf[:,0, :], self.rew_buf[:,0], self.reset_buf[:,0], self.info
    
    def post_physics_step(self) -> None:
        #get current tile positions
        #dists = self.states[:,:, 0:2] - self.track[1].unsqueeze(0) 
        dists = torch.norm(self.states[:,:, 0:2].unsqueeze(2)-self.centerline.unsqueeze(0).unsqueeze(0), dim=3)
        self.active_track_tile = torch.sort(dists, dim = 2)[1][:,:, 0]

        #check if any tire is off track
        self.is_on_track = ~torch.any(~torch.any(self.wheels_on_track_segments, dim = 3), dim=2)
        
        self.check_termination()
        env_ids = torch.where(self.reset_buf)[0]
        if len(env_ids):
            self.reset_envs(env_ids)
        self.compute_observations()
        self.compute_rewards()

        self.old_active_track_tile = self.active_track_tile

        if not self.headless:
            self.render()

    def render(self,) -> None:
        self.viewer_events = self.viewer.render(self.states[:,:,[0,1,2,10]])

    
    def simulate(self) -> None:
        #run physics update
        self.states, self.contact_wrenches, self.wheels_on_track_segments = step_cars(self.states, 
                                                                                      self.actions, 
                                                                                      self.wheel_locations, 
                                                                                      self.R, 
                                                                                      self.contact_wrenches, 
                                                                                      self.modelParameters, 
                                                                                      self.simParameters, 
                                                                                      self.vn,
                                                                                      self.collision_pairs,
                                                                                      self.collision_verts,
                                                                                      self.P_tot,
                                                                                      self.D_tot,
                                                                                      self.S_mat,
                                                                                      self.Repf_mat,
                                                                                      self.Ds,
                                                                                      self.num_envs,
                                                                                      self.zero_pad,
                                                                                      self.collide,
                                                                                      self.wheels_on_track_segments,
                                                                                      self.track[4],
                                                                                      self.track[5],
                                                                                      self.track[6]
                                                                                     )
    def get_state(self):
        return self.states[:,0,:]

    #gym stuff
    @property
    def observation_space(self):
        return self.obs_space
    
    @property
    def action_space(self):
        return self.act_space
    