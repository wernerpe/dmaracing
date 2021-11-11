import torch
from torch._C import dtype
from gym import spaces
from dmaracing.env.car_dynamics import step_cars
from dmaracing.env.car_dynamics_utils import get_varnames, set_dependent_params, allocate_car_dynamics_tensors 
from dmaracing.env.viewer import Viewer
from dmaracing.utils.helpers import rand
from typing import Tuple, Dict, Union
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
        self.num_privileged_obs = None
        self.privileged_obs = None

        self.num_internal_states = self.simParameters['numStates']
        self.num_actions = self.simParameters['numActions']
        self.num_obs = self.simParameters['numObservations']       
        self.num_agents = self.simParameters['numAgents']
        self.num_envs = self.simParameters['numEnv']
        self.collide = self.simParameters['collide']
        self.decimation = self.simParameters['decimation']

        #gym stuff
        self.obs_space = spaces.Box(np.ones(self.num_obs)*-np.Inf, np.ones(self.num_obs)*np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_internal_states)*-np.Inf, np.ones(self.num_internal_states)*np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions)*-np.Inf, np.ones(self.num_actions)*np.Inf)

        self.track, self.tile_len, self.track_num_tiles = get_track(cfg, self.device)
        print("track loaded with ", self.track_num_tiles, " tiles")
        self.centerline = torch.tensor(self.track[1], dtype=torch.float, device=self.device, requires_grad=False)
        self.tile_heading = torch.tensor(self.track[3], device=self.device, dtype=torch.float, requires_grad=False) + np.pi/2
        if not self.headless:
            self.viewer = Viewer(cfg, self.track)
        self.info = {}
        
        #allocate env tensors
        torch_zeros = lambda shape: torch.zeros(shape, device=self.device, dtype= torch.float, requires_grad=False)
        self.states = torch_zeros((self.num_envs, self.num_agents, self.num_internal_states))
        self.contact_wrenches = torch_zeros((self.num_envs, self.num_agents, 3))
        self.actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.last_actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.obs_buf = torch_zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch_zeros((self.num_envs, self.num_agents,))
        self.reset_buf = torch_zeros((self.num_envs, self.num_agents))>1
        self.episode_length_buf = torch_zeros((self.num_envs,1))
        self.time_off_track = torch_zeros((self.num_envs, self.num_agents))
        self.dt = cfg['sim']['dt']
        self.reward_scales = {}
        self.reward_scales['offtrack'] = cfg['learn']['offtrackRewardScale']*self.dt
        self.reward_scales['contouring'] = cfg['learn']['contouringRewardScale']*self.dt
        self.reward_scales['progress'] = cfg['learn']['progressRewardScale']*self.dt
        self.reward_scales['actionrate'] = cfg['learn']['actionRateRewardScale']*self.dt

        self.default_actions = cfg['learn']['defaultactions']
        self.action_scales = cfg['learn']['actionscale']
        self.horizon = cfg['learn']['horizon']
        self.reset_randomization = cfg['learn']['resetrand']
        self.timeout_s = cfg['learn']['timeout']
        self.offtrack_reset_s = cfg['learn']['offtrack_reset']
        self.offtrack_reset = int(self.offtrack_reset_s/(self.decimation*self.dt))
        self.max_episode_length = int(self.timeout_s/(self.decimation*self.dt))

        allocate_car_dynamics_tensors(self)

        #other tensor
        self.active_track_tile = torch.zeros((self.num_envs, self.num_agents), dtype = torch.long, device=self.device, requires_grad=False)
        self.old_active_track_tile = torch.zeros_like(self.active_track_tile)
        self.old_track_progress = torch.zeros_like(self.rew_buf)
        self.reward_terms = {'progress': torch_zeros((self.num_envs, self.num_agents)), 
                             'contouring':torch_zeros((self.num_envs, self.num_agents)),
                             'offtrack': torch_zeros((self.num_envs, self.num_agents)),
                             'actionrate': torch_zeros((self.num_envs, self.num_agents))}
        
        #self.step(self.actions)
        self.reset()
        self.step(self.actions[:,0,:])
        self.viewer.center_cam(self.states)
        if not self.headless:
            self.render()

               
    def compute_observations(self,) -> None:
        #get points along centerline
        #get track boundary polys for every point
        steer =  self.states[:,:,self.vn['S_STEER']].unsqueeze(2)
        gas = self.states[:,:,self.vn['S_GAS']].unsqueeze(2)
        #brake = self.states[:,:,self.vn['S_BRAKE']]
        
        theta = self.states[:,:,2].unsqueeze(2)
        vels = self.states[:,:,3:6].clone()
        tile_idx = torch.remainder(self.active_track_tile.unsqueeze(2) + 4*torch.arange(self.horizon, device=self.device, dtype=torch.long).unsqueeze(0).unsqueeze(0), self.track_num_tiles)
        self.lookahead = (self.centerline[tile_idx, :] - torch.tile(self.states[:,:,0:2].unsqueeze(2), (1,1,self.horizon, 1)))
        
        self.R[:, :, 0, 0 ] = torch.cos(theta[:,:,0])
        self.R[:, :, 0, 1 ] = torch.sin(theta[:,:,0])
        self.R[:, :, 1, 0 ] = -torch.sin(theta[:,:,0])
        self.R[:, :, 1, 1 ] = torch.cos(theta[:,:,0])
        self.lookahead_body = torch.einsum('eaij, eatj->eati', self.R, self.lookahead)
        self.vels_body = vels
        self.vels_body[..., :-1] = torch.einsum('eaij, eaj -> eai',self.R, vels[..., :-1])
        
        self.obs_buf = torch.cat((self.vels_body,
                                  steer,
                                  gas, 
                                  0.2*self.lookahead_body[:,:,:,0],
                                  0.2*self.lookahead_body[:,:,:,1],
                                  self.last_actions
                                  ), 
                                  dim=2)
        

    def compute_rewards(self,) -> None:
        
        #print((self.active_track_tile[0,0])*self.tile_len)
        tile_car_vec = self.states[:,:, 0:2] - self.centerline[self.active_track_tile, :]        
        angs = self.tile_heading[self.active_track_tile]
        trackdir = torch.stack((torch.cos(angs), torch.sin(angs)), dim = 2) 
        trackperp = torch.stack((- torch.sin(angs), torch.cos(angs)), dim = 2) 

        sub_tile_progress = torch.einsum('eac, eac-> ea', trackdir, tile_car_vec)
        self.track_progress = self.active_track_tile[0,0]*self.tile_len + sub_tile_progress
        conturing_err = torch.einsum('eac, eac-> ea', trackperp, tile_car_vec)
        rew_progress = torch.clip(self.track_progress-self.old_track_progress, min = -10, max = 10) 
        rew_contouring = -torch.square(conturing_err) * self.reward_scales['contouring']
        rew_on_track = self.reward_scales['offtrack']*~self.is_on_track 
        rew_actionrate = -torch.sum(torch.square(self.actions-self.last_actions), dim = 2) *self.reward_scales['actionrate']
        self.rew_buf = torch.clip(rew_progress + rew_on_track + rew_contouring + rew_actionrate, min = 0, max = None)
        
        #clip rewards


        self.reward_terms['progress'] += rew_progress
        self.reward_terms['offtrack'] += rew_on_track
        self.reward_terms['contouring'] += rew_contouring
        self.reward_terms['actionrate'] += rew_actionrate
        
    def check_termination(self) -> None:
        self.reset_buf = self.time_off_track > self.offtrack_reset
        self.reset_buf |= self.episode_length_buf > self.max_episode_length

    def reset(self) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        env_ids = torch.arange(self.num_envs, dtype = torch.long)
        self.reset_envs(env_ids)
        self.post_physics_step()
        return self.obs_buf[:, 0, :], self.privileged_obs

    def reset_envs(self, env_ids) -> None:
        
        for agent in range(self.num_agents):
            tile_idx = torch.randint(0, self.track_num_tiles, (len(env_ids),))  
            startpos = torch.tensor(self.track[1][tile_idx, :], device=self.device, dtype=torch.float).view(len(env_ids), 2)
            angs = torch.tensor(self.track[3][tile_idx], device=self.device, dtype=torch.float) + np.pi/2
            vels_long = rand(-0.1*self.reset_randomization[3], self.reset_randomization[3], (len(env_ids),), device=self.device)
            vels_lat = rand(-self.reset_randomization[4], self.reset_randomization[4], (len(env_ids),), device=self.device)
            dir_x = torch.cos(angs)
            dir_y = torch.sin(angs)
            self.states[env_ids, agent, self.vn['S_X']] = startpos[:,0] + rand(-self.reset_randomization[0], self.reset_randomization[0], (len(env_ids),), device=self.device)
            self.states[env_ids, agent, self.vn['S_Y']] = startpos[:,1] + rand(-self.reset_randomization[1], self.reset_randomization[1], (len(env_ids),), device=self.device)
            self.states[env_ids, agent, self.vn['S_THETA']] = angs + rand(-self.reset_randomization[2], self.reset_randomization[2], (len(env_ids),), device=self.device) 
            self.states[env_ids, agent, self.vn['S_DX']] = dir_x * vels_long - dir_y*vels_lat
            self.states[env_ids, agent, self.vn['S_DY']] = dir_y * vels_long + dir_x*vels_lat
            self.states[env_ids, agent, self.vn['S_DTHETA']] = rand(-self.reset_randomization[5], self.reset_randomization[5], (len(env_ids),), device=self.device) 
            self.states[env_ids, agent, self.vn['S_DTHETA']+1:] = 0.0 
            self.states[env_ids, agent, self.vn['S_STEER']] = rand(-self.reset_randomization[6], self.reset_randomization[6], (len(env_ids),), device=self.device)
            #self.states[env_ids, agent, self.vn['S_W0']:self.vn['S_W3']+1] = 10.0
        
        dists = torch.norm(self.states[env_ids,:, 0:2].unsqueeze(2)-self.centerline.unsqueeze(0).unsqueeze(0), dim=3)
        self.old_active_track_tile[env_ids, :] = torch.sort(dists, dim = 2)[1][:,:, 0]
        self.episode_length_buf[env_ids] = 0
        self.old_track_progress [env_ids] = 0.0
        self.time_off_track[env_ids, :] = 0.0
        self.last_actions[env_ids, : ,:] = 0.0
        
        self.info = {}
        self.info['episode'] = {}
        
        for key in self.reward_terms.keys():
            self.info['episode']['reward_'+key] = (torch.mean(self.reward_terms[key][env_ids])/self.timeout_s).view(-1,1)
            self.reward_terms[key][env_ids] = 0.

    def step(self, actions) -> Tuple[torch.Tensor, Union[None, torch.Tensor], torch.Tensor, Dict[str, float]] : 
        actions = actions.clone().to(self.device)
        if actions.requires_grad:
           actions = actions.detach()   
        
        self.actions[:,0, 0] = self.action_scales[0]*actions[:,0] + self.default_actions[0]
        self.actions[:,0, 1] = self.action_scales[1]*actions[:,1] + self.default_actions[1]
        self.actions[:,0, 2] = self.action_scales[2]*actions[:,2] + self.default_actions[2]
    
        #print(self.actions[0,0,:])
        for _ in range(self.decimation):
            self.simulate()
        self.post_physics_step()
        return self.obs_buf[:,0, :].clone(), self.privileged_obs, self.rew_buf[:,0].clone(), self.reset_buf[:,0].clone(), self.info
    
    def post_physics_step(self) -> None:
        self.episode_length_buf +=1
        
        #get current tile positions
        dists = torch.norm(self.states[:,:, 0:2].unsqueeze(2)-self.centerline.unsqueeze(0).unsqueeze(0), dim=3)
        sort = torch.sort(dists, dim = 2)
        self.dist_active_track_tile = sort[0][:,:, 0]
        self.active_track_tile = sort[1][:,:, 0]
        
        #check if any tire is off track
        self.is_on_track = ~torch.any(~torch.any(self.wheels_on_track_segments, dim = 3), dim=2)
        self.time_off_track += 1.0*~self.is_on_track
        
        #get env_ids to reset
        self.check_termination()
        
        env_ids = torch.where(self.reset_buf)[0]
        if len(env_ids):
            self.reset_envs(env_ids)
        
        self.compute_observations()
        self.compute_rewards()
 
        #self.lookahead_markers = self.lookahead_body# + torch.tile(self.states[:,:,0:2].unsqueeze(2), (1,1,self.horizon, 1))
        #pts = self.lookahead_markers[self.viewer.env_idx_render,0,:,:].cpu().numpy()
        #self.viewer.clear_markers()
        #self.viewer.add_point(pts, 10,(5,10,222))

        self.old_active_track_tile = self.active_track_tile
        self.old_track_progress = self.track_progress
        self.last_actions = self.actions.clone()

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
    def get_state(self) -> torch.Tensor:
        return self.states[:,0,:]
    
    def get_observations(self,) -> torch.Tensor:
        return self.obs_buf[:,0,:]

    def get_privileged_observations(self,)->Union[None, torch.Tensor]:
        return self.privileged_obs
 
    #gym stuff
    @property
    def observation_space(self):
        return self.obs_space
    
    @property
    def action_space(self):
        return self.act_space
    