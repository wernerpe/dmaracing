from unicodedata import decimal
import torch
from torch._C import device, dtype
from gym import spaces
from dmaracing.env.car_dynamics import step_cars
from dmaracing.env.car_dynamics_utils import get_varnames, set_dependent_params, allocate_car_dynamics_tensors 
from dmaracing.env.viewer import Viewer
from dmaracing.utils.helpers import rand
from typing import Tuple, Dict, Union
import numpy as np
from dmaracing.utils.trackgen import get_track_ensemble
from torch.utils.tensorboard import SummaryWriter

class DmarEnv():
    def __init__(self, cfg, args) -> None:
        self.device = args.device
        self.rl_device = args.device
        self.headless = args.headless

        #variable names and indices
        self.vn = get_varnames() 
        self.cfg = cfg
        self.modelParameters = cfg['model']
        set_dependent_params(self.modelParameters)
        self.simParameters = cfg['sim']

        #use bootstrapping on vf
        self.use_timeouts = cfg['learn']['use_timeouts']
        
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
        
        #load track
        self.all_envs = torch.arange(self.num_envs, dtype = torch.long)
        self.num_tracks = cfg['track']['num_tracks']
        t, self.tile_len, self.track_tile_counts = get_track_ensemble(self.num_tracks, cfg, self.device)
        self.track_lengths = torch.tensor(self.track_tile_counts, device = self.device) * self.tile_len
        self.track_centerlines, self.track_poly_verts, self.track_alphas, self.track_A, self.track_b, self.track_S,\
        self.track_border_poly_verts, self.track_border_poly_cols = t
        self.track_alphas = self.track_alphas + np.pi/2

        self.active_track_mask = torch.zeros((self.num_envs, self.num_tracks), device = self.device, requires_grad=False,dtype = torch.float)
        self.active_track_ids = torch.randint(0, self.num_tracks, (self.num_envs, ), device = self.device, requires_grad=False, dtype = torch.long)
        self.active_track_mask[self.all_envs, self.active_track_ids] = 1.0
        self.active_centerlines = self.track_centerlines[self.active_track_ids]
        self.active_alphas =  self.track_alphas[self.active_track_ids]
        

        self.max_track_num_tiles = np.max(self.track_tile_counts)
        #take largest track polygon summation template
        self.track_S = self.track_S[np.argmax(self.track_tile_counts), :]
        self.track_tile_counts = torch.tensor(self.track_tile_counts, device=self.device, requires_grad=False)
        self.active_track_tile_counts = self.track_tile_counts[self.active_track_ids]

       
        #print("track loaded with ", self.track_num_tiles, " tiles")
        self.log_video_freq = cfg["viewer"]["logEvery"]
        self.viewer = Viewer(cfg,
                              self.track_centerlines, 
                              self.track_poly_verts, 
                              self.track_border_poly_verts, 
                              self.track_border_poly_cols,
                              self.track_tile_counts,
                              self.active_track_ids,
                              self.headless)
        self.info = {}
        
        #allocate env tensors
        torch_zeros = lambda shape: torch.zeros(shape, device=self.device, dtype= torch.float, requires_grad=False)
        self.states = torch_zeros((self.num_envs, self.num_agents, self.num_internal_states))
        self.contact_wrenches = torch_zeros((self.num_envs, self.num_agents, 3))
        self.shove = torch_zeros((self.num_envs, self.num_agents, 2))
        self.actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.actions_means = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.last_actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.obs_buf = torch_zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch_zeros((self.num_envs, self.num_agents,))
        self.time_out_buf = torch_zeros((self.num_envs, 1)) > 1
        self.reset_buf = torch_zeros((self.num_envs,1))>1
        self.episode_length_buf = torch_zeros((self.num_envs,1))
        self.time_off_track = torch_zeros((self.num_envs, self.num_agents))
        self.dt = cfg['sim']['dt']

        self.reward_scales = {}
        self.reward_scales['offtrack'] = cfg['learn']['offtrackRewardScale']*self.dt
        self.reward_scales['contouring'] = cfg['learn']['contouringRewardScale']*self.dt
        self.reward_scales['progress'] = cfg['learn']['progressRewardScale']*self.dt
        self.reward_scales['actionrate'] = cfg['learn']['actionRateRewardScale']*self.dt
        self.reward_scales['sidevel'] = cfg['learn']['sidevelRewardScale']*self.dt
        self.reward_scales['energy'] = cfg['learn']['energyRewardScale']*self.dt
        self.reward_scales['rank'] = cfg['learn']['rankRewardScale']*self.dt
        self.reward_scales['collision'] = cfg['learn']['collisionRewardScale']*self.dt
        

        self.default_actions = cfg['learn']['defaultactions']
        self.action_scales = cfg['learn']['actionscale']
        self.horizon = cfg['learn']['horizon']
        self.reset_randomization = cfg['learn']['resetrand']
        self.reset_tile_rand = cfg['learn']['reset_tile_rand']
        self.reset_grid = cfg['learn']['resetgrid']
        self.timeout_s = cfg['learn']['timeout']
        self.offtrack_reset_s = cfg['learn']['offtrack_reset']
        self.offtrack_reset = int(self.offtrack_reset_s/(self.decimation*self.dt))
        self.max_episode_length = int(self.timeout_s/(self.decimation*self.dt))
        self.agent_dropout_prob = cfg['learn']['agent_dropout_prob']

        self.lap_counter = torch.zeros((self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int)
        self.track_progress = torch.zeros((self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.float)
        
        allocate_car_dynamics_tensors(self)

        #other tensor
        self.active_track_tile = torch.zeros((self.num_envs, self.num_agents), dtype = torch.long, device=self.device, requires_grad=False)
        self.old_active_track_tile = torch.zeros_like(self.active_track_tile)
        self.old_track_progress = torch.zeros_like(self.rew_buf)
        self.reward_terms = {'progress': torch_zeros((self.num_envs, self.num_agents)), 
                             'contouring':torch_zeros((self.num_envs, self.num_agents)),
                             'offtrack': torch_zeros((self.num_envs, self.num_agents)),
                             'actionrate': torch_zeros((self.num_envs, self.num_agents)),
                             'sidevel': torch_zeros((self.num_envs, self.num_agents)),
                             'energy': torch_zeros((self.num_envs, self.num_agents)),
                             'rank': torch_zeros((self.num_envs, self.num_agents)),
                             'collision': torch_zeros((self.num_envs, self.num_agents)),
                             }
        
        self.is_collision = torch.zeros((self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype = torch.float)

        self.lookahead_scaler = 1/(4*(1+torch.arange(self.horizon, device = self.device , requires_grad=False, dtype=torch.float))*self.tile_len)
        self.lookahead_scaler = self.lookahead_scaler.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        self.ranks = torch.zeros((self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype = torch.int)

        self._global_step = 0
        if self.log_video_freq >= 0:
            self._logdir = cfg["logdir"]
            self._writer = SummaryWriter(log_dir=self._logdir, flush_secs=10)

        self.active_agents = torch_zeros((self.num_envs, self.num_agents)) == 0
        self.active_obs_template = torch.ones((self.num_envs, self.num_agents, self.num_agents-1), requires_grad=False, device=self.device)
        self.ado_idx_lookup = torch.zeros((self.num_agents, self.num_agents), requires_grad=False, device=self.device, dtype=torch.long)
        for row in range(self.num_agents):
            for col in range(self.num_agents):
                if col<row:
                    self.ado_idx_lookup[row, col] = row-1
                elif row<col:
                    self.ado_idx_lookup[row, col] = row
      

        self.total_step = 0
        self.viewer.center_cam(self.states)
        self.reset()
        self.step(self.actions)

        if not self.headless:
            self.viewer.center_cam(self.states)
            self.render()
        
        
               
    def compute_observations(self,) -> None:
        steer =  self.states[:,:,self.vn['S_STEER']].unsqueeze(2)
        gas = self.states[:,:,self.vn['S_GAS']].unsqueeze(2)
        
        theta = self.states[:,:,2]
        vels = self.states[:,:,3:6].clone()
        tile_idx_unwrapped = self.active_track_tile.unsqueeze(2) + (4*torch.arange(self.horizon, device=self.device, dtype=torch.long)).unsqueeze(0).unsqueeze(0)
        tile_idx = torch.remainder(tile_idx_unwrapped, self.active_track_tile_counts.view(-1,1,1))
        centers = self.active_centerlines[:, tile_idx, :]
        centers = centers[self.all_envs,self.all_envs, ...]
        angles_at_centers = self.active_alphas[:,tile_idx]
        angles_at_centers = angles_at_centers[self.all_envs, self.all_envs, ...]
        self.trackdir_lookahead = torch.stack((torch.cos(angles_at_centers), torch.sin(angles_at_centers)), dim = 3)
        self.interpolated_centers = centers + self.trackdir_lookahead*self.sub_tile_progress.view(self.num_envs, self.num_agents, 1, 1)
        
        self.lookahead = (self.interpolated_centers - torch.tile(self.states[:,:,0:2].unsqueeze(2), (1,1,self.horizon, 1)))
        
        self.R[:, :, 0, 0 ] = torch.cos(theta)
        self.R[:, :, 0, 1 ] = torch.sin(theta)
        self.R[:, :, 1, 0 ] = -torch.sin(theta)
        self.R[:, :, 1, 1 ] = torch.cos(theta)
        
        self.lookahead_body = torch.einsum('eaij, eatj->eati', self.R, self.lookahead)
        lookahead_scaled = self.lookahead_scaler*self.lookahead_body

        #otherpositions = []
        otherrotations = []
        othervelocities = []
        otherangularvelocities = []
        other_progress = []
        other_contouringerr = []

        if self.num_agents > 1:
            for agent in range(self.num_agents):
                selfpos = self.states[:, agent, 0:2].view(-1,1,2)
                selfrot = self.states[:, agent, 2].view(-1,1,1)
                selfvel = self.states[:, agent, self.vn['S_DX']: self.vn['S_DY']+1].view(-1,1,2)
                selfangvel = self.states[:, agent, self.vn['S_DTHETA']].view(-1,1,1)
                
                selftrackprogress = self.track_dist[:, agent].view(-1,1,1)
                other_progress.append(torch.cat((self.track_dist[:, :agent], self.track_dist[:, agent+1:]), dim = 1).view(-1, 1, self.num_agents-1) - selftrackprogress)

                selfcontouringerr = self.contouring_err[:, agent]
                other_contouringerr.append(torch.cat((self.contouring_err[:, :agent], self.contouring_err[:, agent+1:]), dim = 1).view(-1, 1, self.num_agents-1) - selfcontouringerr.view(-1,1,1))
                
                otherrotations.append(torch.cat((self.states[:, :agent, 2], self.states[:, agent+1:, 2]), dim = 1).view(-1, 1, self.num_agents-1) - selfrot)

                othervel = torch.cat((self.states[:, :agent, self.vn['S_DX']: self.vn['S_DY']+1], self.states[:, agent+1:, self.vn['S_DX']: self.vn['S_DY']+1]), dim = 1)
                othervelocities.append((othervel - selfvel).view(-1, (self.num_agents-1), 2))    
                otherangvel = torch.cat((self.states[:, :agent, self.vn['S_DTHETA']], self.states[:, agent+1:, self.vn['S_DTHETA']]), dim = 1).view(-1, 1, self.num_agents-1)
                otherangularvelocities.append(otherangvel - selfangvel)


            
            vel_other = torch.cat(tuple([vel.view(-1,1,(self.num_agents-1), 2) for vel in othervelocities]), dim = 1)*\
                        self.active_obs_template.view(self.num_envs, self.num_agents, self.num_agents-1, 1)
            vel_other = torch.einsum('eaij, eaoj->eaoi', self.R, vel_other)
            rot_other = torch.cat(tuple([rot for rot in otherrotations]),dim =1 ) * self.active_obs_template
            angvel_other = torch.cat(tuple([angvel for angvel in otherangularvelocities]),dim =1 ) * self.active_obs_template
            self.progress_other = torch.clip(torch.cat(tuple([prog for prog in other_progress]), dim =1 ), min = -50, max = 50) * self.active_obs_template
            self.contouring_err_other = torch.clip(torch.cat(tuple([err for err in other_contouringerr]), dim =1 ), min = -10, max = 10) * self.active_obs_template

        else:
            dist_other_clipped = torch.zeros((self.num_envs, 1, 1), device=self.device, dtype=torch.float, requires_grad=False)
            dir_other = torch.zeros((self.num_envs, 1, 2), device=self.device, dtype=torch.float, requires_grad=False)
                
        self.vels_body = vels
        self.vels_body[..., :-1] = torch.einsum('eaij, eaj -> eai',self.R, vels[..., :-1])
        
        
        self.obs_buf = torch.cat((self.vels_body*0.1, 
                                  steer, 
                                  gas, 
                                  lookahead_scaled[:,:,:,0], 
                                  lookahead_scaled[:,:,:,1], 
                                  self.progress_other * 0.1, 
                                  self.contouring_err_other * 0.25, 
                                  rot_other,
                                  vel_other[..., 0] * 0.1,
                                  vel_other[..., 1] * 0.1,
                                  angvel_other * 0.1, 
                                  self.last_actions, 
                                  self.ranks.view(-1,self.num_agents,1) 
                                  ), 
                                  dim=2)


    def compute_rewards(self,) -> None:  
        self.rew_buf, self.track_progress, self.reward_terms\
             = compute_rewards_jit(self.active_track_tile,
                                   self.tile_len,
                                   self.sub_tile_progress,
                                   self.contouring_err,
                                   self.old_track_progress,
                                   self.is_on_track,
                                   self.reward_scales,
                                   self.actions,
                                   self.last_actions,
                                   self.vn,
                                   self.states,
                                   self.vels_body,
                                   self.ranks,
                                   self.is_collision,
                                   self.reward_terms,
                                   self.num_agents,
                                   self.active_agents
                                   )        

    def check_termination(self) -> None:
        #dithering step
        #self.reset_buf = torch.rand((self.num_envs, 1), device=self.device) < 0.03
        if False and self.cfg['sim']['test_mode']:
            self.reset_buf = torch.max(1.0*(self.time_off_track[:, :] > self.offtrack_reset), dim = 1)[0].view(-1,1) > 0.0
        else:
            self.reset_buf = self.time_off_track[:, 0].view(-1,1) > self.offtrack_reset
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset(self) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        env_ids = torch.arange(self.num_envs, dtype = torch.long)
        self.reset_envs(env_ids)
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs

    def reset_envs(self, env_ids) -> None:
        self.resample_track(env_ids)
        self.active_agents[env_ids, 1:] = torch.rand((len(env_ids),self.num_agents-1), device=self.device) > self.agent_dropout_prob
        tile_idx_env = (torch.rand((len(env_ids),1), device=self.device) * self.active_track_tile_counts[env_ids].view(-1,1)).to(dtype=torch.long)
        tile_idx_env = torch.tile(tile_idx_env, (self.num_agents,))
        self.drag_reduction_points[env_ids,:,:] = 0.0
        self.drag_reduced[env_ids, :] = False

        if self.num_agents >1:
            if not self.reset_grid:
                if self.cfg['sim']['test_mode']:
                    rands = (torch.rand((len(env_ids),self.num_agents-1), device=self.device) * self.reset_tile_rand - self.reset_tile_rand/4).to(dtype=torch.long)
                else:    
                    rands = (torch.rand((len(env_ids),self.num_agents-1), device=self.device) * self.reset_tile_rand - self.reset_tile_rand/2).to(dtype=torch.long)
                tile_idx_env[:, 1:] = torch.remainder(tile_idx_env[:, 1:] + rands, self.active_track_tile_counts[env_ids].view(-1,1))
            else:
                tile_idx_env = tile_idx_env + 1 * torch.linspace(0, self.num_agents-1, self.num_agents, device=self.device).unsqueeze(0).to(dtype=torch.long)
                tile_idx_env = torch.remainder(tile_idx_env, self.active_track_tile_counts[env_ids].view(-1,1))
        positions = torch.ones((len(env_ids), self.num_agents), device=self.device).multinomial(self.num_agents, replacement=False)
        for agent in range(self.num_agents):
            if not self.reset_grid:
                tile_idx = tile_idx_env[:, agent] 
            else:
                tile_idx = torch.gather(tile_idx_env, 1, positions[:, agent].unsqueeze(-1)).squeeze()
            startpos = self.active_centerlines[env_ids, tile_idx, :]
            angs = self.active_alphas[env_ids, tile_idx]
            vels_long = rand(-0.1*self.reset_randomization[3], self.reset_randomization[3], (len(env_ids),), device=self.device)
            vels_lat = rand(-self.reset_randomization[4], self.reset_randomization[4], (len(env_ids),), device=self.device)
            dir_x = torch.cos(angs)
            dir_y = torch.sin(angs)
            if not self.reset_grid:
                self.states[env_ids, agent, self.vn['S_X']] = startpos[:,0] + rand(-self.reset_randomization[0], self.reset_randomization[0], (len(env_ids),), device=self.device)
                self.states[env_ids, agent, self.vn['S_Y']] = startpos[:,1] + rand(-self.reset_randomization[1], self.reset_randomization[1], (len(env_ids),), device=self.device)
            else:
                self.states[env_ids, agent, self.vn['S_X']] = startpos[:,0]
                self.states[env_ids, agent, self.vn['S_Y']] = startpos[:,1] + (-1)**(positions[:, agent]+1) * 2
            self.states[env_ids, agent, self.vn['S_THETA']] = angs + rand(-self.reset_randomization[2], self.reset_randomization[2], (len(env_ids),), device=self.device) 
            self.states[env_ids, agent, self.vn['S_DX']] = dir_x * vels_long - dir_y*vels_lat
            self.states[env_ids, agent, self.vn['S_DY']] = dir_y * vels_long + dir_x*vels_lat
            self.states[env_ids, agent, self.vn['S_DTHETA']] = rand(-self.reset_randomization[5], self.reset_randomization[5], (len(env_ids),), device=self.device) 
            self.states[env_ids, agent, self.vn['S_DTHETA']+1:] = 0.0 
            self.states[env_ids, agent, self.vn['S_STEER']] = rand(-self.reset_randomization[6], self.reset_randomization[6], (len(env_ids),), device=self.device)
            self.states[env_ids, agent, self.vn['S_W0']:self.vn['S_W3']+1] = (vels_long/self.modelParameters['WHEEL_R']).unsqueeze(1)
        
        idx_inactive, idx2_inactive = torch.where(~self.active_agents[env_ids, :].view(len(env_ids), self.num_agents))
        self.states[idx_inactive, idx2_inactive, self.vn['S_X']:self.vn['S_Y']+1] = 10000.0
        
        dists = torch.norm(self.states[env_ids,:, 0:2].unsqueeze(2)-self.active_centerlines[env_ids].unsqueeze(1), dim=3)
        self.old_active_track_tile[env_ids, :] = torch.sort(dists, dim = 2)[1][:,:, 0]
        
        self.info['episode'] = {}
        
        for key in self.reward_terms.keys():
            self.info['episode']['reward_'+key] = (torch.mean(self.reward_terms[key][env_ids, 0])/self.timeout_s).view(-1,1)
            self.reward_terms[key][env_ids] = 0.

        self.info['episode']['resetcount'] = len(env_ids)

        if self.use_timeouts:
            self.info['time_outs'] = self.time_out_buf.view(-1,)

        self.info['ranking'] = self.ranks[env_ids]
        self.info['percentage_max_episode_length'] = self.episode_length_buf[env_ids]/self.max_episode_length

        self.lap_counter[env_ids, :] = 0
        self.episode_length_buf[env_ids] = 0.0
        self.old_track_progress [env_ids] = 0.0
        self.time_off_track[env_ids, :] = 0.0
        self.last_actions[env_ids, : ,:] = 0.0

        if 0 in env_ids and self.log_video_freq >= 0:
          if len(self.viewer._frames):

              frames = np.stack(self.viewer._frames, axis=0)  # (T, W, H, C)
              frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, W, H)
              frames = np.expand_dims(frames, axis=0)  # (N, T, C, W, H)
              self._writer.add_video('Train/video', frames, global_step=self._global_step, fps=15)

          self.viewer._frames = []
          self._global_step += 1

    def step(self, actions) -> Tuple[torch.Tensor, Union[None, torch.Tensor], torch.Tensor, Dict[str, float]] : 
        actions = actions.clone().to(self.device)
        if actions.requires_grad:
           actions = actions.detach()   
        
        if len(actions.shape) == 2:
            self.actions[:, 0, 0] = self.action_scales[0]*actions[...,0] + self.default_actions[0]
            self.actions[:, 0, 1] = self.action_scales[1]*actions[...,1] + self.default_actions[1]
            self.actions[:, 0, 2] = self.action_scales[2]*actions[...,2] + self.default_actions[2]
        else:
            self.actions[:, :, 0] = self.action_scales[0]*actions[...,0] + self.default_actions[0]
            self.actions[:, :, 1] = self.action_scales[1]*actions[...,1] + self.default_actions[1]
            self.actions[:, :, 2] = self.action_scales[2]*actions[...,2] + self.default_actions[2]
        
        #reset collision detection
        self.is_collision = self.is_collision < 0 
        for _ in range(self.decimation):
            self.simulate()
            #need to check for collisions in inner loop otherwise get missed
            self.is_collision |= (torch.norm(self.contact_wrenches, dim = 2) > 0)
       
        self.post_physics_step()
        
        return self.obs_buf.clone(), self.privileged_obs, self.rew_buf.clone(), self.reset_buf.clone(), self.info
    
    def post_physics_step(self) -> None:
        self.total_step += 1
        self.episode_length_buf +=1
        
        #get current tile positions
        dists = torch.norm(self.states[:, :, 0:2].unsqueeze(2)-self.active_centerlines.unsqueeze(1), dim=3)
        sort = torch.sort(dists, dim = 2)
        self.dist_active_track_tile = sort[0][:,:, 0]
        self.active_track_tile = sort[1][:,:, 0]
        
        #check if any tire is off track
        self.is_on_track = ~torch.any(~torch.any(self.wheels_on_track_segments, dim = 3), dim=2)
        self.time_off_track += 1.0*~self.is_on_track
        
        #update drag_reduction
        self.drag_reduction_points[:, self.drag_reduction_write_idx:self.drag_reduction_write_idx+self.num_agents, :] = \
                        self.states[:,:,0:2]
        self.drag_reduction_write_idx = (self.drag_reduction_write_idx + self.num_agents) % self.drag_reduction_points.shape[1]
        leading_pts = self.states[:,:,0:2] + 5*torch.cat((torch.cos(self.states[:,:,2].view(self.num_envs, self.num_agents,1)),
                                                          torch.sin(self.states[:,:,2].view(self.num_envs, self.num_agents,1))), dim = 2)

        dists = leading_pts.view(self.num_envs, self.num_agents,1,2) - self.drag_reduction_points.view(self.num_envs, 1, -1, 2)
        dists = torch.min(torch.norm(dists, dim=3), dim=2)[0]
        self.drag_reduced[:] = False 
        self.drag_reduced = dists <= 3.0
        
        #get env_ids to reset
        self.time_out_buf *= False
        self.check_termination()
        
        env_ids = torch.where(self.reset_buf)[0]
        self.info = {}
        
        if len(env_ids):
            self.reset_envs(env_ids)
        
        self.info['agent_active'] = self.active_agents

        self.track_dist = self.track_progress + self.lap_counter*self.track_lengths[self.active_track_ids].view(-1,1)
        dist_sort, self.ranks = torch.sort(self.track_dist, dim = 1, descending = True)
        self.ranks = torch.sort(self.ranks, dim = 1)[1]

        #compute closest point on centerline
        self.tile_points = self.active_centerlines[:, self.active_track_tile, :]
        self.tile_points = self.tile_points[self.all_envs, self.all_envs, ...] 
        self.tile_car_vec = self.states[:,:, 0:2] - self.tile_points        
        angs = self.active_alphas[:,self.active_track_tile]
        angs = angs[self.all_envs, self.all_envs, ...]
        self.trackdir = torch.stack((torch.cos(angs), torch.sin(angs)), dim = 2) 
        self.trackperp = torch.stack((- torch.sin(angs), torch.cos(angs)), dim = 2) 

        self.sub_tile_progress = torch.einsum('eac, eac-> ea', self.trackdir, self.tile_car_vec)

        self.contouring_err = torch.einsum('eac, eac-> ea', self.trackperp, self.tile_car_vec)

        self.active_obs_template[...] = 1.0
        idx_1, idx_2 = torch.where(~self.active_agents)
        #remove ado observations of deactivated envs for active ones
        for ag in range(self.num_agents):
            self.active_obs_template[idx_1, ag, self.ado_idx_lookup[idx_2,  ag]] = 0


        self.compute_observations()
        self.compute_rewards()

        increment_idx = torch.where(self.active_track_tile - self.old_active_track_tile < -10)
        decrement_idx = torch.where(self.active_track_tile - self.old_active_track_tile > 10)
        self.lap_counter[increment_idx] += 1
        self.lap_counter[decrement_idx] -= 1
        
        #render before resetting values
        # if not self.headless:
        self.render()
        
        self.old_active_track_tile = self.active_track_tile
        self.old_track_progress = self.track_progress
        self.last_actions = self.actions.clone()

    def render(self,) -> None:
        #if log_video freq is set only redner in fixed intervals 
        if self.log_video_freq>=0:
            if (self._global_step % self.log_video_freq == 0) and (self._global_step > 0):
                self.viewer_events = self.viewer.render(self.states[:,:,[0,1,2,10]], self.slip, self.wheel_locations_world)
        else:
            self.viewer_events = self.viewer.render(self.states[:,:,[0,1,2,10]], self.slip, self.drag_reduced, self.wheel_locations_world)
    
    def simulate(self) -> None:
        #run physics update
        self.states, self.contact_wrenches, self.shove, self.wheels_on_track_segments, self.slip, self.wheel_locations_world\
                                                                                          = step_cars(
                                                                                            self.states, 
                                                                                            self.actions,
                                                                                            self.drag_reduced, 
                                                                                            self.wheel_locations, 
                                                                                            self.R, 
                                                                                            self.contact_wrenches,
                                                                                            self.shove, 
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
                                                                                            self.active_track_mask,
                                                                                            self.track_A, #Ax<=b
                                                                                            self.track_b,
                                                                                            self.track_S #how to sum
                                                                                            )

    def resample_track(self, env_ids) -> None:
        self.active_track_ids[env_ids] = torch.randint(0, self.num_tracks, (len(env_ids),), device = self.device, requires_grad=False, dtype = torch.long)
        self.active_track_mask[env_ids, ...] = 0.0
        self.active_track_mask[env_ids, self.active_track_ids[env_ids]] = 1.0
        self.active_centerlines[env_ids,...] = self.track_centerlines[self.active_track_ids[env_ids]]
        self.active_alphas[env_ids,...] = self.track_alphas[self.active_track_ids[env_ids]]
        self.active_track_tile_counts[env_ids] = self.track_tile_counts[self.active_track_ids[env_ids]]

        #update viewer
        if not self.headless:
            self.viewer.active_track_ids[env_ids] = self.active_track_ids[env_ids]
            #call refresh on track drawing only in render mode
            # if self.viewer.do_render:
            if self.viewer.env_idx_render in env_ids:
                self.viewer.draw_track_reset()
                
 
    def get_state(self) -> torch.Tensor:
        return self.states.squeeze()
    
    def get_observations(self,) -> torch.Tensor:
        return self.obs_buf.squeeze()

    def get_privileged_observations(self,)->Union[None, torch.Tensor]:
        return self.privileged_obs
    
    #gym stuffcompute_observations
    @property
    def observation_space(self):
        return self.obs_space
    
    @property
    def action_space(self):
        return self.act_space
    
### jit functions
@torch.jit.script
def compute_rewards_jit(active_track_tile : torch.Tensor,
                        tile_len : float,
                        sub_tile_progress : torch.Tensor,
                        contouring_err : torch.Tensor,
                        old_track_progress : torch.Tensor,
                        is_on_track : torch.Tensor,
                        reward_scales : Dict[str, float],
                        actions : torch.Tensor,
                        last_actions : torch.Tensor,
                        vn : Dict[str, int],
                        states : torch.Tensor,
                        vels_body : torch.Tensor,
                        ranks : torch.Tensor,
                        is_collision : torch.Tensor,
                        reward_terms : Dict[str, torch.Tensor],
                        num_agents : int,
                        active_agents : torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

            track_progress = active_track_tile*tile_len + sub_tile_progress
            rew_progress = torch.clip(track_progress-old_track_progress, min = -10, max = 10) * reward_scales['progress']
            rew_contouring = -torch.square(0.1*contouring_err) * reward_scales['contouring']
            rew_on_track = reward_scales['offtrack']*~is_on_track 
            rew_actionrate = -torch.sum(torch.square(actions-last_actions), dim = 2) *reward_scales['actionrate']
            rew_energy = -torch.sum(torch.square(states[:,:,vn['S_W0']:vn['S_W3']+1]), dim = 2)*reward_scales['energy']
            rew_sidevel = -torch.square(vels_body[:,:,1])*reward_scales['sidevel']
            num_active_agents = torch.sum(1.0*active_agents, dim = 1)
            rew_rank = 1.0/num_agents*(num_active_agents.view(-1,1)/2.0-ranks)*reward_scales['rank']
            
            rew_collision = is_collision*reward_scales['collision']
            rew_buf = torch.clip(rew_progress + rew_on_track + rew_contouring + rew_actionrate + rew_sidevel + rew_energy + rew_rank + rew_collision, min = 0, max = None)

            reward_terms['progress'] += rew_progress
            reward_terms['offtrack'] += rew_on_track
            reward_terms['contouring'] += rew_contouring
            reward_terms['actionrate'] += rew_actionrate
            reward_terms['energy'] += rew_energy
            reward_terms['sidevel'] += rew_sidevel
            reward_terms['rank'] += rew_rank
            reward_terms['collision'] += rew_collision
            return rew_buf, track_progress, reward_terms
