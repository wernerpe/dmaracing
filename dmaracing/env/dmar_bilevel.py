import torch
from torch._C import device, dtype
#from gym import spaces
from dmaracing.env.viewer import Viewer
from dmaracing.utils.helpers import rand
from typing import Tuple, Dict, Union
import numpy as np
from dmaracing.utils.trackgen import get_track_ensemble
from dmaracing.utils.trackgen_tri import get_tri_track_ensemble
from torch.utils.tensorboard import SummaryWriter
import random

# Add Dynamics directory to python path. Change this path to match that of your local system!
import sys
from gym import spaces
#sys.path.insert(1, "/home/thomasbalch/tri_workspace/dynamics_model_learning/scripts")
# Import Dynamics encoder from TRI dynamics library.
from dmaracing.env.car_dynamics_utils import get_varnames, set_dependent_params, allocate_car_dynamics_tensors
from dynamics_lib import DynamicsEncoder
from dmaracing.env.car_dynamics import step_cars
from dmaracing.controllers.purepursuit import PPController


class DmarEnvBilevel:
    def __init__(self, cfg, args) -> None:
        torch.manual_seed(cfg["track"]["seed"])
        random.seed(cfg["track"]["seed"])
        self.device = args.device
        self.rl_device = args.device
        self.headless = args.headless
        self.test_mode = cfg.get('test', False)
        # variable names and indices
        self.vn = get_varnames()
        self.cfg = cfg
        self.modelParameters = cfg["model"]
        set_dependent_params(self.modelParameters)
        self.simParameters = cfg["sim"]

        # use bootstrapping on vf
        self.use_timeouts = cfg["learn"]["use_timeouts"]
        self.track_stats = cfg["trackRaceStatistics"]

        self.num_states = 0
        self.num_privileged_obs = None
        self.privileged_obs = None

        self.num_internal_states = self.simParameters["numStates"]
        self.num_actions = self.simParameters["numActions"]
        self.num_obs = self.simParameters["numObservations"]
        self.num_agents = self.simParameters["numAgents"]
        self.num_envs = self.simParameters["numEnv"]
        self.collide = self.simParameters["collide"]
        self.decimation = self.simParameters["decimation"]
        self.num_actions_hl = self.simParameters["numActionsHL"]
        self.num_obs_add_ll = self.simParameters["numConstantObservationsLL"]

        # Import TRI dynamics model and weights
        self.dyn_model = DynamicsEncoder.load_from_checkpoint(
            #"/home/peter/git/dynamics_model_learning/sample_models/fixed_integration_current_v25.ckpt").to(self.device)
            "dynamics_models/"+cfg['model']['dynamics_model_name'],
            hparams_file="dynamics_models/"+cfg['model']['hparams_path'], strict=False).to(self.device)
        self.dyn_model.integration_function.initialize_lstm_states(torch.zeros((self.num_envs * self.num_agents, 50, 6)).to(self.device))
        
        if self.test_mode:
            self.dyn_model.dynamics_integrator.dyn_model.set_test_mode()
        self.dyn_model.dynamics_integrator.dyn_model.num_agents = self.num_agents
        self.dyn_model.dynamics_integrator.dyn_model.init_noise_vec(self.num_envs, self.device)
        self.dyn_model.integration_function.dyn_model.init_col_switch(self.num_envs, self.cfg['model']['col_decay_time'], self.device)
        #self.dyn_model.integration_function.dyn_model.gp.noise_lvl = self.cfg['model']['gp_noise_scale']
        
        self.noise_level = self.cfg['model']['noise_level']
        
        # gym stuff
        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(
            np.ones(self.num_internal_states) * -np.Inf, np.ones(self.num_internal_states) * np.Inf
        )
        self.act_space = spaces.Box(np.ones(self.num_actions) * -np.Inf, np.ones(self.num_actions) * np.Inf)

        # load track
        self.all_envs = torch.arange(self.num_envs, dtype=torch.long)
        #self.num_tracks = cfg["track"]["num_tracks"]
        self.track_half_width = cfg["track"]["track_half_width"]
        self.track_poly_spacing = cfg["track"]["track_poly_spacing"]
        t, self.tile_len, self.track_tile_counts, self.num_tracks, self.track_names = get_tri_track_ensemble(self.device, self.track_half_width, self.track_poly_spacing)
        self.track_lengths = torch.tensor(self.track_tile_counts, device=self.device) * self.tile_len
        self.track_successes = torch.ones((self.num_tracks,1), device=self.device, dtype=torch.float, requires_grad=False)
        
        (
            self.track_centerlines,
            self.track_poly_verts,
            self.track_alphas,
            self.track_A,
            self.track_b,
            self.track_S,
            self.track_border_poly_verts,
            self.track_border_poly_cols,
        ) = t
        self.track_alphas = self.track_alphas + np.pi / 2

        self.active_track_mask = torch.zeros(
            (self.num_envs, self.num_tracks), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.active_track_ids = torch.randint(
            0, self.num_tracks, (self.num_envs,), device=self.device, requires_grad=False, dtype=torch.long
        )
        self.active_track_mask[self.all_envs, self.active_track_ids] = 1.0
        self.active_centerlines = self.track_centerlines[self.active_track_ids]
        self.active_alphas = self.track_alphas[self.active_track_ids]

        self.max_track_num_tiles = np.max(self.track_tile_counts)
        # take largest track polygon summation template
        self.track_S = self.track_S[np.argmax(self.track_tile_counts), :]
        self.track_tile_counts = torch.tensor(self.track_tile_counts, device=self.device, requires_grad=False)
        self.active_track_tile_counts = self.track_tile_counts[self.active_track_ids]

        # print("track loaded with ", self.track_num_tiles, " tiles")
        self.log_video_freq = cfg["viewer"]["logEvery"]
        self.viewer = Viewer(
            cfg,
            self.track_centerlines,
            self.track_poly_verts,
            self.track_border_poly_verts,
            self.track_border_poly_cols,
            self.track_tile_counts,
            self.active_track_ids,
            self.headless,
        )
        self.info = {}

        # allocate env tensors
        torch_zeros = lambda shape: torch.zeros(shape, device=self.device, dtype=torch.float, requires_grad=False)
        self.states = torch_zeros((self.num_envs, self.num_agents, self.num_internal_states))
        self.contact_wrenches = torch_zeros((self.num_envs, self.num_agents, 3))
        self.shove = torch_zeros((self.num_envs, self.num_agents, 3))
        self.actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.actions_means = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.last_actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.last_steer = torch_zeros((self.num_envs, self.num_agents, 1))
        self.last_vel = torch_zeros((self.num_envs, self.num_agents,))
        self.last_last_steer = torch_zeros((self.num_envs, self.num_agents, 1))

        self.ll_steps_left = torch_zeros((self.num_envs, self.num_agents, 1))
        
        self.obs_buf = torch_zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch_zeros(
            (
                self.num_envs,
                self.num_agents,
                2
            )
        )
        self.time_out_buf = torch_zeros((self.num_envs, 1)) > 1
        self.reset_buf = torch_zeros((self.num_envs, 1)) > 1
        self.agent_left_track = torch_zeros((self.num_envs, self.num_agents)) > 1
        self.episode_length_buf = torch_zeros((self.num_envs, 1))
        self.time_off_track = torch_zeros((self.num_envs, self.num_agents))
        self.dt = cfg["sim"]["dt"]

        self.dt_hl = cfg["learn"]["episode_length_ll"]

        self.reward_scales = {}
        self.reward_scales["offtrack"] = cfg["learn"]["offtrackRewardScale"] * self.dt
        self.reward_scales["progress"] = cfg["learn"]["progressRewardScale"] * self.dt
        self.reward_scales["actionrate"] = cfg["learn"]["actionRateRewardScale"] * self.dt
        self.reward_scales["rank"] = cfg["learn"]["rankRewardScale"] * self.dt
        self.reward_scales["collision"] = cfg["learn"]["collisionRewardScale"] * self.dt
        self.reward_scales["acceleration"] = cfg["learn"]["accRewardScale"] * self.dt

        self.reward_scales["goal"] = cfg["learn"]["goalRewardScale"]

        self.default_actions = cfg["learn"]["defaultactions"]
        self.action_scales = cfg["learn"]["actionscale"]
        self.action_scales_hl = cfg["learn"]["actionscalehl"]
        self.horizon = cfg["learn"]["horizon"]
        self.race_length_laps = cfg['learn']['race_length_laps']
        self.reset_randomization = cfg["learn"]["resetrand"]
        self.reset_tile_rand = cfg["learn"]["reset_tile_rand"]
        self.reset_grid = cfg["learn"]["resetgrid"]
        self.timeout_s = cfg["learn"]["timeout"]
        self.offtrack_reset_s = cfg["learn"]["offtrack_reset"]
        self.offtrack_reset = int(self.offtrack_reset_s / (self.decimation * self.dt))
        self.max_episode_length = int(self.timeout_s / (self.decimation * self.dt))
        self.agent_dropout_prob = cfg["learn"]["agent_dropout_prob"]

        self.action_min_hl = cfg["learn"]["actionsminhl"]
        self.action_max_hl = cfg["learn"]["actionsmaxhl"]
        self.action_ini_hl = cfg["learn"]["actionsinihl"]

        self.obs_noise_lvl = cfg["learn"]["obs_noise_lvl"]
        self.use_track_curriculum = cfg["learn"]["use_track_curriculum"]

        self.steer_ado_ppc = True if self.cfg['learn']['steer_ado_with_PPC'] and self.num_agents>1 else False
        
        if self.steer_ado_ppc:
            print('[DMAR] Overriding ado actions with PPC')
            self.ppc = PPController(self,
                       lookahead_dist=1.5,
                       maxvel=2.5,
                       k_steer=1.0,
                       k_gas=2.0)

        self.lap_counter = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int
        )
        self.track_progress = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.float
        )
        self.track_progress_no_laps = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.float
        )

        allocate_car_dynamics_tensors(self)

        # other tensor
        self.active_track_tile = torch.zeros(
            (self.num_envs, self.num_agents), dtype=torch.long, device=self.device, requires_grad=False
        )
        self.old_active_track_tile = torch.zeros_like(self.active_track_tile)
        self.old_track_progress = torch.zeros_like(self.time_off_track)
        self.reward_terms = {
            "progress": torch_zeros((self.num_envs, self.num_agents)),
            "offtrack": torch_zeros((self.num_envs, self.num_agents)),
            "actionrate": torch_zeros((self.num_envs, self.num_agents)),
            "energy": torch_zeros((self.num_envs, self.num_agents)),
            "acceleration": torch_zeros((self.num_envs, self.num_agents)),
            "goal": torch_zeros((self.num_envs, self.num_agents)),
        }
        if self.num_agents>1:
            self.reward_terms["rank"] = torch_zeros((self.num_envs, self.num_agents))
            self.reward_terms["collision"] = torch_zeros((self.num_envs, self.num_agents))

        self.is_collision = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.float
        )

        self.lookahead_scaler = 1 / (
            4
            * (1 + torch.arange(self.horizon, device=self.device, requires_grad=False, dtype=torch.float))
            * self.tile_len
        )
        self.lookahead_scaler = self.lookahead_scaler.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        self.ranks = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int
        )

        # HACK: targets
        self.targets_pos_world = torch.zeros(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_dist_world = torch.zeros(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_dist_local = torch.zeros(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_std_world = torch.zeros(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_std_local = torch.zeros(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_rot_world = torch.zeros(
            (self.num_envs, self.num_agents, 1), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_tile_pos = torch.zeros(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_tile_idx = torch.zeros(
            (self.num_envs, self.num_agents, 1), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_dist_track = torch.ones(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.targets_std_track = torch.ones(
            (self.num_envs, self.num_agents, 2), device=self.device, requires_grad=False, dtype=torch.float
        )
        self.ll_ep_done = torch.zeros(
            (self.num_envs, self.num_agents, 1), device=self.device, requires_grad=False, dtype=torch.float
        )
        
        self._action_probs_hl, self._action_mean_ll, self._action_std_ll, self.targets_rew01_local  = None, None, None, None

        self._global_step = 0
        if self.log_video_freq >= 0:
            self._logdir = cfg["logdir"]
            self._writer = SummaryWriter(log_dir=self._logdir, flush_secs=10)

        self.active_agents = torch_zeros((self.num_envs, self.num_agents)) == 0
        self.active_obs_template = torch.ones(
            (self.num_envs, self.num_agents, self.num_agents - 1), requires_grad=False, device=self.device
        )
        self.ado_idx_lookup = torch.zeros(
            (self.num_agents, self.num_agents), requires_grad=False, device=self.device, dtype=torch.long
        )
        for row in range(self.num_agents):
            for col in range(self.num_agents):
                if col < row:
                    self.ado_idx_lookup[row, col] = row - 1
                elif row < col:
                    self.ado_idx_lookup[row, col] = row

        self.trained_agent_slot = 0

        if self.track_stats:
            self.stats_overtakes_current = torch.zeros((self.num_envs), device = self.device)
            self.stats_overtakes_last_race = torch.zeros((self.num_envs), device = self.device)
            self.stats_num_collisions_current = torch.zeros((self.num_envs), device = self.device)
            self.stats_num_collisions_last_race = torch.zeros((self.num_envs), device = self.device)
            self.stats_ego_left_track_current = torch.zeros((self.num_envs), device = self.device)
            self.stats_ego_left_track_last_race = torch.zeros((self.num_envs), device = self.device)
            self.stats_ado_left_track_current = torch.zeros((self.num_envs), device = self.device)
            self.stats_ado_left_track_last_race = torch.zeros((self.num_envs), device = self.device)
            self.stats_start_from_behind_current = torch.zeros((self.num_envs), device = self.device)
            self.stats_win_from_behind_last_race = torch.zeros((self.num_envs), device = self.device)
            self.stats_lead_time_current = torch.zeros((self.num_envs), device = self.device)
            self.stats_lead_time_last_race = torch.zeros((self.num_envs), device = self.device)
            self.stats_last_ranks = torch.zeros((self.num_envs, self.num_agents), device = self.device)
            self.stats_percentage_sa_laptime_last_race = torch.zeros((self.num_envs), device = self.device)
            #track_order 
            #track_names = ['sharp_turns_track_ccw', 'sharp_turns_track_cw', 'orca_ccw', 'orca_cw', 'oversize1_ccw', 'oversize1_cw']

            self.maxvel_to_laptime_in_s_model = torch.tensor([[ 18.76698252,  -6.93244121,   1.09663193],
                                                    [ 19.64517843,  -7.39382932,   1.1109327 ],
                                                    [ 60.76572465, -22.33884921,   3.12345654],
                                                    [ 69.6452333 , -29.56069015,   4.58618748],
                                                    [ 56.92511095, -22.11645183,   3.11423204],
                                                    [ 57.72302405, -22.85980533,   3.29237065]], device=self.device)

        self.train_ll = True

        self.total_step = 0
        self.viewer.center_cam(self.states)
        self.reset()
        self.step(self.actions)

        # Re-init as reset & step increment total_step via post_physics_step
        self.total_step = 0
        self.episode_length_buf = torch_zeros((self.num_envs, 1))  # NOTE: counteract initial +1

        # if not self.headless:
        self.viewer.center_cam(self.states)
        self.render()
        print(10*'#'+' Randomized Vars '+10*'#')
        print('dynamics model:\n')
        print(self.dyn_model.integration_function.dyn_model.randomized_params)
        print('env:\n')
        print('obs noise' if self.test_mode else '')

    def set_train_level(self, low_level=False, high_level=False):
        if not low_level and not high_level:
            print("Neither low-level nor high-level training set")
            print("Defaulting to low-level training ...")
            self.train_ll = True
        elif low_level and high_level:
            print("Only one of {low, high)-level training should be set")
            print("Defaulting to low-level training ...")
            self.train_ll = True
        elif low_level:
            self.train_ll = True
        elif high_level:
            self.train_ll = False

    def track_to_world_coords(self, target_offset_track, tile_length):

        # Account for subtile progress + offset remainder adding 1 tile
        target_posx_track = target_offset_track[..., 0] + self.sub_tile_progress.view(self.num_envs, self.num_agents, 1, 1) + tile_length/2.0
      
        # Get tile lookahead ID and tranform to local center offset
        tile_idx = target_posx_track.div(tile_length, rounding_mode="floor")
        target_offset_track[..., 0] = torch.remainder(target_posx_track[..., 0], tile_length) - tile_length / 2.0

        angles_at_centers = self.active_alphas[:, tile_idx]
        angles_at_centers = angles_at_centers[self.all_envs, self.all_envs, ...]

    def coord_trafo(self, vec, angle):
        """angle: rotation from old to new coord system"""
        vec_x_rot = torch.sum(torch.concat((torch.cos(angle), torch.sin(angle)), dim=-1) * vec, dim=-1)
        vec_y_rot = torch.sum(torch.concat((-torch.sin(angle), torch.cos(angle)), dim=-1) * vec, dim=-1)
        return torch.stack([vec_x_rot, vec_y_rot], axis=-1)

    def set_hl_action_probs(self, probs):
        self._action_probs_hl = probs

    def set_ll_action_stats(self, mean, std):
        self._action_mean_ll = mean
        self._action_std_ll = std

        self._action_mean_ll[..., 0] = self.action_scales[0] * self._action_mean_ll[..., 0] + self.default_actions[0]
        self._action_mean_ll[..., 1] = self.action_scales[1] * self._action_mean_ll[..., 1] + self.default_actions[1]

        self._action_std_ll[..., 0] = self.action_scales[0] * self._action_std_ll[..., 0]
        self._action_std_ll[..., 1] = self.action_scales[1] * self._action_std_ll[..., 1]

    def project_into_track_frame(self, actions_hl):
        actions_hl = actions_hl.clone().to(self.device)
        # actions_hl *= torch.tensor(self.action_scales_hl[:2], device=self.device)

        target_offset_on_centerline_track = actions_hl[..., 0:2]
        target_std_local = actions_hl[..., 2:4]  # .unsqueeze(dim=1)  # NOTE: changed for multi-agent
        # self.ll_steps_left = actions_hl[..., 4]
        # self.ll_steps_left = (self.dt_hl - torch.remainder(self.total_step * torch.ones_like(self.last_steer), self.dt_hl)) / self.dt_hl
        # self.ll_steps_left = self.ll_steps_left.unsqueeze(dim=1)
        # self.ll_steps_left = (self.dt_hl - torch.remainder(self.total_step * torch.ones_like(self.ll_steps_left), self.dt_hl)) / self.dt_hl
        self.ll_steps_left *= 0.0
        self.ll_steps_left += (self.dt_hl - torch.remainder(self.episode_length_buf.unsqueeze(dim=1), self.dt_hl)) / self.dt_hl


        # target_offset_on_centerline_track[..., 0] += 4.0

        # target_offset_on_centerline_track = target_offset_on_centerline_track.unsqueeze(dim=1)  # NOTE: changed for multi-agent
        
        tile_idx_unwrapped = self.active_track_tile.unsqueeze(2) + (
            4 * torch.arange(self.horizon, device=self.device, dtype=torch.long)
        ).unsqueeze(0).unsqueeze(0)
        tile_idx = torch.remainder(tile_idx_unwrapped, self.active_track_tile_counts.view(-1, 1, 1))
        centers = self.active_centerlines[:, tile_idx, :]
        centers = centers[self.all_envs, self.all_envs, ...]
        angles_at_centers = self.active_alphas[:, tile_idx]
        angles_at_centers = angles_at_centers[self.all_envs, self.all_envs, ...]
        self.trackdir_lookahead = torch.stack((torch.cos(angles_at_centers), torch.sin(angles_at_centers)), dim=3)
        self.interpolated_centers = centers + self.trackdir_lookahead * self.sub_tile_progress.view(
            self.num_envs, self.num_agents, 1, 1
        )

        car_offset_from_center_world = self.states[:, :, :2] - centers[:, :, 0]
        car_offset_from_center_local = self.coord_trafo(car_offset_from_center_world, angles_at_centers[:, :, 0].unsqueeze(-1))  # relative pos in local ~= track
        target_offset_from_car_track = car_offset_from_center_local + target_offset_on_centerline_track
        target_offset_from_car_tiles = torch.div(target_offset_from_car_track[..., 0], self.tile_len, rounding_mode="floor")
        target_offset_from_center_track = torch.zeros_like(target_offset_from_car_track)
        target_offset_from_center_track[..., 0] = torch.remainder(target_offset_from_car_track[..., 0], self.tile_len)
        target_offset_from_center_track[..., 1] = target_offset_on_centerline_track[..., 1]

        target_tile_idx = torch.remainder(tile_idx[..., 0] + target_offset_from_car_tiles, self.active_track_tile_counts.view(-1, 1)).type(torch.long)
        target_tile_center = self.active_centerlines[:, target_tile_idx, :]
        target_tile_center = target_tile_center[self.all_envs, self.all_envs, ...]
        target_rot_world = self.active_alphas[:, target_tile_idx]
        target_rot_world = target_rot_world[self.all_envs, self.all_envs, ...].unsqueeze(-1)

        target_offset_local = target_offset_from_center_track
        target_offset_world = self.coord_trafo(target_offset_local, -target_rot_world)
        target_pos_world = target_tile_center + target_offset_world


        # Decide whether to update targets for LL training --> reset calls post_physics_step which +1 steps
        update_target = 1.0 * (torch.remainder(self.episode_length_buf.unsqueeze(-1), self.dt_hl)==0)
        self.targets_pos_world = update_target * target_pos_world + (1.0 - update_target) * self.targets_pos_world
        self.targets_std_local = update_target * target_std_local + (1.0 - update_target) * self.targets_std_local
        self.targets_rot_world = update_target * target_rot_world + (1.0 - update_target) * self.targets_rot_world
        self.targets_tile_idx = update_target * target_tile_idx.unsqueeze(-1) + (1.0 - update_target) * self.targets_tile_idx
        self.targets_tile_pos = update_target * target_offset_local + (1.0 - update_target) * self.targets_tile_pos

        # Compute distance to target in track coordinates
        target_tiles_from_car = torch.remainder(self.targets_tile_idx.squeeze(-1) - tile_idx[..., 0], self.active_track_tile_counts.view(-1, 1))  # account for wrap-around
        target_dist_track_x = self.targets_tile_pos[..., 0] - car_offset_from_center_local[..., 0] + self.tile_len * target_tiles_from_car
        target_dist_track_y = self.targets_tile_pos[..., 1] - car_offset_from_center_local[..., 1]
        target_dist_track = torch.stack([target_dist_track_x, target_dist_track_y], dim=-1)

        # Set observation data
        self.targets_dist_track = target_dist_track
        self.targets_std_track = self.targets_std_local


        self.ll_ep_done = 1.0 * (torch.remainder((self.episode_length_buf.unsqueeze(-1)+1), self.dt_hl)==0)

        # Plot ellipsis for 0.1*(max reward) region
        # Quantile function: Q(p) = sqrt(2) * sigma * inverf(2*p-1) + mu [https://statproofbook.github.io/P/norm-qf.html]
        reward_cut = 0.9 + 0.0 * self.targets_std_local
        self.targets_rew01_local = 2**(1/2) * self.targets_std_local * torch.erfinv(2*reward_cut-1.0) + 0.0
        self.targets_rew01_local = self.targets_rew01_local[:, 0].unsqueeze(dim=1)  # NOTE: only care about plotting for ego agent

        actions_hl = torch.cat(
            (
                self.targets_dist_track,
                self.targets_std_track,
                self.ll_steps_left,
            ),
            dim=2,
            )
        return actions_hl

    def compute_observations(
        self,
    ) -> None:
        #steer = self.states[:, :, self.vn["S_STEER"]].unsqueeze(2)
        #gas = self.states[:, :, self.vn["S_GAS"]].unsqueeze(2)

        theta = self.states[:, :, 2]
        vels = self.states[:, :, 3:6].clone()
        tile_idx_unwrapped = self.active_track_tile.unsqueeze(2) + (
            4 * torch.arange(self.horizon, device=self.device, dtype=torch.long)
        ).unsqueeze(0).unsqueeze(0)
        tile_idx = torch.remainder(tile_idx_unwrapped, self.active_track_tile_counts.view(-1, 1, 1))
        centers = self.active_centerlines[:, tile_idx, :]
        centers = centers[self.all_envs, self.all_envs, ...]
        angles_at_centers = self.active_alphas[:, tile_idx]
        angles_at_centers = angles_at_centers[self.all_envs, self.all_envs, ...]
        self.trackdir_lookahead = torch.stack((torch.cos(angles_at_centers), torch.sin(angles_at_centers)), dim=3)
        self.interpolated_centers = centers + self.trackdir_lookahead * self.sub_tile_progress.view(
            self.num_envs, self.num_agents, 1, 1
        )

        # # ### Setting random target locations (TODO: check for bugs)
        # # # Get target track tile center and rotation
        # # target_tiles_ahead = 5
        # # target_tile_center = centers[..., target_tiles_ahead, :]
        # # target_rot_world = angles_at_centers[..., target_tiles_ahead].unsqueeze(-1)
        # # # Generate random target offset
        # # target_offset_mag = 2.0*torch.rand(size=target_tile_center.shape, dtype=torch.float, device=self.device) - 1.0
        # # target_offset_local = target_offset_mag * torch.tensor([0.5*self.tile_len, self.track_half_width], dtype=torch.float, device=self.device)
        # # target_offset_world = self.coord_trafo(target_offset_local, -target_rot_world)
        # # target_pos_world = target_tile_center + target_offset_world

        # # Alternative target computation (based on continuous target offset)
        # target_offset_on_centerline_track = torch.zeros_like(centers[..., 0, :])  # set x offset
        # target_offset_on_centerline_track[..., 0] = 1.0 * (2.0*torch.rand(size=target_offset_on_centerline_track[..., 0].shape, dtype=torch.float, device=self.device) - 1.0)
        # target_offset_on_centerline_track[..., 1] = 0.8*self.track_half_width * (2.0*torch.rand(size=target_offset_on_centerline_track[..., 1].shape, dtype=torch.float, device=self.device) - 1.0)

        # target_offset_on_centerline_track[..., 0] += 3.5

        # car_offset_from_center_world = self.states[:, :, :2] - centers[:, :, 0]
        # car_offset_from_center_local = self.coord_trafo(car_offset_from_center_world, angles_at_centers[:, :, 0].unsqueeze(-1))  # relative pos in local ~= track
        # target_offset_from_car_track = car_offset_from_center_local + target_offset_on_centerline_track
        # target_offset_from_car_tiles = torch.div(target_offset_from_car_track[..., 0], self.tile_len, rounding_mode="floor")
        # target_offset_from_center_track = torch.zeros_like(target_offset_from_car_track)
        # target_offset_from_center_track[..., 0] = torch.remainder(target_offset_from_car_track[..., 0], self.tile_len)
        # target_offset_from_center_track[..., 1] = target_offset_on_centerline_track[..., 1]

        # target_tile_idx = torch.remainder(tile_idx[..., 0] + target_offset_from_car_tiles, self.active_track_tile_counts.view(-1, 1)).type(torch.long)
        # target_tile_center = self.active_centerlines[:, target_tile_idx, :]
        # target_tile_center = target_tile_center[self.all_envs, self.all_envs, ...]
        # target_rot_world = self.active_alphas[:, target_tile_idx]
        # target_rot_world = target_rot_world[self.all_envs, self.all_envs, ...].unsqueeze(-1)

        # target_offset_local = target_offset_from_center_track
        # target_offset_world = self.coord_trafo(target_offset_local, -target_rot_world)
        # target_pos_world = target_tile_center + target_offset_world

        # # Generate random uncertainties (between [0.5*max, 1.0*max])
        # target_std_mag = 0.5 * (1.0 + torch.rand(size=target_tile_center.shape, dtype=torch.float, device=self.device))
        # std_max_xtrack = 0.50
        # std_max_ytrack = 0.25
        # target_std_local = target_std_mag * torch.tensor([std_max_xtrack, std_max_ytrack], dtype=torch.float, device=self.device)

        # # Decide whether to update targets
        # self.ll_episode_length = 20
        # update_target = 1.0 * (torch.remainder(self.episode_length_buf.unsqueeze(-1), self.ll_episode_length)==1)
        # self.targets_pos_world = update_target * target_pos_world + (1.0 - update_target) * self.targets_pos_world
        # self.targets_std_local = update_target * target_std_local + (1.0 - update_target) * self.targets_std_local
        # self.targets_rot_world = update_target * target_rot_world + (1.0 - update_target) * self.targets_rot_world

        # self.targets_tile_idx = update_target * target_tile_idx.unsqueeze(-1) + (1.0 - update_target) * self.targets_tile_idx
        # self.targets_tile_pos = update_target * target_offset_local + (1.0 - update_target) * self.targets_tile_pos

        # # Compute distance to target in track coordinates
        # target_tiles_from_car = torch.remainder(self.targets_tile_idx.squeeze(-1) - tile_idx[..., 0], self.active_track_tile_counts.view(-1, 1))  # account for wrap-around
        # target_dist_track_x = self.targets_tile_pos[..., 0] - car_offset_from_center_local[..., 0] + self.tile_len * target_tiles_from_car
        # target_dist_track_y = self.targets_tile_pos[..., 1] - car_offset_from_center_local[..., 1]
        # target_dist_track = torch.stack([target_dist_track_x, target_dist_track_y], dim=-1)

        # # Set observation data
        # # self.targets_dist_world = self.targets_pos_world - self.states[:, 0, 0:2].view(-1, 1, 2)  # probably better expressed in track local coordinates
        # # self.targets_dist_local = self.coord_trafo(self.targets_dist_world, self.targets_rot_world)
        # # self.targets_std_world = self.coord_trafo(self.targets_std_local, -self.targets_rot_world)
        # self.targets_dist_track = target_dist_track
        # self.targets_std_track = self.targets_std_local
        # self.ll_steps_left = self.ll_episode_length - torch.remainder(self.episode_length_buf.unsqueeze(-1), self.ll_episode_length)
        # self.ll_steps_left /= self.ll_episode_length

        # self.ll_ep_done = 1.0 * (torch.remainder(self.episode_length_buf.unsqueeze(-1), self.ll_episode_length)==0)

        # # Plot ellipsis for 0.1*(max reward) region
        # # Quantile function: Q(p) = sqrt(2) * sigma * inverf(2*p-1) + mu [https://statproofbook.github.io/P/norm-qf.html]
        # reward_cut = 0.9 + 0.0 * self.targets_std_local
        # self.targets_rew01_local = 2**(1/2) * self.targets_std_local * torch.erfinv(2*reward_cut-1.0) + 0.0

        # # ##############################################################################################################

        trackdir_lookahead_rbound = torch.stack((torch.sin(angles_at_centers), -torch.cos(angles_at_centers)), dim = 3)        
        trackdir_lookahead_lbound = torch.stack((-torch.sin(angles_at_centers), torch.cos(angles_at_centers)), dim = 3)        
        interpolated_rbound = self.interpolated_centers + self.track_half_width * trackdir_lookahead_rbound
        interpolated_lbound = self.interpolated_centers + self.track_half_width * trackdir_lookahead_lbound
        self.interpolated_bounds = torch.concat((interpolated_rbound, interpolated_lbound), dim=-1)

        self.lookahead = self.interpolated_centers - torch.tile(
            self.states[:, :, 0:2].unsqueeze(2), (1, 1, self.horizon, 1)
        )
        
        self.lookahead_rbound = (self.interpolated_bounds[..., [0, 1]] - torch.tile(self.states[:,:,0:2].unsqueeze(2), (1,1,self.horizon, 1)))
        self.lookahead_lbound = (self.interpolated_bounds[..., [2, 3]] - torch.tile(self.states[:,:,0:2].unsqueeze(2), (1,1,self.horizon, 1)))

        self.R[:, :, 0, 0] = torch.cos(theta)
        self.R[:, :, 0, 1] = torch.sin(theta)
        self.R[:, :, 1, 0] = -torch.sin(theta)
        self.R[:, :, 1, 1] = torch.cos(theta)

        # self.lookahead_body = torch.einsum("eaij, eatj->eati", self.R, self.lookahead)
        # lookahead_scaled = self.lookahead_scaler * self.lookahead_body

        self.lookahead_rbound_body = torch.einsum('eaij, eatj->eati', self.R, self.lookahead_rbound)
        lookahead_rbound_scaled = self.lookahead_scaler*self.lookahead_rbound_body
        self.lookahead_lbound_body = torch.einsum('eaij, eatj->eati', self.R, self.lookahead_lbound)
        lookahead_lbound_scaled = self.lookahead_scaler*self.lookahead_lbound_body


        self.vels_body = vels
        self.vels_body[..., :-1] = torch.einsum("eaij, eaj -> eai", self.R, vels[..., :-1])

        otherpositions = []
        otherrotations = []
        othervelocities = []
        otherangularvelocities = []
        other_progress = []
        other_contouringerr = []

        other_progress_w_laps = []

        if self.num_agents > 1:
            
            for agent in range(self.num_agents):
                selfpos = self.states[:, agent, 0:2].view(-1, 1, 2)
                selfrot = self.states[:, agent, 2].view(-1, 1, 1)
                selfvel = self.states[:, agent, self.vn["S_DX"] : self.vn["S_DY"] + 1].view(-1, 1, 2)
                selfangvel = self.states[:, agent, self.vn["S_DTHETA"]].view(-1, 1, 1)

                otherpos = torch.cat((self.states[:, :agent, 0:2], self.states[:, agent + 1 :, 0:2]), dim=1)
                otherpositions.append((otherpos - selfpos).view(-1, (self.num_agents - 1), 2))

                selftrackprogress = self.track_progress_no_laps[:, agent].view(-1, 1, 1)
                oth_prog = (
                    torch.cat(
                        (self.track_progress_no_laps[:, :agent], self.track_progress_no_laps[:, agent + 1 :]), dim=1
                    ).view(-1, 1, self.num_agents - 1)
                    - selftrackprogress
                )
                
                oth_progress_w_laps = torch.clip(
                    torch.cat(
                        (self.track_progress[:, :agent], self.track_progress[:, agent + 1 :]), dim=1
                    ).view(-1, 1, self.num_agents - 1)
                    - self.track_progress[:, agent].view(-1, 1, 1), -self.cfg["learn"]["distance_obs_cutoff"], self.cfg["learn"]["distance_obs_cutoff"]
                ) 

                oth_prog = (
                    oth_prog
                    + 1.0
                    * (oth_prog < -0.51 * self.track_lengths[self.active_track_ids].view(-1, 1, 1))
                    * self.track_lengths[self.active_track_ids].view(-1, 1, 1)
                    - 1.0
                    * (oth_prog > 0.51 * self.track_lengths[self.active_track_ids].view(-1, 1, 1))
                    * self.track_lengths[self.active_track_ids].view(-1, 1, 1)
                )

                other_progress.append(oth_prog)
                other_progress_w_laps.append(oth_progress_w_laps)

                selfcontouringerr = self.contouring_err[:, agent]
                other_contouringerr.append(
                    torch.cat((self.contouring_err[:, :agent], self.contouring_err[:, agent + 1 :]), dim=1).view(
                        -1, 1, self.num_agents - 1
                    )
                    - selfcontouringerr.view(-1, 1, 1)
                )

                otherrotations.append(
                    torch.cat((self.states[:, :agent, 2], self.states[:, agent + 1 :, 2]), dim=1).view(
                        -1, 1, self.num_agents - 1
                    )
                    - selfrot
                )

                othervel = torch.cat(
                    (
                        self.states[:, :agent, self.vn["S_DX"] : self.vn["S_DY"] + 1],
                        self.states[:, agent + 1 :, self.vn["S_DX"] : self.vn["S_DY"] + 1],
                    ),
                    dim=1,
                )
                othervelocities.append((othervel - selfvel).view(-1, (self.num_agents - 1), 2))
                otherangvel = torch.cat(
                    (self.states[:, :agent, self.vn["S_DTHETA"]], self.states[:, agent + 1 :, self.vn["S_DTHETA"]]),
                    dim=1,
                ).view(-1, 1, self.num_agents - 1)
                otherangularvelocities.append(otherangvel - selfangvel)

            pos_other = torch.cat(tuple([pos.view(-1, 1, (self.num_agents - 1), 2) for pos in otherpositions]), dim=1)
            pos_other = torch.einsum("eaij, eaoj->eaoi", self.R, pos_other)
            norm_pos_other = torch.norm(pos_other, dim=3).view(self.num_envs, self.num_agents, -1)
            is_other_close = norm_pos_other < self.cfg["learn"]["distance_obs_cutoff"]

            vel_other = torch.cat(
                tuple([vel.view(-1, 1, (self.num_agents - 1), 2) for vel in othervelocities]), dim=1
            ) * self.active_obs_template.view(self.num_envs, self.num_agents, self.num_agents - 1, 1)

            vel_other = torch.einsum("eaij, eaoj->eaoi", self.R, vel_other) * is_other_close.view(
                self.num_envs, self.num_agents, self.num_agents - 1, 1
            )

            rot_other = (
                torch.cat(tuple([rot for rot in otherrotations]), dim=1) * self.active_obs_template * is_other_close
            )
            angvel_other = (
                torch.cat(tuple([angvel for angvel in otherangularvelocities]), dim=1)
                * self.active_obs_template
                * is_other_close
            )

            self.progress_other = (
                torch.clip(torch.cat(tuple([prog for prog in other_progress]), dim=1), min=-2.5, max=2.5)
                * self.active_obs_template - 2.5*(1-self.active_obs_template)
            )

            self.progress_other_w_laps = (
                torch.clip(torch.cat(tuple([prog for prog in other_progress_w_laps]), dim=1), min=-2.5, max=2.5)
                * self.active_obs_template - 2.5*(1-self.active_obs_template)
            )
            self.contouring_err_other = (
                torch.clip(torch.cat(tuple([err for err in other_contouringerr]), dim=1), min=-1, max=1)
                * self.active_obs_template
                * is_other_close
            )
            last_raw_actions = self.last_actions 
            
            last_raw_actions[:,:, 0 ] = (last_raw_actions[:,:, 0 ] - self.default_actions[0])/self.action_scales[0]  
            last_raw_actions[:,:, 1] = (last_raw_actions[:,:, 1] - self.default_actions[1])/self.action_scales[1]  
            
            #maxvel obs self.dyn_model.dynamics_integrator.dyn_model.max_vel_vec
            self.obs_buf = torch.cat(
                (
                    self.vels_body * 0.1,
                    lookahead_rbound_scaled[:,:,:,0], 
                    lookahead_rbound_scaled[:,:,:,1], 
                    lookahead_lbound_scaled[:,:,:,0], 
                    lookahead_lbound_scaled[:,:,:,1],
                    self.dyn_model.dynamics_integrator.dyn_model.max_vel_vec,
                    last_raw_actions,
                    self.progress_other * 0.1,
                    self.contouring_err_other * 0.25,
                    torch.sin(rot_other),
                    torch.cos(rot_other),
                    vel_other[..., 0] * 0.1,
                    vel_other[..., 1] * 0.1,
                    angvel_other * 0.1,
                ),
                dim=2,
            )
        else:
            last_raw_actions = self.last_actions 
            last_raw_actions[:,:,0] = torch.clip(last_raw_actions[:,:,0], min = -0.35, max= 0.35)
            last_raw_actions[:,:,1] = torch.clip(last_raw_actions[:,:,1], min = -0.3, max= 1.3)

            last_raw_actions[:,:, 0] = (last_raw_actions[:,:, 0] - self.default_actions[0])/self.action_scales[0]  
            last_raw_actions[:,:, 1] = (last_raw_actions[:,:, 1] - self.default_actions[1])/self.action_scales[1]  
            self.obs_buf = torch.cat(
            (
                self.vels_body * 0.1,
                lookahead_rbound_scaled[:,:,:,0], 
                lookahead_rbound_scaled[:,:,:,1], 
                lookahead_lbound_scaled[:,:,:,0], 
                lookahead_lbound_scaled[:,:,:,1],
                self.dyn_model.dynamics_integrator.dyn_model.max_vel_vec, 
                last_raw_actions,
                # self.targets_dist_world,
                # self.targets_std_world,
                # self.targets_dist_track,
                # self.targets_std_track,
                # self.ll_steps_left,
            ),
            dim=2,
            )

        if self.obs_noise_lvl>0 and not self.test_mode:

            noise_vec = torch.randn_like(self.obs_buf) * 0.1 * self.obs_noise_lvl
            self.obs_buf += noise_vec


    def compute_rewards(
        self,
    ) -> None:
        self.rew_buf, self.reward_terms = compute_rewards_jit(
            self.rew_buf,
            self.track_progress,
            self.old_track_progress,
            self.progress_other_w_laps if self.num_agents>1 else None,
            self.is_on_track,
            self.reward_scales,
            self.actions,
            self.last_actions,
            self.vn,
            self.states,
            self.vel,
            self.last_vel,
            self.ranks if self.num_agents>1 else None,
            self.is_collision,
            self.reward_terms,
            self.num_agents,
            self.active_agents,
            self.dt,
            # self.targets_dist_local,
            # self.targets_std_local,
            self.targets_dist_track,
            self.targets_std_track,
            self.ll_ep_done,
            1.0 * self.train_ll,
        )

    def check_termination(self) -> None:
        # dithering step
        # self.reset_buf = torch.rand((self.num_envs, 1), device=self.device) < 0.00005
        self.reset_buf = self.time_off_track[:, self.trained_agent_slot].view(-1, 1) > self.offtrack_reset
        lapdist = self.track_progress[:,self.trained_agent_slot].view(-1, 1)/self.track_lengths[self.active_track_ids].view(-1,1)
        #self.reset_buf |= lapdist > self.race_length_laps
        # self.reset_buf |= (torch.abs(self.track_progress[:,0] - self.old_track_progress[:,0]).view(-1,1) > 1.0)*(self.episode_length_buf > 2)  # NOTE: commented this because it was terminating a lot (maybe due to off-->on track jumps?)
        # self.reset_buf = torch.any(self.time_off_track[:, :] > self.offtrack_reset, dim = 1).view(-1,1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        if not self.train_ll:  # FIXME: check this
            self.reset_buf = torch.logical_and(self.reset_buf, torch.remainder((self.total_step) * torch.ones_like(self.reset_buf), self.dt_hl)==0)

    def reset(self) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        print("Resetting")
        env_ids = torch.arange(self.num_envs, dtype=torch.long)
        self.reset_envs(env_ids)
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs

    def reset_envs(self, env_ids) -> None:
        self.dyn_model.integration_function.dyn_model.reset(env_ids)
        if self.track_stats:
            last_track_ids = self.active_track_ids[env_ids].clone()
        self.resample_track(env_ids)
        self.active_agents[env_ids, 1:] = (
            torch.rand((len(env_ids), self.num_agents - 1), device=self.device) > self.agent_dropout_prob
        )
        tile_idx_env = (
            torch.rand((len(env_ids), 1), device=self.device) * self.active_track_tile_counts[env_ids].view(-1, 1)
        ).to(dtype=torch.long)
        if self.test_mode:
            tile_idx_env *=0 
        tile_idx_env = torch.tile(tile_idx_env, (self.num_agents,))
        self.drag_reduction_points[env_ids, :, :] = 0.0
        self.drag_reduced[env_ids, :] = False

        if self.num_agents > 1:
            if not self.reset_grid:
                if self.cfg["sim"]["test_mode"]:
                    rands = (
                        torch.rand((len(env_ids), self.num_agents - 1), device=self.device) * self.reset_tile_rand
                        - self.reset_tile_rand / 4
                    ).to(dtype=torch.long)
                else:
                    rands = (
                        torch.rand((len(env_ids), self.num_agents - 1), device=self.device) * self.reset_tile_rand
                        - self.reset_tile_rand / 2
                    ).to(dtype=torch.long)
                tile_idx_env[:, 1:] = torch.remainder(
                    tile_idx_env[:, 1:] + rands, self.active_track_tile_counts[env_ids].view(-1, 1)
                )
            else:
                tile_idx_env = tile_idx_env + 3 * torch.linspace(
                    0, self.num_agents - 1, self.num_agents, device=self.device
                ).unsqueeze(0).to(dtype=torch.long)
                tile_idx_env = torch.remainder(tile_idx_env, self.active_track_tile_counts[env_ids].view(-1, 1))
        positions = torch.ones((len(env_ids), self.num_agents), device=self.device).multinomial(
            self.num_agents, replacement=False
        )
        for agent in range(self.num_agents):
            if not self.reset_grid:
                tile_idx = tile_idx_env[:, agent]
            else:
                tile_idx = torch.gather(tile_idx_env, 1, positions[:, agent].unsqueeze(-1)).squeeze()
            startpos = self.active_centerlines[env_ids, tile_idx, :]
            angs = self.active_alphas[env_ids, tile_idx]
            vels_long = rand(
                -0.1 * self.reset_randomization[3], self.reset_randomization[3], (len(env_ids),), device=self.device
            )
            vels_lat = rand(
                -self.reset_randomization[4], self.reset_randomization[4], (len(env_ids),), device=self.device
            )
            dir_x = torch.cos(angs)
            dir_y = torch.sin(angs)
            if not self.reset_grid:
                self.states[env_ids, agent, self.vn["S_X"]] = startpos[:, 0] + rand(
                    -self.reset_randomization[0], self.reset_randomization[0], (len(env_ids),), device=self.device
                )
                self.states[env_ids, agent, self.vn["S_Y"]] = startpos[:, 1] + rand(
                    -self.reset_randomization[1], self.reset_randomization[1], (len(env_ids),), device=self.device
                )
            else:
                self.states[env_ids, agent, self.vn["S_X"]] = (
                    startpos[:, 0] - dir_y * (-1) ** (positions[:, agent] + 1) * 0.2
                )
                self.states[env_ids, agent, self.vn["S_Y"]] = (
                    startpos[:, 1] + dir_x * (-1) ** (positions[:, agent] + 1) * 0.2
                )
            self.states[env_ids, agent, self.vn["S_THETA"]] = angs + rand(
                -self.reset_randomization[2], self.reset_randomization[2], (len(env_ids),), device=self.device
            )
            self.states[env_ids, agent, self.vn["S_DX"]] = vels_long
            self.states[env_ids, agent, self.vn["S_DY"]] = 0#dir_y * vels_long + dir_x * vels_lat
            self.states[env_ids, agent, self.vn["S_DTHETA"]] = rand(
                -self.reset_randomization[5], self.reset_randomization[5], (len(env_ids),), device=self.device
            )
            self.states[env_ids, agent, self.vn["S_DTHETA"] + 1 :] = 0.0
            # self.states[env_ids, agent, self.vn["S_STEER"]] = rand(
            #     -self.reset_randomization[6], self.reset_randomization[6], (len(env_ids),), device=self.device
            # )
            # self.states[env_ids, agent, self.vn["S_W0"] : self.vn["S_W3"] + 1] = (
            #     vels_long / self.modelParameters["WHEEL_R"]
            # ).unsqueeze(1)

            # HACK: reset goal to current position?

        idx_inactive, idx2_inactive = torch.where(~self.active_agents[env_ids, :].view(len(env_ids), self.num_agents))
        if len(idx_inactive):
            self.states[
                env_ids[idx_inactive], idx2_inactive, self.vn["S_X"] : self.vn["S_Y"] + 1
            ] = 10000.0 + 1000 * torch.rand((len(idx2_inactive), 2), device=self.device, requires_grad=False)

        dists = torch.norm(
            self.states[env_ids, :, 0:2].unsqueeze(2) - self.active_centerlines[env_ids].unsqueeze(1), dim=3
        )
        self.old_active_track_tile[env_ids, :] = torch.sort(dists, dim=2)[1][:, :, 0]
        self.active_track_tile[env_ids, :] = torch.sort(dists, dim=2)[1][:, :, 0]
        
        self.info["episode"] = {}

        for key in self.reward_terms.keys():
            self.info["episode"]["reward_" + key] = (
                torch.mean(self.reward_terms[key][env_ids, 0]) / self.timeout_s
            ).view(-1, 1)
            self.reward_terms[key][env_ids] = 0.0

        self.info["episode"]["resetcount"] = len(env_ids)

        if self.use_timeouts:
            self.info["time_outs"] = self.time_out_buf.view(
                -1,
            )
        if self.num_agents>1:
            self.info["ranking"] = [
                torch.mean(
                    (1.0 * self.ranks[env_ids, :] * ~self.agent_left_track[env_ids, :])
                    + 1.0 * self.num_agents * self.agent_left_track[env_ids, :],
                    dim=0,
                ),
                1.0 * len(env_ids) / (1.0 * len(self.reset_buf)),
            ]
        #        self.info['ranking'] = self.ranks[env_ids]
        #        self.info['percentage_max_episode_length'] = 1.0*self.episode_length_buf[env_ids]/(self.max_episode_length)
        if self.track_stats:
            #compute percentage of laptime achieved
            mv = self.dyn_model.dynamics_integrator.dyn_model.max_vel_vec[env_ids, 0]
            X = torch.cat(tuple([mv.view(-1,1)**idx for idx in range(3)]), dim = -1)
            w = self.maxvel_to_laptime_in_s_model[last_track_ids, :]
            sa_lap_time_targets = torch.einsum('ef, ef -> e', X, w)
            time_ep = (self.episode_length_buf[env_ids]*self.dt*self.decimation).view(-1,)
            prog_ep = torch.clip(self.track_progress[env_ids, 0].view(-1,), min = 0)
            track_len_ep = self.track_lengths[last_track_ids].view(-1,)
            lap_time_actual = (track_len_ep/ (prog_ep+1e-9)) * time_ep 
            self.stats_percentage_sa_laptime_last_race[env_ids] = lap_time_actual/sa_lap_time_targets
            
            self.info["proginfo"] = [last_track_ids, 
            self.episode_length_buf[env_ids].clone(), 
            self.lap_counter[env_ids].clone(), 
            self.track_progress[env_ids].clone(), 
            self.track_progress_no_laps[env_ids].clone(),
            self.track_lengths[last_track_ids],
            #env_ids.clone(),
            self.dyn_model.dynamics_integrator.dyn_model.max_vel_vec[env_ids].clone()]
            
        #dynamics randomization
        self.dyn_model.dynamics_integrator.dyn_model.update_noise_vec(env_ids, self.noise_level) 

        self.lap_counter[env_ids, :] = 0
        self.episode_length_buf[env_ids] = 0.0
        self.old_track_progress[env_ids] = 0.0
        self.track_progress[env_ids] = 0.0
        self.track_progress_no_laps[env_ids] = 0.0
        self.time_off_track[env_ids, :] = 0.0
        self.last_actions[env_ids, :, :] = 0.0
        self.last_steer[env_ids, :, :] = 0.0
        self.last_last_steer[env_ids, :, :] = 0.0
        self.last_vel[env_ids, ...] = 0.0
        self.agent_left_track[env_ids, :] = False

        self.ll_ep_done[env_ids, ...] = 0.0  # FIXME: remove?

        if 0 in env_ids and self.log_video_freq >= 0:
            if len(self.viewer._frames):

                frames = np.stack(self.viewer._frames, axis=0)  # (T, W, H, C)
                frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, W, H)
                frames = np.expand_dims(frames, axis=0)  # (N, T, C, W, H)
                self._writer.add_video("Train/video", frames, global_step=self._global_step, fps=15)

            self.viewer._frames = []
            self._global_step += 1

    def step(self, actions) -> Tuple[torch.Tensor, Union[None, torch.Tensor], torch.Tensor, Dict[str, float]]:
        actions = actions.clone().to(self.device)
        if actions.requires_grad:
            actions = actions.detach()

        if self.steer_ado_ppc:
            actions[:,1:,:] = self.ppc.step()[:,1:,:]
        if len(actions.shape) == 2:
            self.actions[:, 0, 0] = self.action_scales[0] * actions[..., 0] + self.default_actions[0]
            self.actions[:, 0, 1] = self.action_scales[1] * actions[..., 1] + self.default_actions[1]
        #    self.actions[:, 0, 2] = self.action_scales[2] * actions[..., 2] + self.default_actions[2]
            #self.actions = torch.clamp(self.actions, min = -1, max = 1)        
        else:
            self.actions[:, :, 0] = self.action_scales[0] * actions[..., 0] + self.default_actions[0]
            self.actions[:, :, 1] = self.action_scales[1] * actions[..., 1] + self.default_actions[1]
        #    self.actions[:, :, 2] = self.action_scales[2] * actions[..., 2] + self.default_actions[2]

        # reset collision detection
        self.is_collision = self.is_collision < 0
        for j in range(self.decimation):
            self.simulate()
            # need to check for collisions in inner loop otherwise get missed
            self.is_collision |= torch.norm(self.contact_wrenches, dim=2) > 0

        self.post_physics_step()
        # if self.num_agents>1:
        #     return self.obs_buf.clone(), self.privileged_obs, self.rew_buf.clone(), self.reset_buf.clone(), self.info
        # else:
        #     return self.obs_buf[:,0,:].clone(), self.privileged_obs, self.rew_buf[:,0].clone(), self.reset_buf[:,0].clone(), self.info
        return self.obs_buf.clone(), self.privileged_obs, self.rew_buf.clone(), self.reset_buf.clone(), self.info

    def post_physics_step(self) -> None:
        self.total_step += 1
        self.episode_length_buf += 1

        # get current tile positions
        dists = torch.norm(self.states[:, :, 0:2].unsqueeze(2) - self.active_centerlines.unsqueeze(1), dim=3)
        sort = torch.sort(dists, dim=2)
        self.dist_active_track_tile = sort[0][:, :, 0]
        self.active_track_tile = sort[1][:, :, 0]

        # check if any tire is off track

        self.is_on_track = ~torch.any(~torch.any(self.wheels_on_track_segments, dim=3), dim=2)
        self.time_off_track += 1.0 * ~self.is_on_track
        self.time_off_track *= 1.0 * ~self.is_on_track  # reset if on track again

        # # update drag_reduction
        # self.drag_reduction_points[
        #     :, self.drag_reduction_write_idx : self.drag_reduction_write_idx + self.num_agents, :
        # ] = self.states[:, :, 0:2]
        # self.drag_reduction_write_idx = (
        #     self.drag_reduction_write_idx + self.num_agents
        # ) % self.drag_reduction_points.shape[1]
        # leading_pts = self.states[:, :, 0:2] + 5 * torch.cat(
        #     (
        #         torch.cos(self.states[:, :, 2].view(self.num_envs, self.num_agents, 1)),
        #         torch.sin(self.states[:, :, 2].view(self.num_envs, self.num_agents, 1)),
        #     ),
        #     dim=2,
        # )

        # dists = leading_pts.view(self.num_envs, self.num_agents, 1, 2) - self.drag_reduction_points.view(
        #     self.num_envs, 1, -1, 2
        # )
        # dists = torch.min(torch.norm(dists, dim=3), dim=2)[0]
        # self.drag_reduced[:] = False
        # self.drag_reduced = dists <= 3.0

        self.info["agent_active"] = self.active_agents

        # compute closest point on centerline
        self.tile_points = self.active_centerlines[:, self.active_track_tile, :]
        self.tile_points = self.tile_points[self.all_envs, self.all_envs, ...]
        self.tile_car_vec = self.states[:, :, 0:2] - self.tile_points
        angs = self.active_alphas[:, self.active_track_tile]
        angs = angs[self.all_envs, self.all_envs, ...]
        self.trackdir = torch.stack((torch.cos(angs), torch.sin(angs)), dim=2)
        self.trackperp = torch.stack((-torch.sin(angs), torch.cos(angs)), dim=2)

        increment_idx = torch.where(self.active_track_tile - self.old_active_track_tile < -10)
        decrement_idx = torch.where(self.active_track_tile - self.old_active_track_tile > 10)
        self.lap_counter[increment_idx] += 1
        self.lap_counter[decrement_idx] -= 1

        self.sub_tile_progress = torch.einsum("eac, eac-> ea", self.trackdir, self.tile_car_vec)
        self.track_progress_no_laps = self.active_track_tile * self.tile_len + self.sub_tile_progress
        self.track_progress = self.track_progress_no_laps + self.lap_counter * self.track_lengths[
            self.active_track_ids
        ].view(-1, 1)
        dist_sort, self.ranks = torch.sort(self.track_progress*self.active_agents, dim=1, descending=True)
        self.ranks = torch.sort(self.ranks, dim=1)[1]
        
        if self.track_stats:
            #set to 1 when not starting from rank 1
            idx_uninit = torch.where(self.stats_start_from_behind_current == -1)[0]
            if len(idx_uninit):
                self.stats_start_from_behind_current[idx_uninit] = 1.0*(self.ranks[idx_uninit, 0]>0)
                self.stats_last_ranks[idx_uninit] = 1.0*self.ranks[idx_uninit]
                self.stats_overtakes_current[idx_uninit] = 0

        self.contouring_err = torch.einsum("eac, eac-> ea", self.trackperp, self.tile_car_vec)
        
        # get env_ids to reset
        self.time_out_buf *= False
        self.check_termination()

        env_ids = torch.where(self.reset_buf)[0]
        # env_ids = torch.where(self.reset_buf_hl)[0]  # NOTE: step LL env until HL update
        self.info = {}
        self.agent_left_track |= self.time_off_track[:, :] > self.offtrack_reset
        # ids_report = torch.where(self.rank_reporting_active * (~all_agents_on_track|self.reset_buf))[0]
        # self.rank_reporting_active *= all_agents_on_track
        # if len(ids_report):
        #    self.info['ranking'] = [torch.mean(1.0*self.ranks[ids_report, :], dim = 0), 1.0*len(ids_report)/(1.0*len(self.reset_buf))]

        if len(env_ids):
            if self.track_stats:
                self.reset_stats(env_ids)
            self.reset_envs(env_ids)

        self.active_obs_template[...] = 1.0
        idx_1, idx_2 = torch.where(~self.active_agents)
        # remove ado observations of deactivated envs for active ones
        for ag in range(self.num_agents):
            self.active_obs_template[idx_1, ag, self.ado_idx_lookup[idx_2, ag]] = 0

        self.vel = torch.norm(self.states[:,:,self.vn['S_DX']:self.vn['S_DY']+1], dim = -1)

        if self.track_stats:
            self.update_current_stats()
        self.compute_observations()
        self.compute_rewards()

        # render before resetting values
        # if not self.headless:
        self.render()

        self.old_active_track_tile = self.active_track_tile
        self.old_track_progress = self.track_progress
        self.last_actions = self.actions.clone()
        self.last_steer[:,:,0] = self.actions[..., self.vn['A_STEER']]
        self.last_last_steer[:] = self.last_steer[:]
        self.last_vel[:] = self.vel

    def render(
        self,
    ) -> None:
        # if log_video freq is set only redner in fixed intervals
        if self.log_video_freq >= 0:
            self.viewer.mark_env(self.trained_agent_slot)
            if (self._global_step % self.log_video_freq == 0) and (self._global_step > 0):
                self.viewer_events = self.viewer.render(
                    self.states[:, :, [0, 1, 2, 6]], self.slip, self.drag_reduced, self.wheel_locations_world, self.interpolated_centers, self.interpolated_bounds, 
                    self.targets_pos_world, self.targets_rew01_local, self.targets_rot_world, self.targets_dist_track, self.actions, self.time_off_track,
                    self._action_probs_hl, self._action_mean_ll, self._action_std_ll,
                )
        else:
            self.viewer_events = self.viewer.render(
                self.states[:, :, [0, 1, 2, 6]], self.slip, self.drag_reduced, self.wheel_locations_world, self.interpolated_centers, self.interpolated_bounds, 
                self.targets_pos_world, self.targets_rew01_local, self.targets_rot_world, self.targets_dist_track, self.actions, self.time_off_track,
                self._action_probs_hl, self._action_mean_ll, self._action_std_ll,
            )

    def simulate(self) -> None:
        # run physics update
        act = self.last_actions.clone()
        act[..., self.vn['A_STEER']] = self.last_last_steer[..., 0]

        (
            self.states,
            self.contact_wrenches,
            self.shove,
            self.wheels_on_track_segments,
            self.slip,
            self.wheel_locations_world,
        ) = step_cars(
            self.states,
            act,
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
            self.track_A,  # Ax<=b
            self.track_b,
            self.track_S,  # how to sum
            self.dyn_model,
        )

    def reset_lstm(self):
        self.dyn_model.integration_function.initialize_lstm_states(torch.zeros((self.num_envs * self.num_agents, 50, 6)).to(self.device))

    def resample_track(self, env_ids) -> None:
        success = self.episode_length_buf[env_ids]> 0.9*self.max_episode_length
        if self.use_track_curriculum:
            self.track_successes[self.active_track_ids[env_ids]] += success
        a = 1/self.track_successes
        probs = a/torch.sum(a)
        dist = torch.distributions.Categorical(probs.view(-1,))
        self.active_track_ids[env_ids] = dist.sample((len(env_ids),))
        # self.active_track_ids[env_ids] = torch.randint(
        #     0, self.num_tracks, (len(env_ids),), device=self.device, requires_grad=False, dtype=torch.long
        # )
        self.active_track_mask[env_ids, ...] = 0.0
        self.active_track_mask[env_ids, self.active_track_ids[env_ids]] = 1.0
        self.active_centerlines[env_ids, ...] = self.track_centerlines[self.active_track_ids[env_ids]]
        self.active_alphas[env_ids, ...] = self.track_alphas[self.active_track_ids[env_ids]]
        self.active_track_tile_counts[env_ids] = self.track_tile_counts[self.active_track_ids[env_ids]]

        # update viewer
        # if not self.headless:
        self.viewer.active_track_ids[env_ids] = self.active_track_ids[env_ids]
        # call refresh on track drawing only in render mode
        # if self.viewer.do_render:
        if self.viewer.env_idx_render in env_ids:
            self.viewer.draw_track_reset()

    def get_state(self) -> torch.Tensor:
        return self.states.squeeze()

    def get_observations(
        self,
    ) -> torch.Tensor:
        return self.obs_buf  # .squeeze()

    def get_privileged_observations(
        self,
    ) -> Union[None, torch.Tensor]:
        return self.privileged_obs

    # gym stuffcompute_observations
    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    def set_trained_agent_slot(self, slot):
        self.trained_agent_slot = slot

    def reset_stats(self, env_ids):
        self.stats_overtakes_last_race[env_ids] = self.stats_overtakes_current[env_ids]
        self.stats_num_collisions_last_race[env_ids] = self.stats_num_collisions_current[env_ids]
        self.stats_ego_left_track_last_race[env_ids] = self.stats_ego_left_track_current[env_ids]
        self.stats_ado_left_track_last_race[env_ids] = self.stats_ado_left_track_current[env_ids]
        self.stats_win_from_behind_last_race[env_ids] = (self.ranks[env_ids, 0] == 0)*self.stats_start_from_behind_current[env_ids]  -1* (self.ranks[env_ids, 0] != 0)*self.stats_start_from_behind_current[env_ids]
        self.stats_lead_time_last_race[env_ids] = self.stats_lead_time_current[env_ids]
        self.stats_percentage_sa_laptime_last_race[env_ids] = -1

        self.stats_overtakes_current[env_ids] = 0
        self.stats_num_collisions_current[env_ids] = 0
        self.stats_ego_left_track_current[env_ids] = 0 
        self.stats_ado_left_track_current[env_ids] = 0 
        self.stats_start_from_behind_current[env_ids] = -1
        self.stats_lead_time_current[env_ids] = 0

    def update_current_stats(self):
        self.stats_overtakes_current[:] += 1.0*((self.ranks[:,0] - self.stats_last_ranks[:,0]) < 0)
        self.stats_num_collisions_current[:] += 1.0 * self.is_collision[:,0]
        self.stats_ego_left_track_current[:] += 1.0 * ~self.is_on_track[:,0]
        self.stats_ado_left_track_current[:] += 1.0 * torch.any(~self.is_on_track[:,1:] , dim =1).view(-1,)
        self.stats_lead_time_current[:] += self.dt*(self.ranks[:, 0])
        
        self.stats_last_ranks[:] = self.ranks[:] 

### jit functions
#@torch.jit.script
def compute_rewards_jit(
    rew_buf: torch.Tensor,
    track_progress: torch.Tensor,
    old_track_progress: torch.Tensor,
    progress_other: torch.Tensor,
    is_on_track: torch.Tensor,
    reward_scales: Dict[str, float],
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    vn: Dict[str, int],
    states: torch.Tensor,
    vel: torch.Tensor,
    last_vel : torch.Tensor,
    ranks: Union[torch.Tensor,None],
    is_collision: torch.Tensor,
    reward_terms: Dict[str, torch.Tensor],
    num_agents: int,
    active_agents: torch.Tensor,
    dt: float,
    targets_dist_local: torch.Tensor,
    targets_std_local: torch.Tensor,
    ll_ep_done: torch.Tensor,
    train_ll: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    rew_progress = torch.clip(track_progress - old_track_progress, min=-10, max=10) * reward_scales["progress"] * (1.0 - train_ll)
    rew_on_track = reward_scales["offtrack"] * ~is_on_track * train_ll
    act_diff = torch.square(actions - last_actions)
    rew_acc = torch.square(vel-last_vel.view(-1,num_agents))*reward_scales['acceleration'] * train_ll
    act_diff[..., 0] *= 5.0  # 25.0
    act_diff[..., 1] *= 10.0
    rew_actionrate = -torch.sum(act_diff, dim=2) * reward_scales["actionrate"] * train_ll
    #rew_energy = -torch.sum(torch.square(states[:, :, vn["S_W0"] : vn["S_W3"] + 1]), dim=2) * reward_scales["energy"]
    num_active_agents = torch.sum(1.0 * active_agents, dim=1)

    # Goal distance reward --> FIXME: does not rotate distribution
    exponent = -0.5 * (targets_dist_local / targets_std_local)**2
    pdf_dist = 1.0/(np.sqrt(2.0 * np.pi) * targets_std_local) * torch.exp(exponent)
    rew_dist = ll_ep_done * pdf_dist * reward_scales['goal']
    # rew_dist = torch.mean(rew_dist, dim=-1)
    # Only reward y proximity if actually close to target
    rew_dist = rew_dist[..., 0] + 1.0 / (5e1 * targets_dist_local[..., 0].abs() + 1e0) * rew_dist[..., 1]

    if ranks is not None:
        #compute relative progress
        rew_rel_prog = -torch.clip(progress_other, -1, 1).sum(dim=-1)* reward_scales["rank"]
        
        #rew_rank = 1.0 / num_agents * (num_active_agents.view(-1, 1) / 2.0 - ranks) * reward_scales["rank"]

        rew_collision = is_collision * reward_scales["collision"]

        rew_buf[..., 0] = torch.clip(
            rew_progress + rew_on_track + rew_actionrate + 0*rew_rel_prog + rew_acc + 0.15*dt, min=-5*dt, max=None
        )+ rew_collision + rew_dist
    else:
        #needs to be a tensor
        rew_rank = 0*rew_on_track
        rew_collision = 0*rew_on_track
        rew_buf[..., 0] = torch.clip(
            rew_progress + rew_on_track + rew_actionrate + rew_acc + 0.1*dt , min=-5*dt, max=None
        ) + rew_dist

    reward_terms["progress"] += rew_progress
    reward_terms["offtrack"] += rew_on_track
    reward_terms["actionrate"] += rew_actionrate
    #reward_terms["energy"] += rew_energy
    reward_terms["acceleration"] += rew_acc

    reward_terms["goal"] += rew_dist

    if ranks is not None:
        reward_terms["rank"] += rew_rel_prog
        reward_terms["collision"] += rew_collision
    return rew_buf, reward_terms
