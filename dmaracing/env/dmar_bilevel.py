#from cgitb import enable
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
from dmaracing.env.car_dynamics_utils import (get_varnames, set_dependent_params, 
allocate_car_dynamics_tensors, SwitchedBicycleKinodynamicModel)
#from dynamics_lib import DynamicsEncoder
from dmaracing.env.car_dynamics import step_cars
from dmaracing.controllers.purepursuit import PPController

import time
from dmaracing.utils.helpers import RunningStats


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

        #extract hardware-level controller parameters
        self.k_steer = self.modelParameters["k_steer"]
        self.k_vel = self.modelParameters["k_vel"]
        self.d_steer = self.modelParameters["d_steer"]
        self.d_vel = self.modelParameters["d_vel"]

        self.log_behavior_freq = cfg["sim"]["logBehaviorEvery"]
        self.reset_timeout_only = cfg["sim"]["reset_timeout_only"]

        # teams
        self.num_teams = cfg['teams']['numteams']
        self.team_size = cfg['teams']['teamsize']
        self.teams = [torch.tensor([idx for idx in range(self.team_size*start, self.team_size*start+self.team_size)], dtype=torch.long) for start in range(self.num_teams)]

        self.set_reset_agent_ids(reset_id_num=self.team_size)

        # use bootstrapping on vf
        self.use_timeouts = cfg["learn"]["use_timeouts"]
        self.track_stats = cfg["trackRaceStatistics"]
        self.log_behavior_stats = cfg["logBehaviorStatistics"]

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

        self.use_hierarchical_policy = cfg["policy"]["use_hierarchical_policy"]

        self.tile_jump_lim = self.simParameters["tile_jump_lim"]
        self.prog_jump_lim = self.tile_jump_lim * cfg["track"]["track_poly_spacing"]

        # Import TRIKART dynamics model and weights
        self.dyn_model = SwitchedBicycleKinodynamicModel( sim_cfg=self.cfg['sim'],
                                                          model_cfg=self.cfg['model'],
                                                          vn = self.vn,
                                                          device=self.device
                                                         )
        
        if self.test_mode:
            self.dyn_model.set_test_mode()
        
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
        self.log_video_active = True
        self.viewer = Viewer(
            cfg,
            self.track_centerlines,
            self.track_poly_verts,
            self.track_border_poly_verts,
            self.track_border_poly_cols,
            self.track_tile_counts,
            self.active_track_ids,
            teamsize=self.team_size,
            headless=self.headless,
        )
        self.info = {}

        self._values_ll = None

        # allocate env tensors
        torch_zeros = lambda shape: torch.zeros(shape, device=self.device, dtype=torch.float, requires_grad=False)
        self.states = torch_zeros((self.num_envs, self.num_agents, self.num_internal_states))
        self.contact_wrenches = torch_zeros((self.num_envs, self.num_agents, 3))
        self.rear_end_contact = torch_zeros((self.num_envs, self.num_agents, 1))

        self.shove = torch_zeros((self.num_envs, self.num_agents, 3))
        self.actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.hardware_commands = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.actions_means = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        self.last_actions = torch_zeros((self.num_envs, self.num_agents, self.num_actions))
        #self.last_steer = torch_zeros((self.num_envs, self.num_agents, 1))
        self.last_vel = torch_zeros((self.num_envs, self.num_agents,))
        #self.last_last_steer = torch_zeros((self.num_envs, self.num_agents, 1))

        self.ll_steps_left = torch_zeros((self.num_envs, self.num_agents, 1)) + 1.0
        
        self.obs_buf = torch_zeros((self.num_envs, self.num_agents, self.num_obs))
        self.rew_buf = torch_zeros((self.num_envs, self.num_agents, 2))
        self.time_out_buf = torch_zeros((self.num_envs, self.team_size)) > 1
        self.reset_buf = torch_zeros((self.num_envs, self.team_size)) > 1
        self.reset_buf_tmp = torch_zeros((self.num_envs, self.team_size)) > 1
        self.update_data = 1.0 + torch_zeros((self.num_envs, self.team_size))
        self.agent_left_track = torch_zeros((self.num_envs, self.num_agents)) > 1
        self.episode_length_buf = torch_zeros((self.num_envs, 1))
        self.time_off_track = torch_zeros((self.num_envs, self.num_agents))
        self.dt = cfg["sim"]["dt"]

        self.time_left_frac = torch_zeros((self.num_envs, self.num_agents, 1))
        self.wheel_off_frac = torch_zeros((self.num_envs, self.num_agents, 1))

        self.reset_cause = torch_zeros((self.num_envs, 1))

        self.dt_hl = cfg["learn"]["episode_length_ll"]

        self.reward_scales = {}
        self.reward_scales["offtrack"] = cfg["learn"]["offtrackRewardScale"]  # * self.dt
        self.reward_scales["progress"] = cfg["learn"]["progressRewardScale"]  # * self.dt
        self.reward_scales["actionrate"] = cfg["learn"]["actionRateRewardScale"]  # * self.dt
        self.reward_scales["rank"] = cfg["learn"]["rankRewardScale"]  # * self.dt
        self.reward_scales["collision"] = cfg["learn"]["collisionRewardScale"]  # * self.dt
        self.reward_scales["rearendcollision"] = cfg["learn"]["rearendcollisionRewardScale"]  # * self.dt
        self.reward_scales["acceleration"] = cfg["learn"]["accRewardScale"]  # * self.dt
        self.reward_scales["velocity"] = cfg["learn"]["velRewardScale"]  # * self.dt
        self.reward_scales["goal"] = cfg["learn"]["goalRewardScale"]  # * self.dt
        self.reward_scales["total"] = cfg["learn"]["totalRewardScale"]

        self.reward_stats = RunningStats()

        # # Could consider schedule on offtrack-like penalties to not hinder early learning? 
        # self.penalty_track_scale_ini = cfg["learn"]["trackPenaltyScaleIni"]
        # self.penalty_track_scale_itr = cfg["learn"]["trackPenaltyScaleItr"]
        # self.penalty_track_scale = self.penalty_track_scale_ini

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
        
        #obs noise scaling
        self.vel_noise_scale = cfg['learn']['vel_noise_scale'] 
        self.track_boundary_noise_scale = cfg['learn']['track_boundary_noise_scale']
        self.ado_pos_noise_scale = cfg['learn']['ado_pos_noise_scale']
        self.ado_rot_noise_scale = cfg['learn']['ado_rot_noise_scale']
        self.ado_vel_noise_scale = cfg['learn']['ado_vel_noise_scale']
        self.ado_angvel_noise_scale = cfg['learn']['ado_angvel_noise_scale']

        try:
            self.agent_dropout_prob_val_ini = cfg["learn"]["agent_dropout_prob_val_ini"]
            self.agent_dropout_prob_val_end = cfg["learn"]["agent_dropout_prob_val_end"]
            self.agent_dropout_prob_itr_ini = cfg["learn"]["agent_dropout_prob_itr_ini"]
            self.agent_dropout_prob_itr_end = cfg["learn"]["agent_dropout_prob_itr_end"]
            self.agent_dropout_prob = self.agent_dropout_prob_val_ini
        except:
            self.agent_dropout_prob = cfg["learn"]["agent_dropout_prob"]
        self.agent_dropout_egos = 1*cfg["learn"]["agent_dropout_egos"]

        self.use_ppc_vec = torch_zeros((self.num_envs, self.num_agents, 1))
        self.ppc_prob_val_ini = cfg["learn"]["ppc_prob_val_ini"]
        self.ppc_prob_val_end = cfg["learn"]["ppc_prob_val_end"]
        self.ppc_prob_itr_ini = cfg["learn"]["ppc_prob_itr_ini"]
        self.ppc_prob_itr_end = cfg["learn"]["ppc_prob_itr_end"]
        self.ppc_prob = self.ppc_prob_val_ini

        self.vm_noise_scale_ado_val_ini = cfg["model"]["vm_noise_scale_ado_val_ini"]
        self.vm_noise_scale_ado_val_end = cfg["model"]["vm_noise_scale_ado_val_end"]
        self.vm_noise_scale_ado_itr_ini = cfg["model"]["vm_noise_scale_ado_itr_ini"]
        self.vm_noise_scale_ado_itr_end = cfg["model"]["vm_noise_scale_ado_itr_end"]

        self.action_min_hl = cfg["learn"]["actionsminhl"]
        self.action_max_hl = cfg["learn"]["actionsmaxhl"]
        self.action_ini_hl = cfg["learn"]["actionsinihl"]
        self.action_all_hl = cfg["learn"]["actionsallhl"]
        self.use_hl_uncertainty = cfg["learn"]["use_hl_uncertainty"]

        if not self.use_hl_uncertainty:
            self.num_actions_hl = 2

        self.obs_noise_lvl = cfg["learn"]["obs_noise_lvl"]
        self.use_track_curriculum = cfg["learn"]["use_track_curriculum"]

        self.steer_ado_ppc = cfg['learn']['steer_ado_with_PPC']
        self.iters_ado_ppc = self.cfg['learn']['iters_ado_with_PPC'] if (self.cfg['learn']['steer_ado_with_PPC'] and self.num_agents>1) else 0
        
        if self.iters_ado_ppc != 0:
            print('[DMAR] Overriding ado actions with PPC')
        self.ppc = PPController(self,
                    lookahead_dist=1.5,
                    maxvel=-1.0,  # 3.0,  # 2.5,
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
        self.old_track_progress_no_laps = torch.zeros_like(self.time_off_track)
        self.reward_terms = {
            "progress": torch_zeros((self.num_envs, self.num_agents)),
            "offtrack": torch_zeros((self.num_envs, self.num_agents)),
            "actionrate": torch_zeros((self.num_envs, self.num_agents)),
            #"energy": torch_zeros((self.num_envs, self.num_agents)),
            "acceleration": torch_zeros((self.num_envs, self.num_agents)),
            "goal": torch_zeros((self.num_envs, self.num_agents)),
            "velocity": torch_zeros((self.num_envs, self.num_agents)),
        }
        if self.num_agents>1:
            self.reward_terms["rank"] = torch_zeros((self.num_envs, self.num_agents))
            self.reward_terms["collision"] = torch_zeros((self.num_envs, self.num_agents))
            self.reward_terms["rearendcollision"] = torch_zeros((self.num_envs, self.num_agents))

        self.step_reward_terms = {
            "progress": torch_zeros((self.num_envs, self.num_agents)),
            "offtrack": torch_zeros((self.num_envs, self.num_agents)),
            "actionrate": torch_zeros((self.num_envs, self.num_agents)),
            "acceleration": torch_zeros((self.num_envs, self.num_agents)),
            "goal": torch_zeros((self.num_envs, self.num_agents)),
            "velocity": torch_zeros((self.num_envs, self.num_agents)),
        }
        if self.num_agents>1:
            self.step_reward_terms["rank"] = torch_zeros((self.num_envs, self.num_agents))
            self.step_reward_terms["collision"] = torch_zeros((self.num_envs, self.num_agents))
            self.step_reward_terms["rearendcollision"] = torch_zeros((self.num_envs, self.num_agents))


        self.is_collision = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.float
        )
        
        self.is_rearend_collision = torch.zeros(
            (self.num_envs, self.num_agents, 1), requires_grad=False, device=self.device, dtype=torch.float
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
        self.teamranks = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int)
        self.prev_ranks = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int
        )
        self.prev_teamranks = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int)
        self.initial_team_rank = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int)
        self.initial_rank = torch.zeros(
            (self.num_envs, self.num_agents), requires_grad=False, device=self.device, dtype=torch.int)

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
        self.targets_off_ahead = torch.zeros(
            (self.num_envs, self.num_agents, 1), device=self.device, requires_grad=False, dtype=torch.float
        )

        self.all_targets_pos_world_env0 = []
        self.all_egoagent_pos_world_env0 = []
        self.ego_pos_world = torch.zeros(
            (2,), device=self.device, requires_grad=False, dtype=torch.float
        )
        
        self._action_probs_hl, self._action_mean_ll, self._action_std_ll, self.targets_rew01_local  = None, None, None, None
        self._svo_probs_hl = None

        self._global_step = 0
        self.log_episode = False
        self.log_episode_next = False
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

        if self.log_behavior_stats:
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
            self.stats_lap_time_current = torch.zeros((self.num_envs), device = self.device)
            self.stats_lead_time_last_race = torch.zeros((self.num_envs), device = self.device)
            self.stats_last_ranks = torch.zeros((self.num_envs, self.num_agents), device = self.device)

            self.batch_stats_overtakes = []
            self.batch_stats_num_collisions = []
            self.batch_stats_ego_left_track = []
            self.batch_stats_lead_time = []
            self.batch_stats_rank_team = []
            self.batch_stats_rank_lower = []

        if self.track_stats:
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
        print(self.dyn_model.randomized_params)
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

    def set_dropout_prob(self, iter):
        progress = min(max(iter-self.agent_dropout_prob_itr_ini, 0.0)/(self.agent_dropout_prob_itr_end - self.agent_dropout_prob_itr_ini), 1.0)
        self.agent_dropout_prob = self.agent_dropout_prob_val_ini + progress * (self.agent_dropout_prob_val_end - self.agent_dropout_prob_val_ini)

    def set_ppc_prob(self, iter):
        progress = min(max(iter-self.ppc_prob_itr_ini, 0.0)/(self.ppc_prob_itr_end - self.ppc_prob_itr_ini), 1.0)
        self.ppc_prob = self.ppc_prob_val_ini + progress * (self.ppc_prob_val_end - self.ppc_prob_val_ini)

    def set_vm_noise_scale_ado(self, iter):
        progress = min(max(iter-self.vm_noise_scale_ado_itr_ini, 0.0)/(self.vm_noise_scale_ado_itr_end - self.vm_noise_scale_ado_itr_ini), 1.0)
        vm_noise_scale_ado = self.vm_noise_scale_ado_val_ini + progress * (self.vm_noise_scale_ado_val_end - self.vm_noise_scale_ado_val_ini)
        self.dyn_model.update_vm_noise_scale_ado(vm_noise_scale_ado)

    def set_video_log_ep(self, iter):
        self.log_episode_next = self.log_episode_next or ((iter % self.log_video_freq)==0 and iter!=0)
        if self.log_episode_next:
            self._global_step = iter

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
        self._action_probs_hl = probs.clone()

    def set_hl_svo_probs(self, probs):
        self._svo_probs_hl = probs.clone()

    def set_ll_action_stats(self, mean, std):
        self._action_mean_ll = mean.clone()
        self._action_std_ll = std.clone()

        self._action_mean_ll[..., 0] = self.action_scales[0] * self._action_mean_ll[..., 0] + self.default_actions[0]
        self._action_mean_ll[..., 1] = self.action_scales[1] * self._action_mean_ll[..., 1] + self.default_actions[1]

        self._action_std_ll[..., 0] = self.action_scales[0] * self._action_std_ll[..., 0]
        self._action_std_ll[..., 1] = self.action_scales[1] * self._action_std_ll[..., 1]

    def set_ll_values_preds(self, values):
        self._values_ll = values[:, 0].clone()

    def project_into_track_frame(self, actions_hl):
        actions_hl = actions_hl.clone().to(self.device)
        # actions_hl *= torch.tensor(self.action_scales_hl[:2], device=self.device)

        target_offset_on_centerline_track = actions_hl[..., 0:2]
        if self.use_hl_uncertainty:
            target_std_local = actions_hl[..., 2:4]  # .unsqueeze(dim=1)  # NOTE: changed for multi-agent
        else:
            target_std_local = 0.0*target_offset_on_centerline_track
            target_std_local[..., 0] += 0.3
            target_std_local[..., 1] += 0.1
        
        self.ll_steps_left *= 0.0
        self.ll_steps_left += (self.dt_hl - torch.remainder(self.episode_length_buf.unsqueeze(dim=1), self.dt_hl)) / self.dt_hl

        center = self.active_centerlines[self.all_envs.view(-1,1), self.active_track_tile]
        angle = self.active_alphas[self.all_envs.view(-1,1), self.active_track_tile]

        #use subtile progress?
        car_offset_from_center_world = self.states[:, :, :2] - center
        car_offset_from_center_local = self.coord_trafo(car_offset_from_center_world, angle.unsqueeze(-1))  # relative pos in local ~= track
        target_offset_from_car_track = car_offset_from_center_local + target_offset_on_centerline_track
        target_offset_from_car_tiles = torch.div(target_offset_from_car_track[..., 0], self.tile_len, rounding_mode="floor")
        target_offset_from_center_track = torch.zeros_like(target_offset_from_car_track)
        target_offset_from_center_track[..., 0] = torch.remainder(target_offset_from_car_track[..., 0], self.tile_len)
        target_offset_from_center_track[..., 1] = target_offset_on_centerline_track[..., 1]

        target_tile_idx = torch.remainder(self.active_track_tile + target_offset_from_car_tiles, self.active_track_tile_counts.view(-1, 1)).type(torch.long)
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
        self.targets_off_ahead = update_target * target_offset_on_centerline_track[..., 0].unsqueeze(dim=-1) + (1.0 - update_target) * self.targets_off_ahead

        # Compute distance to target in track coordinates
        target_tiles_from_car = self.targets_tile_idx.squeeze(-1) - self.active_track_tile
        target_tiles_from_car += self.active_track_tile_counts.view(-1, 1) * (target_tiles_from_car < -self.active_track_tile_counts.view(-1, 1)/2.0)
        target_tiles_from_car -= self.active_track_tile_counts.view(-1, 1) * (target_tiles_from_car > +self.active_track_tile_counts.view(-1, 1)/2.0)
        target_dist_track_x = self.targets_tile_pos[..., 0] - car_offset_from_center_local[..., 0] + self.tile_len * target_tiles_from_car
        target_dist_track_y = self.targets_tile_pos[..., 1] - car_offset_from_center_local[..., 1]
        target_dist_track = torch.stack([target_dist_track_x, target_dist_track_y], dim=-1)

        # self.target_tiles_from_car  # NOTE: debug

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
        
        theta = self.states[:, :, 2]
        vels = self.states[:, :, 3:6].clone()

        tile_idx_unwrapped = self.active_track_tile.unsqueeze(2) + (
            4 * torch.arange(self.horizon, device=self.device, dtype=torch.long)  # 4
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

        # self.R_world_to_track[:, :, 0, 0] = torch.cos(angles_at_centers[..., 0])
        # self.R_world_to_track[:, :, 0, 1] = torch.sin(angles_at_centers[..., 0])
        # self.R_world_to_track[:, :, 1, 0] = -torch.sin(angles_at_centers[..., 0])
        # self.R_world_to_track[:, :, 1, 1] = torch.cos(angles_at_centers[..., 0])


        linvel_world = torch.einsum("eaij, eaj -> eai", torch.transpose(self.R, 2, 3), self.states[:, :, 3:5])
        ### NOTE: use for velocity observation in track frame
        # linvel_track = torch.einsum("eaij, eaj -> eai", self.R_world_to_track, linvel_world)

        # self.lookahead_body = torch.einsum("eaij, eatj->eati", self.R, self.lookahead)
        # lookahead_scaled = self.lookahead_scaler * self.lookahead_body

        self.lookahead_rbound_body = torch.einsum('eaij, eatj->eati', self.R, self.lookahead_rbound)
        lookahead_rbound_scaled = self.lookahead_rbound_body #* self.lookahead_scaler  # HACK: testing 04/12/23
        self.lookahead_lbound_body = torch.einsum('eaij, eatj->eati', self.R, self.lookahead_lbound)
        lookahead_lbound_scaled = self.lookahead_lbound_body #* self.lookahead_scaler  # HACK: testing 04/12/23


        self.vels_body = vels
        # self.vels_body[..., :-1] = torch.einsum("eaij, eaj -> eai", self.R, vels[..., :-1])  # FIXME: remove

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

                # selfvel = self.states[:, agent, self.vn["S_DX"] : self.vn["S_DY"] + 1].view(-1, 1, 2)  # FIXME: remove
                ### NOTE: use for velocity observation in body frame
                selfvel = linvel_world[:, agent, :].view(-1, 1, 2)  # self.states[:, agent, self.vn["S_DX"] : self.vn["S_DY"] + 1].view(-1, 1, 2)
                ### NOTE: use for velocity observation in track frame
                # selfvel = linvel_track[:, agent, :].view(-1, 1, 2)

                selfangvel = self.states[:, agent, self.vn["S_DTHETA"]].view(-1, 1, 1)

                all_team = self.teams.copy()
                ego_team_idx = sum([idx * (agent in team) for idx, team in enumerate(self.teams)])
                ego_team = all_team.pop(ego_team_idx)
                opponents = torch.concat((ego_team, *all_team), dim=-1)
                opponents = opponents[opponents!=agent]

                otherpos = self.states[:, opponents, 0:2]
                otherpositions.append((otherpos - selfpos).view(-1, (self.num_agents - 1), 2))

                selftrackprogress = self.track_progress_no_laps[:, agent].view(-1, 1, 1)
                oth_prog = self.track_progress_no_laps[:, opponents].view(-1, 1, self.num_agents-1) - selftrackprogress
                
                oth_progress_w_laps = torch.clip(
                        self.track_progress[:, opponents].view(-1, 1, self.num_agents - 1)
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
                other_contouringerr.append(self.contouring_err[:, opponents].view(-1, 1, self.num_agents-1) - selfcontouringerr.view(-1,1,1))
                otherrotations.append(self.states[:, opponents, 2].view(-1, 1, self.num_agents-1) - selfrot)

                # othervel = self.states[:, opponents, self.vn['S_DX']: self.vn['S_DY']+1]  # FIXME: remove
                ### NOTE: use for velocity observation in body frame
                othervel = linvel_world[:, opponents, :]  # self.states[:, opponents, self.vn['S_DX']: self.vn['S_DY']+1]
                ### NOTE: use for velocity observation in track frame
                # othervel = linvel_track[:, opponents, :]

                othervelocities.append((othervel - selfvel).view(-1, (self.num_agents - 1), 2))
                
                otherangvel = self.states[:, opponents, self.vn['S_DTHETA']].view(-1, 1, self.num_agents-1)
                otherangularvelocities.append(otherangvel - selfangvel)

            pos_other = torch.cat(tuple([pos.view(-1, 1, (self.num_agents - 1), 2) for pos in otherpositions]), dim=1)
            pos_other = torch.einsum("eaij, eaoj->eaoi", self.R, pos_other)
            norm_pos_other = torch.norm(pos_other, dim=3).view(self.num_envs, self.num_agents, -1)
            is_other_close = norm_pos_other < self.cfg["learn"]["distance_obs_cutoff"]

            # NOTE: rotate relative velocities from world to body frame
            vel_other = torch.cat(
                tuple([vel.view(-1, 1, (self.num_agents - 1), 2) for vel in othervelocities]), dim=1
            ) * self.active_obs_template.view(self.num_envs, self.num_agents, self.num_agents - 1, 1)

            # vel_other = torch.einsum("eaij, eaoj->eaoi", self.R, vel_other) * is_other_close.view(   # FIXME: remove
            ### NOTE: use for velocity observation in body frame
            vel_other = torch.einsum("eaij, eaoj->eaoi", self.R, vel_other) * is_other_close.view(
                self.num_envs, self.num_agents, self.num_agents - 1, 1
            )
            ### NOTE: use for velocity observation in track frame
            # vel_other = vel_other * is_other_close.view(
            #     self.num_envs, self.num_agents, self.num_agents - 1, 1
            # )

            rot_other = (
                torch.cat(tuple([rot for rot in otherrotations]), dim=1) * self.active_obs_template * is_other_close
            )
            angvel_other = (
                torch.cat(tuple([angvel for angvel in otherangularvelocities]), dim=1)
                * self.active_obs_template
                * is_other_close
            )

            self.progress_other = (
                torch.clip(torch.cat(tuple([prog for prog in other_progress]), dim=1), min=-self.cfg["learn"]["distance_obs_cutoff"], max=self.cfg["learn"]["distance_obs_cutoff"])
                * self.active_obs_template - self.cfg["learn"]["distance_obs_cutoff"]*(1-self.active_obs_template)  # 2.5
            )

            self.contouring_err_other = (
                torch.clip(torch.cat(tuple([err for err in other_contouringerr]), dim=1), min=-1, max=1)
                * self.active_obs_template
                * is_other_close
            )
            last_raw_actions = self.last_actions.clone()
            last_raw_actions[:,:, 0] = (last_raw_actions[:,:, 0] - self.default_actions[0])/self.action_scales[0]  
            last_raw_actions[:,:, 1] = (last_raw_actions[:,:, 1] - self.default_actions[1])/self.action_scales[1]  

            self.time_left_frac[:, :, 0] = 1.0 - torch.clamp(1.0 - self.episode_length_buf / (0.8*self.max_episode_length), min=0.0)
            self.wheel_off_frac[:, :, 0] = ((~self.is_on_track_per_wheel) * 1.0).sum(dim=-1) / 4.0
            progress_jump = (self.track_progress - self.old_track_progress) * (self.episode_length_buf > 1)
            progress_jump = torch.clamp(progress_jump, min=-5.0, max=+5.0)

            # print("--------------")
            # print(self.episode_length_buf[0])
            # print(self.time_left_frac[0])
            # print(self.wheel_off_frac[0])
            # print(progress_jump[0])
            
            #maxvel obs self.dyn_model.dynamics_integrator.dyn_model.max_vel_vec
            self.obs_buf = torch.cat(
                (
                    self.vels_body,  #  * 0.1,  # HACK: testing 04/21/23
                    lookahead_rbound_scaled[:,:,:,0], 
                    lookahead_rbound_scaled[:,:,:,1], 
                    lookahead_lbound_scaled[:,:,:,0], 
                    lookahead_lbound_scaled[:,:,:,1],
                    # self.dyn_model.max_vel_vec,  # NOTE: removed if only used for robustness
                    2.0 * self.time_left_frac - 1.0,  # NOTE: removed 05/04/23 evening
                    2.0 * self.wheel_off_frac - 1.0,  # NOTE: removed 05/04/23 evening
                    progress_jump.view(-1, self.num_agents, 1),
                    last_raw_actions,
                    self.ranks.view(-1, self.num_agents, 1) - 1.5,
                    self.teamranks.view(-1, self.num_agents, 1) - 1.0,
                    self.progress_other * 0.3,  # * 0.1,  # HACK: testing 04/24/23
                    self.contouring_err_other,  #  * 0.25,  # HACK: testing 04/24/23
                    torch.sin(rot_other),
                    torch.cos(rot_other),
                    vel_other[..., 0],  # * 0.1,  # HACK: testing 04/21/23
                    vel_other[..., 1],  # * 0.1,  # HACK: testing 04/21/23
                    angvel_other,  # * 0.1,  # HACK: testing 04/21/23
                ),
                dim=2,
            )
        else:
            last_raw_actions = self.last_actions.clone()
           
            last_raw_actions[:,:, 0] = (last_raw_actions[:,:, 0] - self.default_actions[0])/self.action_scales[0]  
            last_raw_actions[:,:, 1] = (last_raw_actions[:,:, 1] - self.default_actions[1])/self.action_scales[1]  
            self.obs_buf = torch.cat(
            (
                self.vels_body * 0.1,
                lookahead_rbound_scaled[:,:,:,0], 
                lookahead_rbound_scaled[:,:,:,1], 
                lookahead_lbound_scaled[:,:,:,0], 
                lookahead_lbound_scaled[:,:,:,1],
                self.dyn_model.max_vel_vec, 
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

            # noise_vec = torch.randn_like(self.obs_buf) * 0.1 * self.obs_noise_lvl
            noise_vec = self.generate_noise_vec(self.obs_buf.size())  # self.obs_noise_lvl * (2.0 * (torch.rand(self.obs_buf.size(), device=self.obs_buf.device) - 0.5))
            # noise_vec[..., -7*self.num_agents+5:-7*self.num_agents] = 0.0
            noise_vec[..., -7*(self.num_agents-1)-7:-7*(self.num_agents-1)] = 0.0  # -4 w/o time & off-track & jump ; 7 with all new obs
            #velocity
            noise_vec[..., 0:3] *= self.vel_noise_scale
            #trackboundaries
            noise_vec[..., 3: 3+self.horizon *4] *= self.track_boundary_noise_scale
            #ado_pos
            noise_vec[..., -7*self.num_agents: -5*self.num_agents] *= self.ado_pos_noise_scale
            #ado_rot
            noise_vec[..., -5*self.num_agents: -3*self.num_agents] *= self.ado_rot_noise_scale
            #ado_vel
            noise_vec[..., -3*self.num_agents: -1*self.num_agents] *= self.ado_vel_noise_scale
            #ado_angvel
            noise_vec[..., -1*self.num_agents] *= self.ado_angvel_noise_scale
            noise_vec[..., -21:] *= self.active_obs_template.repeat(1, 1, 7)  # HACK: hard-coded setting ado noise 0

            # FIXME: 10x noise added for ado relative velocities
            #noise_vec[..., -self.num_agents * 3:] *= 5
            self.obs_buf += noise_vec

    def generate_noise_vec(self, size):
        return self.obs_noise_lvl * (2.0 * (torch.rand(size, device=self.obs_buf.device) - 0.5))

    def compute_rewards(
        self,
    ) -> None:
        self.rew_buf, self.reward_terms, self.step_reward_terms = compute_rewards_jit(
            self.rew_buf,
            self.track_progress,
            self.old_track_progress,
            # self.progress_other_w_laps if self.num_agents>1 else None,
            self.is_on_track,
            self.reward_scales,
            self.actions,
            self.last_actions,
            self.vn,
            self.states,
            self.vel,
            self.last_vel,
            self.ranks if self.num_agents>1 else None,
            self.teamranks if self.num_agents>1 else None,
            self.prev_ranks if self.num_agents>1 else None,
            self.prev_teamranks if self.num_agents>1 else None,
            self.is_collision,
            self.is_rearend_collision,
            self.reward_terms,
            self.step_reward_terms,
            self.num_agents,
            self.active_agents,
            self.dt,
            self.dt_hl,
            self.targets_dist_track.clone(),
            self.targets_std_track.clone(),
            self.ll_ep_done,
            self.ll_steps_left,
            1.0 * self.train_ll,
            self.is_on_track_per_wheel,
            self.targets_off_ahead,
            self.time_left_frac,
            self.prog_jump_lim,
            self.initial_team_rank,
            self.initial_rank,
            1.0 * self.use_hierarchical_policy,
        )

    def check_progress_jump(self, progress, old_progress) -> torch.Tensor:
        return torch.abs(progress - old_progress) > self.prog_jump_lim  # 5.0  # 3.5

    def check_termination(self) -> None:
        
        ### Check offtrack
        # reset_offtrack = self.time_off_track[:, self.trained_agent_slot].view(-1, 1) > self.offtrack_reset
        reset_offtrack = self.time_off_track[:, :self.team_size] > self.offtrack_reset
        ### Check cutting
        # reset_progress = (self.check_progress_jump(self.track_progress, self.old_track_progress)[..., 0].view(-1, 1))*(self.episode_length_buf > 2)
        reset_progress = (self.check_progress_jump(self.track_progress, self.old_track_progress)[..., :self.team_size])*(self.episode_length_buf > 2)
        reset_progress = reset_progress & False  # self.train_ll
        ### Check timeout
        reset_timeout = self.episode_length_buf > self.max_episode_length - 1
        
        self.reset_cause = 1e0 * reset_offtrack + 1e1 * reset_progress + 1e2 * reset_timeout
        # self.reset_buf_tmp = reset_offtrack | reset_progress | reset_timeout  # | self.reset_buf_tmp
        if self.reset_timeout_only:
            self.reset_buf_tmp = reset_timeout | self.reset_buf_tmp
        else:
            self.reset_buf_tmp = reset_offtrack | reset_progress | reset_timeout | self.reset_buf_tmp

        self.time_out_buf = reset_timeout

        if not self.train_ll:  # FIXME: check this
            self.reset_buf = torch.logical_and(self.reset_buf_tmp, torch.remainder((self.total_step) * torch.ones_like(self.reset_buf), self.dt_hl)==0)
        else:
            self.reset_buf[:] = self.reset_buf_tmp

    def reset(self) -> Tuple[torch.Tensor, Union[None, torch.Tensor]]:
        print("Resetting")
        env_ids = torch.arange(self.num_envs, dtype=torch.long)
        self.reset_envs(env_ids)
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs

    def reset_envs(self, env_ids) -> None:
        self.dyn_model.reset(env_ids)
        if self.track_stats:
            last_track_ids = self.active_track_ids[env_ids].clone()
        self.resample_track(env_ids)
        max_agent_drop = self.num_agents - (self.team_size - self.agent_dropout_egos*(self.team_size-1))
        self.active_agents[env_ids, -max_agent_drop:] = (
            torch.rand((len(env_ids), max_agent_drop), device=self.device) > self.agent_dropout_prob
        )
        max_start_tile = self.active_track_tile_counts[env_ids].view(-1, 1) - 8 * self.num_agents  # Avoid init lap mismatch via grid across finish line
        tile_idx_env = (
            torch.rand((len(env_ids), 1), device=self.device) * max_start_tile  # (max_start_tile + 0*self.active_track_tile_counts[env_ids].view(-1, 1))
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
                tile_idx_env = tile_idx_env + 6 * torch.linspace(  # 6
                    0, self.num_agents - 1, self.num_agents, device=self.device
                ).unsqueeze(0).to(dtype=torch.long)
                tile_idx_env = torch.remainder(tile_idx_env, self.active_track_tile_counts[env_ids].view(-1, 1))
        positions = torch.ones((len(env_ids), self.num_agents), device=self.device).multinomial(
            self.num_agents, replacement=False
        )
        # positions = torch.tensor([[0, 1, 2, 3]], device=self.device)  # FIXME: remove after testing
        # positions[..., 0] = 3
        # positions[..., 0] = 4
        # positions[..., 0] = 1
        # positions[..., 0] = 2
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
            ] = 100000.0 + 100 * torch.rand((len(idx2_inactive), 2), device=self.device, requires_grad=False)

        dists = torch.norm(
            self.states[env_ids, :, 0:2].unsqueeze(2) - self.active_centerlines[env_ids].unsqueeze(1), dim=3
        )
        self.old_active_track_tile[env_ids, :] = torch.sort(dists, dim=2)[1][:, :, 0]
        self.active_track_tile[env_ids, :] = torch.sort(dists, dim=2)[1][:, :, 0]
        
        self.info["episode"] = {}

        self.info["episode"]["rewards"] = torch.mean(torch.stack(list(self.reward_terms.values()), dim=0).sum(dim=0)[env_ids, 0] / self.timeout_s).view(-1, 1)
        for key in self.reward_terms.keys():
            self.info["episode"]["reward_" + key] = (
                torch.mean(self.reward_terms[key][env_ids, 0]) / self.timeout_s
            ).view(-1, 1)
            self.reward_terms[key][env_ids] = 0.0
            self.step_reward_terms[key][env_ids] = 0.0

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
            mv = self.dyn_model.max_vel_vec[env_ids, 0]
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
            self.dyn_model.max_vel_vec[env_ids].clone()]
            
        #dynamics randomization
        self.dyn_model.update_noise_vec(env_ids, self.noise_level, self.team_size)

        # ### Update quantities on reset ###
        # get current tile positions
        dists = torch.norm(self.states[env_ids, :, 0:2].unsqueeze(2) - self.active_centerlines[env_ids, ...].unsqueeze(1), dim=3)
        sort = torch.sort(dists, dim=2)
        self.active_track_tile[env_ids] = sort[1][:, :, 0]

        # compute closest point on centerline
        self.tile_points = self.active_centerlines[:, self.active_track_tile, :][self.all_envs, self.all_envs]
        tile_car_vec = self.states[env_ids, :, 0:2] - self.tile_points[env_ids]
        angs = self.active_alphas[env_ids.view(-1, 1), self.active_track_tile[env_ids]]
        trackdir = torch.stack((torch.cos(angs), torch.sin(angs)), dim=2)
        # self.trackperp = torch.stack((-torch.sin(angs), torch.cos(angs)), dim=2)

        sub_tile_progress = torch.einsum("eac, eac-> ea", trackdir, tile_car_vec)
        self.track_progress_no_laps[env_ids] = self.active_track_tile[env_ids] * self.tile_len + sub_tile_progress
        self.track_progress[env_ids, ...] = self.track_progress_no_laps[env_ids, ...]
        dist_sort, ranks = torch.sort(self.track_progress[env_ids, ...]*self.active_agents[env_ids], dim=1, descending=True)
        self.ranks[env_ids, ...] = torch.sort(ranks, dim=1)[1].to(self.ranks.dtype)

        for team in self.teams:
            self.teamranks[:, team] = torch.min(self.ranks[:, team], dim = 1)[0].reshape(-1,1).type(torch.int)

        self.initial_team_rank[env_ids, :] = self.teamranks[env_ids, :].clone()
        self.initial_rank[env_ids, :] = self.ranks[env_ids, :].clone()

        self.is_collision[env_ids, ...] = False
        self.is_rearend_collision[env_ids, ...] = 0
        # #####################################
        num_opponents = (self.num_teams - 1) * self.team_size
        self.use_ppc_vec[env_ids, -num_opponents:] = 1.0 * (torch.rand(len(env_ids), num_opponents, 1, device=self.device) < self.ppc_prob)

        self.prev_ranks[env_ids, ...] = self.ranks[env_ids, ...]
        self.prev_teamranks[env_ids, ...] = self.teamranks[env_ids, ...]

        self.lap_counter[env_ids, :] = 0
        self.episode_length_buf[env_ids] = 0.0
        self.old_track_progress[env_ids] = self.track_progress[env_ids]
        # self.track_progress[env_ids] = 0.0
        # self.track_progress_no_laps[env_ids] = 0.0
        self.time_off_track[env_ids, :] = 0.0
        self.last_actions[env_ids, :, :] = 0.0
        #self.last_steer[env_ids, :, :] = 0.0
        #self.last_last_steer[env_ids, :, :] = 0.0
        self.last_vel[env_ids, ...] = 0.0
        self.agent_left_track[env_ids, :] = False

        # self.reset_buf[env_ids] = False
        self.reset_buf_tmp[env_ids] = False
        self.update_data[env_ids] = 1.0

        self.wheels_on_track_segments[env_ids, ...] = True  # HACK

        self.ll_ep_done[env_ids, ...] = 0.0  # FIXME: remove?

        if 0 in env_ids and self.log_video_freq >= 0:
            if len(self.viewer._frames):

                frames = np.stack(self.viewer._frames, axis=0)  # (T, W, H, C)
                frames = np.transpose(frames, (0, 3, 1, 2))  # (T, C, W, H)
                frames = np.expand_dims(frames, axis=0)  # (N, T, C, W, H)
                self._writer.add_video("Train/video", frames, global_step=self._global_step, fps=15)
                self.log_episode = False

            # if len(self.viewer._frames_val):

            #     frames_val = np.stack(self.viewer._frames_val, axis=0)  # (T, W, H, C)
            #     frames_val = np.transpose(frames_val, (0, 3, 1, 2))  # (T, C, W, H)
            #     frames_val = np.expand_dims(frames_val, axis=0)  # (N, T, C, W, H)
            #     self._writer.add_video("Train/values", frames_val, global_step=self._global_step, fps=15)
            #     self.log_episode = False

            self.viewer._frames = []
            self.viewer._frames_val = []

            if self.log_episode_next:
                self.log_episode = True
                self.log_episode_next = False

    def set_steer_ado_ppc(self, steer_ado_ppc):
        self.steer_ado_ppc = steer_ado_ppc

    def step(self, actions) -> Tuple[torch.Tensor, Union[None, torch.Tensor], torch.Tensor, Dict[str, float]]:
        actions = actions.clone().to(self.device)
        if actions.requires_grad:
            actions = actions.detach()

        # if self.steer_ado_ppc:
        #     actions[:,-((self.num_teams - 1) * self.team_size):,:] = self.ppc.step()[:,-((self.num_teams - 1) * self.team_size):,:]
        actions_ppc = self.ppc.step()
        if len(actions.shape) == 2:
            self.actions[:, 0, 0] = self.action_scales[0] * actions[..., 0] + self.default_actions[0]
            self.actions[:, 0, 1] = self.action_scales[1] * actions[..., 1] + self.default_actions[1] 
            # self.actions[:, 0, 2] = self.action_scales[2] * actions[..., 2] + self.default_actions[2]      
        else:
            self.actions[:, :, 0] = self.action_scales[0] * actions[..., 0] + self.default_actions[0]
            self.actions[:, :, 1] = self.action_scales[1] * actions[..., 1] + self.default_actions[1]
            # self.actions[:, :, 2] = self.action_scales[2] * actions[..., 2] + self.default_actions[2]
        
        heading_offset = self.actions[:,:,0]
        heading_setpoint = self.states[:,:,2] + heading_offset
        if self.cfg['model']['use_drag_reduction']:
            velocity_setpoint = self.actions[..., 1]*(1+self.drag_reduced*self.cfg['model']['drag_reduction_multiplier'])
        else:
            velocity_setpoint = self.actions[..., 1]
        last_velocity = self.last_vel
        heading_error_rate = -self.states[..., self.vn['S_DTHETA']]
        # reset collision detection
        self.is_collision = self.is_collision < 0
        self.is_rearend_collision = self.is_rearend_collision < 0
        for j in range(self.decimation):
            heading_error = heading_setpoint - self.states[:,:,2] 
            current_velocity = torch.norm(self.states[:,:,3:5], dim =-1)
            velocity_error = velocity_setpoint - current_velocity
            vel_error_rate = (last_velocity - current_velocity)/self.dt
            self.hardware_commands[:,:, self.vn['A_STEER']] =  torch.clip(self.k_steer * heading_error + self.d_steer*heading_error_rate, -0.35, 0.35)
            self.hardware_commands[:,:, self.vn['A_GAS']] =  self.k_vel * velocity_error + self.d_vel*vel_error_rate
            #ppc actions are interpreted as steering angle and accelerations
            self.hardware_commands[:] = (1.0 - self.use_ppc_vec) * self.hardware_commands + self.use_ppc_vec * actions_ppc
        
            self.simulate()

            heading_error_rate = -self.states[..., self.vn['S_DTHETA']]
            last_velocity = current_velocity 
            # need to check for collisions in inner loop otherwise get missed
            self.is_collision |= torch.norm(self.contact_wrenches, dim=2) > 0
            self.is_rearend_collision[:] = 1.0*((self.is_rearend_collision + self.rear_end_contact)>0)
            
        self.post_physics_step()

        if True:
            self.log_info_ado()

        # self.reward_stats.update(self.rew_buf[:, 0, 0])
        # self.rew_buf /= self.reward_stats.std()

        return self.obs_buf.clone(), self.privileged_obs, self.rew_buf.clone(), self.reset_buf.clone(), self.info

    def log_info_ado(self):
        self.info['observations'] = {}
        # [-21, -18, -15, -12, -9, -6, -3] --> [-20, -17, -14, -11, -8, -5, -2] --> [-19, -16, -13, -10, -7, -4, -1]
        for idx in range(self.num_agents-1):
            self.log_info_stats(self.obs_buf[..., -7 * (self.num_agents-1)+idx], 'observations', 'ado_prog' + str(idx+1))
            self.log_info_stats(self.obs_buf[..., -6 * (self.num_agents-1)+idx], 'observations', 'ado_cerr' + str(idx+1))
            self.log_info_stats(self.obs_buf[..., -5 * (self.num_agents-1)+idx], 'observations', 'ado_rots' + str(idx+1))
            self.log_info_stats(self.obs_buf[..., -4 * (self.num_agents-1)+idx], 'observations', 'ado_rotc' + str(idx+1))
            self.log_info_stats(self.obs_buf[..., -3 * (self.num_agents-1)+idx], 'observations', 'ado_velx' + str(idx+1))
            self.log_info_stats(self.obs_buf[..., -2 * (self.num_agents-1)+idx], 'observations', 'ado_vely' + str(idx+1))
            self.log_info_stats(self.obs_buf[..., -1 * (self.num_agents-1)+idx], 'observations', 'ado_vela' + str(idx+1))
        self.log_info_stats(self.obs_buf[..., 0], 'observations', 'ego_velx')
        self.log_info_stats(self.obs_buf[..., 1], 'observations', 'ego_vely')
        self.log_info_stats(self.obs_buf[..., 2], 'observations', 'ego_vela')

    def log_info_stats(self, scalar, name1, name2):
        self.info[name1][name2 + '_mean'] = scalar.mean()
        self.info[name1][name2 + '_min'] = scalar.min()
        self.info[name1][name2 + '_max'] = scalar.max()
        self.info[name1][name2 + '_std'] = scalar.std()

    def post_physics_step(self) -> None:
        self.total_step += 1
        self.episode_length_buf += 1

        # get current tile positions
        dists = torch.norm(self.states[:, :, 0:2].unsqueeze(2) - self.active_centerlines.unsqueeze(1), dim=3)
        sort = torch.sort(dists, dim=2)
        self.dist_active_track_tile = sort[0][:, :, 0]
        self.active_track_tile = sort[1][:, :, 0]

        # check if any tire is off track  --> Moved below reset
        # self.is_on_track = ~torch.any(~torch.any(self.wheels_on_track_segments, dim=3), dim=2)
        self.is_on_track_per_wheel = torch.any(self.wheels_on_track_segments, dim=3)
        # self.num_wheels_off_track = ((~self.is_on_track_per_wheel) * 1.0).sum(dim=-1)
        self.is_on_track = torch.any(torch.any(self.wheels_on_track_segments, dim=3), dim=2)
        self.time_off_track += 1.0 * ~self.is_on_track
        self.time_off_track *= 1.0 * ~self.is_on_track  # reset if on track again


        # #update drag_reduction
        # self.drag_reduction_points[:, self.drag_reduction_write_idx:self.drag_reduction_write_idx+self.num_agents, :] = \
        #                 self.states[:,:,0:2]
        # self.drag_reduction_write_idx = (self.drag_reduction_write_idx + self.num_agents) % self.drag_reduction_points.shape[1]
        # leading_pts = self.states[:,:,0:2] + 5*torch.cat((torch.cos(self.states[:,:,2].view(self.num_envs, self.num_agents,1)),
        #                                                   torch.sin(self.states[:,:,2].view(self.num_envs, self.num_agents,1))), dim = 2)

        # dists = leading_pts.view(self.num_envs, self.num_agents,1,2) - self.drag_reduction_points.view(self.num_envs, 1, -1, 2)
        # dists = torch.min(torch.norm(dists, dim=3), dim=2)[0]
        # self.drag_reduced[:] = False 
        # self.drag_reduced = dists <= 3.0


        # Update ego position at end of LL episodes
        update_ego_pos = self.ll_ep_done[0, 0]
        # ego_pos_world = update_ego_pos * self.states[0, 0, :2]  #  + (1.0 - update_ego_pos) * self.ego_pos_world
        self.all_egoagent_pos_world_env0.append(update_ego_pos * self.states[0, 0, :2])
        self.all_targets_pos_world_env0.append(update_ego_pos * self.targets_pos_world[0, 0])

          
        #update drag_reduction
        self.drag_reduction_points[:, self.drag_reduction_write_idx:self.drag_reduction_write_idx+self.num_agents, :] = \
                        self.states[:,:,0:2]
        self.drag_reduction_write_idx = (self.drag_reduction_write_idx + self.num_agents) % self.drag_reduction_points.shape[1]
        self.leading_pts = self.states[:,:,0:2] + self.cfg['model']['drag_lead_distance']*torch.cat((torch.cos(self.states[:,:,2].view(self.num_envs, self.num_agents,1)),
                                                          torch.sin(self.states[:,:,2].view(self.num_envs, self.num_agents,1))), dim = 2)

        dists = self.leading_pts.view(self.num_envs, self.num_agents,1,2) - self.drag_reduction_points.view(self.num_envs, 1, -1, 2)
        dists = torch.min(torch.norm(dists, dim=3), dim=2)[0]
        self.drag_reduced[:] = False 
        if self.cfg['model']['use_drag_reduction']:
            self.drag_reduced = dists <= self.cfg['model']['drag_reduction_wake_width'] #3.0 #

        self.info["agent_active"] = self.active_agents

        # compute closest point on centerline
        self.tile_points = self.active_centerlines[:, self.active_track_tile, :]
        self.tile_points = self.tile_points[self.all_envs, self.all_envs, ...]
        self.tile_car_vec = self.states[:, :, 0:2] - self.tile_points
        angs = self.active_alphas[:, self.active_track_tile]
        angs = angs[self.all_envs, self.all_envs, ...]
        self.trackdir = torch.stack((torch.cos(angs), torch.sin(angs)), dim=2)
        self.trackperp = torch.stack((-torch.sin(angs), torch.cos(angs)), dim=2)

        # increment_idx = torch.where(self.active_track_tile - self.old_active_track_tile < 15 - self.track_tile_counts[self.active_track_ids].view(-1, 1))  # avoid farming turns? NOTE: track length - buffer
        # decrement_idx = torch.where(self.active_track_tile - self.old_active_track_tile > 15)

        # FIXME: decrement relative to track length not good, should use absolute tile count!
        increment_idx = torch.where(self.active_track_tile - self.old_active_track_tile < -0.95 * self.track_tile_counts[self.active_track_ids].view(-1, 1))  # 0.9
        decrement_idx = torch.where(self.active_track_tile - self.old_active_track_tile > 12)  # +0.20 * self.track_tile_counts[self.active_track_ids].view(-1, 1))  # 0.9

        self.lap_counter[increment_idx] += 1
        self.lap_counter[decrement_idx] -= 1

        self.sub_tile_progress = torch.einsum("eac, eac-> ea", self.trackdir, self.tile_car_vec)
        self.track_progress_no_laps = self.active_track_tile * self.tile_len + self.sub_tile_progress
        self.track_progress = self.track_progress_no_laps + self.lap_counter * self.track_lengths[
            self.active_track_ids
        ].view(-1, 1)
        dist_sort, ranks = torch.sort(self.track_progress*self.active_agents, dim=1, descending=True)
        self.ranks = torch.sort(ranks, dim=1)[1].to(self.ranks.dtype)

        for team in self.teams:
            self.teamranks[:, team] = torch.min(self.ranks[:, team], dim = 1)[0].reshape(-1,1).type(torch.int)
        
        if self.log_behavior_stats:
            #set to 1 when not starting from rank 1
            idx_uninit = torch.where(self.stats_start_from_behind_current == -1)[0]
            if len(idx_uninit):
                self.stats_start_from_behind_current[idx_uninit] = 1.0*(self.ranks[idx_uninit, 0]>0)
                self.stats_last_ranks[idx_uninit] = 1.0*self.ranks[idx_uninit]
                self.stats_overtakes_current[idx_uninit] = 0

        self.contouring_err = torch.einsum("eac, eac-> ea", self.trackperp, self.tile_car_vec)

        # ### HACK
        # self.dyn_model.max_vel_vec_dynamic = self.dyn_model.max_vel_vec + self.ranks[..., None] * 0.1

        # Compute reward terms
        self.vel = torch.norm(self.states[:,:,self.vn['S_DX']:self.vn['S_DY']+1], dim = -1)
        self.compute_rewards()
        self.rew_buf[:, :self.team_size] = self.rew_buf[:, :self.team_size] * self.update_data[..., None]  # mask HL obs that terminated
        if self.log_behavior_stats:
            self.update_current_behavior_stats()
        
        # get env_ids to reset
        self.time_out_buf *= False
        self.check_termination()

        # env_ids = torch.where(self.reset_buf)[0]
        env_ids = torch.where(torch.cumsum(self.reset_buf, axis=-1)[..., self.reset_ids])[0]
        # env_ids = torch.where(self.reset_buf_hl)[0]  # NOTE: step LL env until HL update
        self.info = {}
        self.agent_left_track |= self.time_off_track[:, :] > self.offtrack_reset
    
        is_reset_frame = False
        if len(env_ids):
            if 0 in env_ids:
                self.render()  # Additional render to display reset cause
                is_reset_frame = True
                self.viewer.update_attention()
                self.all_targets_pos_world_env0 = []
                self.all_egoagent_pos_world_env0 = []
                self.viewer.clear_values()
            if self.track_stats:
                self.reset_track_stats(env_ids)
            if self.log_behavior_stats:  #  and 0 in env_ids:
                self.reset_behavior_stats(env_ids)
                self.write_behavior_stats()
            self.reset_envs(env_ids)
            # Reset vars to track off-track / reset cause
            self.is_on_track_per_wheel[env_ids, ...] = True
            self.is_on_track[env_ids, ...] = True
            self.reset_cause[env_ids] = 0.
            self.time_left_frac[env_ids] = 0.
            self.wheel_off_frac[env_ids] = 0.
            # Last vel same as init (acc reward)
            self.last_vel = torch.norm(self.states[:,:,self.vn['S_DX']:self.vn['S_DY']+1], dim = -1)


        self.active_obs_template[...] = 1.0
        idx_1, idx_2 = torch.where(~self.active_agents)
        # remove ado observations of deactivated envs for active ones  -->  FIXME: doesn't seem quite right from ado team perspective
        for ag in range(self.num_agents):
            self.active_obs_template[idx_1, ag, self.ado_idx_lookup[idx_2, ag]] = 0

        self.compute_observations()
        
        # render before resetting values
        if not is_reset_frame:
            self.render()

        self.old_active_track_tile = self.active_track_tile
        self.old_track_progress = self.track_progress
        self.last_actions[:] = self.actions
        #self.last_steer[:,:,0] = self.actions[..., self.vn['A_STEER']]
        #self.last_last_steer[:] = self.last_steer[:]
        self.last_vel[:] = self.vel
        self.prev_ranks[:] = self.ranks
        self.prev_teamranks[:] = self.teamranks

        self.update_data = 1.0 * (self.reset_buf_tmp==False)
      

    def render(
        self,
    ) -> None:
        # if log_video freq is set only redner in fixed intervals
        if self.log_video_freq >= 0:
            self.viewer.mark_env(self.trained_agent_slot)
            if self.log_episode and self.log_video_active:
                self.viewer_events = self.viewer.render(
                    self.states[:, :, [0, 1, 2, 6]], self.slip, self.drag_reduced, self.wheel_locations_world, self.interpolated_centers, self.interpolated_bounds, 
                    self.targets_pos_world, self.targets_rew01_local, self.targets_rot_world, self.targets_dist_track, self.actions, self.time_off_track,
                    self._action_probs_hl, self._action_mean_ll, self._action_std_ll,
                    self.is_on_track_per_wheel, self.vels_body, self.dyn_model.max_vel_vec, self.step_reward_terms, self.reset_cause, self.track_progress, self.active_track_tile,
                    self.ranks if self.num_agents>1 else None, self._global_step, self.all_targets_pos_world_env0, self.all_egoagent_pos_world_env0, self.last_actions,
                    self.active_track_tile, self.targets_tile_idx, self._values_ll, None, None, self.use_ppc_vec, None, self._svo_probs_hl,
                )
        else:
            self.viewer_events = self.viewer.render(
                self.states[:, :, [0, 1, 2, 6]], self.slip, self.drag_reduced, self.wheel_locations_world, self.interpolated_centers, self.interpolated_bounds, 
                self.targets_pos_world, self.targets_rew01_local, self.targets_rot_world, self.targets_dist_track, self.actions, self.time_off_track,
                self._action_probs_hl, self._action_mean_ll, self._action_std_ll,
                self.is_on_track_per_wheel, self.vels_body, self.dyn_model.max_vel_vec, self.step_reward_terms, self.reset_cause, self.track_progress, self.active_track_tile,
                self.ranks if self.num_agents>1 else None, self._global_step, self.all_targets_pos_world_env0, self.all_egoagent_pos_world_env0, self.last_actions,
                self.active_track_tile, self.targets_tile_idx, self._values_ll, None, None, self.use_ppc_vec, None, self._svo_probs_hl,
            )

    def write_behavior_stats(self) -> None:
        # if (self._global_step % self.log_behavior_freq == 0) and (self._global_step > 0):
        if len(torch.concat(self.batch_stats_overtakes)) > 100:
          self.info["behavior"] = {}
          self.info["behavior"]["samples"] = len(torch.concat(self.batch_stats_overtakes))
          self.info["behavior"]["overtakes"] = torch.concat(self.batch_stats_overtakes).mean()
          self.info["behavior"]["collisions"] = torch.concat(self.batch_stats_num_collisions).mean()
          self.info["behavior"]["offtrack"] = torch.concat(self.batch_stats_ego_left_track).mean()
          self.info["behavior"]["leadtime"] = torch.concat(self.batch_stats_lead_time).mean()
          self.info["behavior"]["teamranks"] = torch.concat(self.batch_stats_rank_team).mean()
          self.info["behavior"]["lowerrank"] = torch.concat(self.batch_stats_rank_lower).mean()

          self.batch_stats_overtakes = []
          self.batch_stats_num_collisions = []
          self.batch_stats_ego_left_track = []
          self.batch_stats_lead_time = []
          self.batch_stats_rank_team = []
          self.batch_stats_rank_lower = []

    def simulate(self) -> None:
        # run physics update
        #FIXME removed delay
        #hardware_commands = self.last_actions.clone()
        #hardware_commands[..., self.vn['A_STEER']] = self.last_last_steer[..., 0]
        
        (
            self.states[:],
            self.contact_wrenches,
           #self.is_rear_end_contact,
            self.shove,
            self.wheels_on_track_segments,
            self.slip,
            self.wheel_locations_world,
        ) = step_cars(
            self.states,
            self.hardware_commands,
            self.drag_reduced,
            self.wheel_locations,
            self.R,
            self.contact_wrenches,
            self.rear_end_contact,
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
     
    # def reset_lstm(self):
    #     self.dyn_model.initialize_lstm_states(torch.zeros((self.num_envs * self.num_agents, 50, 6)).to(self.device))

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

    def set_reset_agent_ids(self, reset_id_num):
        self.reset_ids = reset_id_num - 1

    def reset_behavior_stats(self, env_ids):
        self.batch_stats_overtakes.append(self.stats_overtakes_current[env_ids])
        self.batch_stats_num_collisions.append(self.stats_num_collisions_current[env_ids])
        self.batch_stats_ego_left_track.append(self.stats_ego_left_track_current[env_ids])
        self.batch_stats_lead_time.append(self.stats_lead_time_current[env_ids])
        self.batch_stats_rank_team.append((1.0*self.teamranks[env_ids, 0]))
        self.batch_stats_rank_lower.append((1.0*self.ranks[env_ids, :self.team_size].max(dim=-1)[0]))

        self.stats_overtakes_last_race[env_ids] = self.stats_overtakes_current[env_ids]
        self.stats_num_collisions_last_race[env_ids] = self.stats_num_collisions_current[env_ids]
        self.stats_ego_left_track_last_race[env_ids] = self.stats_ego_left_track_current[env_ids]
        self.stats_ado_left_track_last_race[env_ids] = self.stats_ado_left_track_current[env_ids]
        self.stats_win_from_behind_last_race[env_ids] = (self.ranks[env_ids, 0] == 0)*self.stats_start_from_behind_current[env_ids]  -1* (self.ranks[env_ids, 0] != 0)*self.stats_start_from_behind_current[env_ids]
        self.stats_lead_time_last_race[env_ids] = self.stats_lead_time_current[env_ids]

        self.stats_overtakes_current[env_ids] = 0
        self.stats_num_collisions_current[env_ids] = 0
        self.stats_ego_left_track_current[env_ids] = 0 
        self.stats_ado_left_track_current[env_ids] = 0 
        self.stats_start_from_behind_current[env_ids] = -1
        self.stats_lead_time_current[env_ids] = 0
        self.stats_lap_time_current[env_ids] = 0

    def reset_track_stats(self, env_ids):
        self.stats_percentage_sa_laptime_last_race[env_ids] = -1

    def update_current_behavior_stats(self):
        self.stats_overtakes_current[:] += 1.0*((self.ranks[:,0] - self.stats_last_ranks[:,0]) < 0)  # FIXME: does not account for rank loss due to off-track switch
        self.stats_num_collisions_current[:] += 1.0 * self.is_collision[:,0]
        self.stats_ego_left_track_current[:] += 1.0 * ~self.is_on_track[:,0]
        self.stats_ado_left_track_current[:] += 1.0 * torch.any(~self.is_on_track[:,1:] , dim =1).view(-1,)
        self.stats_lead_time_current[:] += self.dt*(self.ranks[:, 0]==0)
        
        self.stats_last_ranks[:] = self.ranks[:] 

        time_ep = (self.episode_length_buf*self.dt*self.decimation).view(-1,)
        prog_ep = torch.clip(self.track_progress[:, 0].view(-1,), min = 0)
        track_len_ep = self.track_lengths[self.active_track_ids].view(-1,)
        self.stats_lap_time_current[:] = (track_len_ep/ (prog_ep+1e-6)) * time_ep 


# OLD REFERENCE
# def compute_rewards_jit(
#     rew_buf: torch.Tensor,
#     track_progress: torch.Tensor,
#     old_track_progress: torch.Tensor,
#     # progress_other: torch.Tensor,
#     is_on_track: torch.Tensor,
#     reward_scales: Dict[str, float],
#     actions: torch.Tensor,
#     last_actions: torch.Tensor,
#     vn: Dict[str, int],
#     states: torch.Tensor,
#     vel: torch.Tensor,
#     last_vel : torch.Tensor,
#     ranks: Union[torch.Tensor,None],
#     teamranks: Union[torch.Tensor,None],
#     prev_ranks: Union[torch.Tensor,None],
#     is_collision: torch.Tensor,
#     reward_terms: Dict[str, torch.Tensor],
#     step_reward_terms: Dict[str, torch.Tensor],
#     num_agents: int,
#     active_agents: torch.Tensor,
#     dt: float,
#     dt_hl: float,
#     targets_dist_local: torch.Tensor,
#     targets_std_local: torch.Tensor,
#     ll_ep_done: torch.Tensor,
#     ll_steps_left: torch.Tensor,
#     train_ll: float,
#     is_on_track_per_wheel: torch.Tensor,
#     targets_off_ahead: torch.Tensor,
# ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

#     train_hl = 1.0 - train_ll

#     ### Progress reward
#     # Base
#     progress_delta = track_progress - old_track_progress
#     progress_jump = 1.0 * (torch.abs(delta_progress) > 3.5)
#     rew_progress = torch.clip(progress_delta, min=-10, max=10) * (1.0 - progress_jump)
#     # LL mod
#     rew_progress_ll = 1.0 * rew_progress
#     # HL mod
#     rew_progress_hl = 1.0 * rew_progress
#     # Total
#     rew_progress *= train_ll * rew_progress_ll + train_hl * rew_progress_hl
#     rew_progress *= reward_scales["progress"]




#     delta_progress = track_progress - old_track_progress
#     progress_jump = 1.0 * (torch.abs(delta_progress) > 3.5)
#     positive_prog = 1.0 * (delta_progress > 0.0)
#     rew_progress = 0.0
#     rew_progress += torch.clip(delta_progress, min=-10, max=10) * reward_scales["progress"] * (1.0 - progress_jump)
#     rew_progress *= train_ll * 1.0 + (1.0 - train_ll) * 1.0 / (1.0 + teamranks)

#     pen_progress = -2.0 * progress_jump  # NOTE: alternative, penalize reset step
#     # pen_progress *= (train_ll + (1.0-train_ll))  #  * 1.0 / dt)
#     rew_progress += pen_progress * reward_scales["progress"]

#     # Penalize small forward velocities
#     rew_velocity = 1.0 * torch.exp(-0.5 * (vel / 0.05)**2) * reward_scales["velocity"] * train_ll

#     # ### Off-track penalties ###
#     # rew_on_track = reward_scales["offtrack"] * ~is_on_track * train_ll
#     # rew_on_track = reward_scales["offtrack"] * (torch.exp(num_wheels_off_track) - 1.0) / 1.0 * train_ll
#     # Until 02/20/2023
#     front_axel_off_track = (1.0 * ~is_on_track_per_wheel[..., 0]) * (1.0 * ~is_on_track_per_wheel[..., 1])
#     back_axel_off_track = (1.0 * ~is_on_track_per_wheel[..., 2]) * (1.0 * ~is_on_track_per_wheel[..., 3])
#     num_wheels_off_track = ((~is_on_track_per_wheel) * 1.0).sum(dim=-1)
#     rew_on_track = reward_scales["offtrack"] * (0.5*num_wheels_off_track + 1.0*front_axel_off_track + 1.0*back_axel_off_track) * train_ll
#     # rew_on_track = (1.0 - 1.0*is_on_track_per_wheel.all(dim=-1)) * reward_scales["offtrack"]

#     act_diff = torch.square(actions - last_actions)
#     rew_acc = torch.square(vel-last_vel.view(-1,num_agents))*reward_scales['acceleration'] * train_ll
#     act_diff[..., 0] *= 5.0  # 20.0  # 5.0  # 25.0
#     act_diff[..., 1] *= 1.0  # 10.0
#     rew_actionrate = torch.sum(act_diff, dim=2) * reward_scales["actionrate"] * train_ll
#     #rew_energy = -torch.sum(torch.square(states[:, :, vn["S_W0"] : vn["S_W3"] + 1]), dim=2) * reward_scales["energy"]
#     num_active_agents = torch.sum(1.0 * active_agents, dim=1)

#     ### Goal distance reward --> FIXME: does not rotate distribution
#     # # OLD
#     # targets_dist_local *= 2**(2.0-0.75*targets_std_local/0.1)  # new scaling to separate distributions
#     # exponent = -0.5 * (targets_dist_local / targets_std_local)**2
#     # pdf_dist = 1.0/(np.sqrt(2.0 * np.pi) * targets_std_local) * torch.exp(exponent)
#     # pdf_dist /= 3.0  # 4.0 = max attainable ~1.0
#     # # pdf_dist /= 1.0/(np.sqrt(2.0 * np.pi) * torch.sqrt(targets_std_local)/2.0)

#     # NEW
#     pdf_dist = get_multivariate_reward(targets_dist_local, targets_std_local)

#     rew_dist_base = pdf_dist * reward_scales['goal']
#     # Sparse
#     scale_sparse = 0.0  #  0.3 * ll_ep_done * targets_off_ahead / 5.0 #  / dt # *0.1  # * targets_off_ahead / 5.0  # encourage larger offsets (?) [/5.0]
#     # Dense
#     # scale_dense = (1.0 / (torch.exp(1.0 * torch.norm(targets_dist_local, dim=-1, keepdim=True))))
#     # scale_dense = 1.0 / torch.exp(5. * (ll_steps_left-0.05))
#     scale_dense = 0.05 / ll_steps_left.squeeze(-1)
#     # Combined
#     scale = train_ll * scale_dense + (1.0-train_ll) * scale_sparse 
#     rew_dist = scale * rew_dist_base

#     # rew_dist = torch.mean(rew_dist, dim=-1)
#     # Only reward y proximity if actually close to target
#     # rew_dist = rew_dist[..., 0] + 1.0 / (5e1 * targets_dist_local[..., 0].abs() + 1e0) * rew_dist[..., 1]
#     # rew_dist = rew_dist * (targets_off_ahead[..., 0] / 5.0)  # encourage larger offsets (?) [/5.0]

#     # # OLD
#     # rew_dist = rew_dist[..., 0] * rew_dist[..., 1]  # *


#     if ranks is not None:
#         #compute relative progress
#         rew_rel_prog = 0.0*rew_on_track  # -torch.clip(progress_other, -1, 1).sum(dim=-1)* reward_scales["rank"] * (1.0 - train_ll)
        
#         #rew_rank = 1.0 / num_agents * (num_active_agents.view(-1, 1) / 2.0 - ranks) * reward_scales["rank"]
#         # rew_rank = torch.exp(-2.0 * teamranks) * reward_scales["rank"] * (1.0 - train_ll) * ll_ep_done[..., 0]
#         # rew_rank = torch.exp(-2.0 * ranks) * reward_scales["rank"] * (1.0 - train_ll) * ll_ep_done[..., 0]
#         # rew_rank = (1.2*torch.exp(-1.0 * ranks) - 0.2) * reward_scales["rank"] * (1.0 - train_ll) * ll_ep_done[..., 0] * dt_hl  #  / dt
        
#         # rew_rank = 1.0 * (prev_ranks < ranks) - 1.0 * (prev_ranks > ranks)
#         # rew_rank = 1.0 * (prev_ranks - ranks) + 0.1 * (teamranks==0)
#         rew_rank = 1.0 * (prev_ranks - ranks) / (1.0 + torch.minimum(prev_ranks, ranks))
#         rew_rank *= reward_scales["rank"] * (1.0 - train_ll) * ll_ep_done[..., 0] * dt_hl

#         rew_collision = is_collision * reward_scales["collision"] * train_ll

#         rew_baseline = 0.15*dt * (train_ll + (1.0 - train_ll) * ll_ep_done[..., 0])

#         rew_buf[..., 0] = torch.clip(
#             rew_progress + rew_on_track + rew_actionrate + rew_rel_prog + rew_acc + rew_baseline, min=-5*dt, max=None
#         ) + rew_collision + rew_dist + rew_rank + rew_velocity
#     else:
#         #needs to be a tensor
#         rew_rank = 0*rew_on_track
#         rew_collision = 0*rew_on_track
#         rew_buf[..., 0] = torch.clip(
#             rew_progress + rew_on_track + rew_actionrate + rew_acc + 0.1*dt , min=-5*dt, max=None
#         ) + rew_dist + rew_rank + rew_velocity

#     reward_terms["progress"] += rew_progress
#     reward_terms["offtrack"] += rew_on_track
#     reward_terms["actionrate"] += rew_actionrate
#     #reward_terms["energy"] += rew_energy
#     reward_terms["acceleration"] += rew_acc

#     reward_terms["goal"] += rew_dist

#     reward_terms["velocity"] += rew_velocity

#     if ranks is not None:
#         reward_terms["rank"] += rew_rank  # rew_rel_prog
#         reward_terms["collision"] += rew_collision

#     step_reward_terms["progress"] = rew_progress / dt
#     step_reward_terms["offtrack"] = rew_on_track / dt
#     step_reward_terms["actionrate"] = rew_actionrate / dt
#     #step_reward_terms["energy"] += rew_energy
#     step_reward_terms["acceleration"] = rew_acc / dt
#     step_reward_terms["goal"] = rew_dist / dt
#     step_reward_terms["velocity"] = rew_velocity / dt
#     if ranks is not None:
#         step_reward_terms["rank"] = rew_rank / dt  # rew_rel_prog / dt
#         step_reward_terms["collision"] = rew_collision / dt


#     return rew_buf, reward_terms, step_reward_terms




### jit functions
#@torch.jit.script
def compute_rewards_jit(
    rew_buf: torch.Tensor,
    track_progress: torch.Tensor,
    old_track_progress: torch.Tensor,
    # progress_other: torch.Tensor,
    is_on_track: torch.Tensor,
    reward_scales: Dict[str, float],
    actions: torch.Tensor,
    last_actions: torch.Tensor,
    vn: Dict[str, int],
    states: torch.Tensor,
    vel: torch.Tensor,
    last_vel : torch.Tensor,
    ranks: Union[torch.Tensor,None],
    teamranks: Union[torch.Tensor,None],
    prev_ranks: Union[torch.Tensor,None],
    prev_teamranks: Union[torch.Tensor,None],
    is_collision: torch.Tensor,
    is_rear_end_collision: torch.Tensor,
    reward_terms: Dict[str, torch.Tensor],
    step_reward_terms: Dict[str, torch.Tensor],
    num_agents: int,
    active_agents: torch.Tensor,
    dt: float,
    dt_hl: float,
    targets_dist_local: torch.Tensor,
    targets_std_local: torch.Tensor,
    ll_ep_done: torch.Tensor,
    ll_steps_left: torch.Tensor,
    train_ll: float,
    is_on_track_per_wheel: torch.Tensor,
    targets_off_ahead: torch.Tensor,
    time_left_frac: torch.Tensor,
    prog_jump_lim: float,
    initial_team_rank: torch.Tensor,
    initial_rank: torch.Tensor,
    use_hierarchical_policy: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    train_hl = 1.0 - train_ll

    ### Progress reward
    # Base
    progress_delta = track_progress - old_track_progress
    progress_jump = 1.0 * (torch.abs(progress_delta) > prog_jump_lim)
    rew_progress = 0 * torch.clip(progress_delta, min=-10, max=10) * (1.0 - progress_jump)
    rew_progress -= 5.0 * progress_jump  # -3.0  # -5.0
    # LL mod
    rew_progress_ll = 1.0 * rew_progress
    # HL mod
    rew_progress_hl = 0.0 * rew_progress
    
    ### Small velocity penalty
    # Base
    rew_velocity = 1.0 *  torch.exp(-torch.abs(vel))  # torch.exp(-0.5 * (vel / 0.05)**2)
    # LL mod
    rew_velocity_ll = 1.0 * rew_velocity
    # HL mod
    rew_velocity_hl = 0.0 * rew_velocity

    ### Off-track penalty
    # Base
    front_axel_off_track = (1.0 * ~is_on_track_per_wheel[..., 0]) * (1.0 * ~is_on_track_per_wheel[..., 1])
    back_axel_off_track = (1.0 * ~is_on_track_per_wheel[..., 2]) * (1.0 * ~is_on_track_per_wheel[..., 3])
    num_wheels_off_track = ((~is_on_track_per_wheel) * 1.0).sum(dim=-1)
    rew_on_track = 0.5*num_wheels_off_track + 1.0*front_axel_off_track + 1.0*back_axel_off_track
    # rew_on_track -= (num_wheels_off_track == 0) * 0.1  # 0.5
    # LL mod
    rew_on_track_ll = 1.0 * rew_on_track
    # HL mod
    rew_on_track_hl = 0.0 * rew_on_track

    ### Acceleration penalty
    # Base
    rew_acc = torch.square(vel-last_vel.view(-1,num_agents))
    # LL mod
    rew_acc_ll = 1.0 * rew_acc
    # HL mod
    rew_acc_hl = 0.0 * rew_acc

    ### Action rate penalty
    # Base
    act_diff = torch.square(actions - last_actions)
    act_diff[..., 0] *= 1.0  # 5.0  # 20.0  # 5.0  # 25.0
    act_diff[..., 1] *= 1.0  # 1.0  # 10.0
    rew_actionrate = torch.sum(act_diff, dim=2)
    # LL mod
    rew_actionrate_ll = 1.0 * rew_actionrate
    # HL mod
    rew_actionrate_hl = 0.0 * rew_actionrate

    ### Goal distance reward
    # Base
    rew_goal = get_multivariate_reward(targets_dist_local, targets_std_local)
    # LL mod
    # rew_goal_ll = 0.05 / ll_steps_left.squeeze(-1) * rew_goal * (vel > 0.5) * torch.exp((vel-0.5)/5.0)  # NOTE: pre-04/24/23
    rew_goal_ll = (ll_steps_left.squeeze(-1) < 0.15) * rew_goal * torch.exp((vel-2.0)/4.0)  # NOTE: pre-05/05/23  ||| * (vel > 0.5)
    # rew_goal_ll = 0.05 / ll_steps_left.squeeze(-1) * rew_goal * torch.exp((vel-2.0)/4.0)
    # HL mod
    rew_goal_hl = 0.0 * rew_goal  #+ 5.e-2 * targets_off_ahead[..., 0] / 5.0 * ll_ep_done[..., 0]  # bias towards far away goals

    ### Baseline reward
    # Base
    rew_baseline = 0.15*dt
    # LL mod
    rew_baseline_ll = 1.0 * rew_baseline
    # HL mod
    rew_baseline_hl = ll_ep_done[..., 0] * rew_baseline

    if ranks is not None:
        ### Relative progress reward
        # Base
        rew_rel_prog = 0.0*rew_on_track
        # LL mod
        rew_rel_prog_ll = 0.0 * rew_rel_prog
        # HL mod
        rew_rel_prog_hl = 0.0 * rew_rel_prog

        ### Rank reward
        # Base
        # rew_rank = 1.0 * (prev_ranks - ranks) / (1.0 + torch.minimum(prev_ranks, ranks)) # + 1.e-2*torch.exp(-teamranks)
        # ### Team-based
        # rew_rank = 1.0 * (prev_teamranks - teamranks) / (1.0 + torch.minimum(prev_teamranks, teamranks))  # NOTE: testing as alternative to above
        # rew_rank += (time_left_frac[..., 0]==1.0) * (teamranks==0) * 2.0 # NOTE: removed this and below on 05/04/23 evening
        # rew_rank += (time_left_frac[..., 0]==1.0) * (initial_team_rank - teamranks) * 0.5  # * 0.25  # 0.1  # NOTE: not visible from obs
        # ### Individual-based
        # rew_rank = 1.0 * (prev_ranks - ranks) / (1.0 + torch.minimum(prev_ranks, ranks))  # NOTE: testing as alternative to above
        # rew_rank += (time_left_frac[..., 0]==1.0) * (ranks==0) * 2.0 # NOTE: removed this and below on 05/04/23 evening
        # rew_rank += (time_left_frac[..., 0]==1.0) * (initial_rank - ranks) * 0.5  # * 0.25  # 0.1  # NOTE: not visible from obs
        # rew_rank += 1.e-1*torch.exp(-ranks)  # NOTE: removed 05/09/2023
        # # NOTE: replaced above with this
        # rew_rank += (teamranks==0) * 0.5
        # rew_rank += (initial_team_rank - teamranks) * 0.1

        ### Different rank versions
        # # Version 1
        # rew_rank = 1.0 * (prev_ranks - ranks) / (1.0 + torch.minimum(prev_ranks, ranks))
        # rew_rank += (time_left_frac[..., 0]==1.0) * (teamranks==0) * 2.0
        # rew_rank += (time_left_frac[..., 0]==1.0) * (initial_team_rank - teamranks) * 0.25
        # rew_rank += 1.e-1*torch.exp(-ranks)
        # rew_rank += 1.e-2*torch.exp(-teamranks)
        # # Version 2
        # rew_rank = 1.0 * (prev_ranks - ranks) / (1.0 + torch.minimum(prev_ranks, ranks))
        # rew_rank += (time_left_frac[..., 0]==1.0) * (teamranks==0) * 2.0
        # rew_rank += (time_left_frac[..., 0]==1.0) * (initial_team_rank - teamranks) * 0.25
        # # Version 3
        # rew_rank = 1.0 * (prev_teamranks - teamranks) / (1.0 + torch.minimum(prev_teamranks, teamranks))
        # rew_rank += (time_left_frac[..., 0]==1.0) * (teamranks==0) * 2.0
        # rew_rank += (time_left_frac[..., 0]==1.0) * (initial_team_rank - teamranks) * 0.25
        # # Version 4
        # rew_rank = 1.0 * (prev_teamranks - teamranks) / (1.0 + torch.minimum(prev_teamranks, teamranks))
        # rew_rank += (time_left_frac[..., 0]==1.0) * (teamranks==0) * 2.0
        # rew_rank += (time_left_frac[..., 0]==1.0) * (initial_team_rank - teamranks) * 0.25
        # rew_rank += 1.e-1*torch.exp(-ranks)
        # rew_rank += 1.e-2*torch.exp(-teamranks)
        # # Version 9
        # rew_rank = 1.0 * (prev_ranks - ranks) / (1.0 + torch.minimum(prev_ranks, ranks))
        # rew_rank += (time_left_frac[..., 0]==1.0) * (ranks==0) * 2.0
        # rew_rank += (time_left_frac[..., 0]==1.0) * (initial_rank - ranks) * 0.25
        # rew_rank += 1.e-1*torch.exp(-ranks)

        # # Version 11
        # rew_rank = 1.0 * (prev_teamranks - teamranks) / (1.0 + torch.minimum(prev_teamranks, teamranks))
        # rew_rank += 1.0 * (teamranks==0)
        # rew_rank += 1.e-1*torch.exp(-ranks)

        # Version 13
        rew_rank = 1.0 * (prev_ranks - ranks) / (1.0 + torch.minimum(prev_ranks, ranks))
        rew_rank += 1.0 * (ranks==0)
        rew_rank += 1.e-1*torch.exp(-ranks)

        # LL mod
        rew_rank_ll = 0.0 * rew_rank
        # HL mod
        rew_rank_hl = 1.0 * rew_rank * ll_ep_done[..., 0]  # * dt_hl  #  + (teamranks==0) * dt_hl
        # rew_rank_hl += 10.0 * (eps_length > 0.95*max_length) * (teamranks==0)

        rew_rank_ll = (1.0 - use_hierarchical_policy) * rew_rank_hl

        ### Collision reward
        # Base
        rew_collision = is_collision
        rew_rearend_collision = is_rear_end_collision
        # LL mod
        rew_collision_ll = 1.0 * rew_collision
        rew_rearend_collision_ll = 1.0 * rew_rearend_collision
        # HL mod
        rew_collision_hl = 0.0 * rew_collision
        rew_rearend_collision_hl = 0.0 * rew_rearend_collision
        
    else:
        ### Relative progress reward
        # Base
        rew_rel_prog = 0.0*rew_on_track
        # LL mod
        rew_rel_prog_ll = 0.0 * rew_rel_prog
        # HL mod
        rew_rel_prog_hl = 0.0 * rew_rel_prog

        ### Rank reward
        # Base
        rew_rank = 0.0*rew_on_track
        # LL mod
        rew_rank_ll = 0.0 * rew_rank
        # HL mod
        rew_rank_hl = 0.0 * rew_rank

        ### Collision reward
        # Base
        rew_collision = 0.0*rew_on_track
        # LL mod
        rew_collision_ll = 0.0 * rew_collision
        # HL mod
        rew_collision_hl = 0.0 * rew_collision

    ### Total rewards
    # Progress
    rew_progress = train_ll * rew_progress_ll + train_hl * rew_progress_hl
    rew_progress *= reward_scales["progress"] * reward_scales["total"]
    # Small velocity
    rew_velocity = train_ll * rew_velocity_ll + train_hl * rew_velocity_hl
    rew_velocity *= reward_scales["velocity"] * reward_scales["total"]
    # Off-track
    rew_on_track = train_ll * rew_on_track_ll + train_hl * rew_on_track_hl
    rew_on_track *= reward_scales["offtrack"] * reward_scales["total"]
    # Acceleration
    rew_acc = train_ll * rew_acc_ll + train_hl * rew_acc_hl
    rew_acc *= reward_scales['acceleration'] * reward_scales["total"]
    # Action rate
    rew_actionrate = train_ll * rew_actionrate_ll + train_hl * rew_actionrate_hl
    rew_actionrate *= reward_scales["actionrate"] * reward_scales["total"]
    # Goal
    rew_goal = train_ll * rew_goal_ll + train_hl * rew_goal_hl
    rew_goal *= reward_scales['goal'] * reward_scales["total"]
    # Relative progress
    rew_rel_prog = train_ll * rew_rel_prog_ll + train_hl * rew_rel_prog_hl
    rew_rel_prog *= reward_scales['rank'] * reward_scales["total"]
    # Rank
    rew_rank = train_ll * rew_rank_ll + train_hl * rew_rank_hl
    rew_rank *= reward_scales["rank"] * reward_scales["total"]
    # Collision
    rew_collision = train_ll * rew_collision_ll + train_hl * rew_collision_hl
    rew_collision *= reward_scales["collision"] * reward_scales["total"]
    # Penalize Rear End Collision Perpetrator extra
    rew_rearend_collision = train_ll * rew_rearend_collision_ll[..., 0] + train_hl * rew_rearend_collision_hl[..., 0]
    rew_rearend_collision *= reward_scales["rearendcollision"] * reward_scales["total"]
    # Baseline
    rew_baseline = (train_ll * rew_baseline_ll + train_hl * rew_baseline_hl) * reward_scales["total"]

    # Combined
    rew_buf[..., 0] = torch.clip(
            rew_progress + rew_collision +rew_rearend_collision+ rew_velocity + rew_on_track + rew_actionrate + rew_rel_prog + rew_acc + rew_baseline, min=-5*dt, max=None  #-5*dt
        ) + rew_goal + rew_rank

    reward_terms["progress"] += rew_progress
    reward_terms["offtrack"] += rew_on_track
    reward_terms["actionrate"] += rew_actionrate
    reward_terms["acceleration"] += rew_acc

    reward_terms["goal"] += rew_goal

    reward_terms["velocity"] += rew_velocity

    if ranks is not None:
        reward_terms["rank"] += rew_rank  # rew_rel_prog
        reward_terms["collision"] += rew_collision
        reward_terms["rearendcollision"] += rew_rearend_collision

    step_reward_terms["progress"] = rew_progress / dt
    step_reward_terms["offtrack"] = rew_on_track / dt
    step_reward_terms["actionrate"] = rew_actionrate / dt
    step_reward_terms["acceleration"] = rew_acc / dt
    step_reward_terms["goal"] = rew_goal / dt
    step_reward_terms["velocity"] = rew_velocity / dt
    if ranks is not None:
        step_reward_terms["rank"] = rew_rank / dt  # rew_rel_prog / dt
        step_reward_terms["collision"] = rew_collision / dt
        step_reward_terms["rearendcollision"] = rew_rearend_collision / dt


    return rew_buf, reward_terms, step_reward_terms


def get_multivariate_reward(dist, stddev):
    exponent = -0.5 * ((dist / stddev)**2).sum(axis=-1)
    denominator = (2*torch.pi * torch.sqrt(torch.prod(stddev, axis=-1)))
    pdf_dist = torch.exp(exponent) / denominator
    return pdf_dist
