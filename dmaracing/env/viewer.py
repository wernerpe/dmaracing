import cv2 as cv
import torch
import numpy as np
import sys

from matplotlib import pyplot as plt
from trueskill import TrueSkill

from dmaracing.utils.trackgen import draw_track


class Viewer:
    def __init__(self,
                 cfg,
                 track_centerlines, 
                 track_poly_verts, 
                 track_border_poly_verts, 
                 track_border_poly_cols,
                 track_tile_counts,
                 active_track_ids,
                 teamsize=None,
                 headless=True):

        self.device = 'cuda:0'
        self.cfg = cfg

        self._headless = headless
        self._frames = []
        self._frames_val = []

        self._values_ll_min = +100.0
        self._values_ll_max = -100.0
        self._values_ll = []
        self._states_ll = []

        self.track_centerlines = track_centerlines.cpu().numpy()
        self.track_poly_verts = track_poly_verts
        self.track_border_poly_verts = track_border_poly_verts 
        self.track_border_poly_cols = track_border_poly_cols
        self.active_track_ids = active_track_ids
        self.track_tile_counts = track_tile_counts.cpu().numpy()

        #load cfg
        self.width = self.cfg['viewer']['width_tb'] if self._headless else self.cfg['viewer']['width']
        self.height = self.cfg['viewer']['height_tb'] if self._headless else self.cfg['viewer']['height']
        self.scale_x = self.cfg['viewer']['scale']
        self.scale_y = self.height/self.width*self.scale_x
        self.thickness = self.cfg['viewer']['linethickness']
        self.draw_multiagent = self.cfg['viewer']['multiagent']
        self.max_agents = self.cfg['viewer']['maxAgents']
        self.num_agents = self.cfg['sim']['numAgents']
        self.num_envs = self.cfg['sim']['numEnv']

        self.act_ll_off = self.cfg['learn']['defaultactions']
        self.act_ll_scl = self.cfg['learn']['actionscale']

        self.teamsize = teamsize
        
        if self.draw_multiagent:
            self.num_cars = self.num_agents
        else:
            self.num_cars = min(self.max_agents, self.num_envs)
        
        #bounding box of car in model frame
    
        w = self.cfg['model']['W_coll']
        lf = self.cfg['model']['L_coll']/2  
        lr = self.cfg['model']['L_coll']/2
        self.car_box_m = torch.tensor([[lf, -w/2],[lf, w/2],[-lr, w/2], [-lr, -w/2]], device = self.device)
        self.car_box_m = self.car_box_m.unsqueeze(2).repeat(1, 1, self.num_cars)
        self.car_box_m = torch.transpose(self.car_box_m, 0,1) 
        self.car_heading_m = torch.tensor([[0, 0],[lf, 0]], device = self.device)
        self.car_heading_m = self.car_heading_m.unsqueeze(2).repeat(1, 1, self.num_cars)
        self.car_heading_m = torch.transpose(self.car_heading_m, 0,1)
        self.R = torch.zeros((2, 2, self.num_cars), device=self.device, dtype = torch.float)
        self.img = 255*np.ones((self.height, self.width, 3), np.uint8)
        self.img_val = 255*np.ones((self.height, self.width, 3), np.uint8)
        self.track_canvas = self.img.copy()
        self.colors = 255.0/self.num_cars*np.arange(self.num_cars) 
        self.font = cv.FONT_HERSHEY_SIMPLEX

        self.do_render = True
        self.env_idx_render = 0
        
        self.x_offset = -50
        self.y_offset = 0
        
        self.points = []
        self.msg = []
        self.marked_env = None
        self.state = []
        self.slip_markers = []
        self.drag_reduced_markers = []
        self.lines = []
        self.draw_track()
        if not self._headless:
            cv.imshow('dmaracing', self.track_canvas)

        self.attention = None

    def center_cam(self, state):
        self.scale_x /= 0.7
        self.scale_y /= 0.7 
        self.draw_track()


    def _render_tb(self, show_val=False):
        self.draw_track()
        self._frames.append(self.img)
        if show_val:
            self._frames_val.append(self.img_val)
        return None
        

    def render(
      self, 
      state, slip, drag_reduced, wheel_locs, lookahead, bounds, 
      targets=None, targets_rew01=None, targets_angle=None, targets_distance=None, 
      actions=None, time_off_track=None, 
      hl_action_probs=None, ll_action_mean=None, ll_action_std=None,
      wheels_on_track=None, curr_vels=None, max_vels=None,
      reward_terms=None, reset_cause=None, track_progress=None, active_tile=None,
      ranks=None, global_step=None, all_targets_pos_world_env0=None, all_egoagent_pos_world_env0=None,
      last_actions=None, tile_idx_car=None, tile_idx_trg=None, values_ll=None, values_ll_min=None, values_ll_max=None,
      use_ppc_vec=None, ppc_lookahead_points=None, hl_svo_probs=None):
        self.state = state.clone()
        self.slip = slip.clone()
        self.wheel_locs = wheel_locs.clone()
        self.drag_reduced = drag_reduced.clone()

        show_val = False

        draw_reduced = False  # False

        if self.do_render:
            self.img = self.track_canvas.copy()
            self.car_img = self.img.copy()
           
            if self.draw_multiagent:
                self.add_slip_markers()
                self.draw_slip_markers()
                self.draw_multiagent_rep(state)
                if not draw_reduced:
                  self.draw_lookahead_markers(lookahead, bounds)
                if targets_rew01 is not None and not draw_reduced:
                  self.draw_target_marker(targets, targets_rew01, targets_angle)
            else:
                self.draw_singleagent_rep(state[:self.num_cars])
            cv.putText(self.img, "Env ID: " + str(self.env_idx_render), (15, 50), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            if global_step is not None:
                  cv.putText(self.img, "Gstep: " + str(global_step), (15, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            # if targets_distance is not None:
            #     target_dist = targets_distance[0, 0, :].cpu().numpy()
            #     cv.putText(self.img, "dGx: " + "{:.2f}".format(target_dist[0]), (350, 50), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            #     cv.putText(self.img, "dGy: " + "{:.2f}".format(target_dist[1]), (350, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            # if actions is not None:
            #     actions = actions[0, 0, :].cpu().numpy()
            #     cv.putText(self.img, "uSt: " + "{:.2f}".format(actions[0]), (450, 50), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            #     cv.putText(self.img, "uTh: " + "{:.2f}".format(actions[1]), (450, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            # if time_off_track is not None:
            #     time_off_track = time_off_track[0, 0].cpu().numpy()
            #     cv.putText(self.img, "off: " + str(int(time_off_track)), (35, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            
            # Moved below
            # self.draw_points()
            # self.draw_lines()
            # self.draw_string()
            # self.draw_marked_agents()

            if hl_action_probs is not None:
                self.draw_action_distributions_new(hl_action_probs, ll_action_mean, ll_action_std, targets_distance, actions, last_actions)

            if hl_svo_probs is not None:
                self.draw_svo_distribution(hl_svo_probs)

            if wheels_on_track is not None and not draw_reduced:
                self.draw_wheel_on_track_indicators(wheels_on_track, time_off_track)

            if max_vels is not None:
                self.draw_max_vel(max_vels, use_ppc_vec)

            if curr_vels is not None:
                self.draw_curr_vel(curr_vels)

            if ranks is not None:
                self.draw_ranks(ranks)

            if reward_terms is not None and not draw_reduced:
                self.draw_rew_terms(reward_terms)

            if self.teamsize is not None and False:
                self.draw_team_stats_debug(state, actions)
            
            if reset_cause is not None:
                self.draw_reset_cause(reset_cause)

            if tile_idx_car is not None and tile_idx_trg is not None:
                self.draw_tile_idx(tile_idx_car, tile_idx_trg)

            # if track_progress is not None and active_tile is not None:
            #     self.draw_track_progress(track_progress, active_tile)

            if self.attention is not None:
                self._draw_attention(state)

            if all_targets_pos_world_env0 is not None and all_egoagent_pos_world_env0 is not None and not draw_reduced:
                self.draw_all_targets(all_targets_pos_world_env0, all_egoagent_pos_world_env0)

            if False:
                self.display_ego_state(state)

            
            # ### PPC debug
            # self.add_point(ppc_lookahead_points[0, 2, 0].cpu().numpy()[None], 5, (0, 255, 0), -1)
            # self.add_point(ppc_lookahead_points[0, 3, 0].cpu().numpy()[None], 5, (255, 0, 0), -1)
            # wot_ado0 = wheels_on_track[0, 2].cpu().numpy()
            # wot_ado0_str = str(int(wot_ado0[0])) + str(int(wot_ado0[1])) + str(int(wot_ado0[2])) + str(int(wot_ado0[3]))
            # cv.putText(self.img, "Ado0=" + wot_ado0_str, (470, 140), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            # wot_ado1 = wheels_on_track[0, 3].cpu().numpy()
            # wot_ado1_str = str(int(wot_ado1[0])) + str(int(wot_ado1[1])) + str(int(wot_ado1[2])) + str(int(wot_ado1[3]))
            # cv.putText(self.img, "Ado1=" + wot_ado1_str, (470, 170), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

            self.draw_points(self.img)
            self.draw_lines(self.img)
            self.draw_string(self.img)
            if not draw_reduced:
              self.draw_marked_agents(self.img)


            show_val = values_ll is not None
            if show_val:
                self.img_val = self.track_canvas.copy()
                self._values_ll.append(values_ll[self.env_idx_render, 0].cpu().numpy())
                self._states_ll.append(state[self.env_idx_render, 0, 0:2].cpu().numpy())
                self._values_ll_min = np.minimum(self._values_ll_min, values_ll.min().cpu().numpy())
                self._values_ll_max = np.maximum(self._values_ll_max, values_ll.max().cpu().numpy())
                self.draw_value_predictions(np.stack(self._values_ll, axis=0), np.stack(self._states_ll, axis=0))

        
        #listen for keypressed events
        if not self._headless:
            key = self._render_interactive()
        else:
            key = self._render_tb(show_val=show_val)

        return key


    def draw_value_predictions(self, values, states):

        states = self.cords2px_np(states)
        scale =  int(80/(self.scale_x))
        for val, pos in zip(values, states):
            val = self.rgb_convert(value=val, minimum=self._values_ll_min, maximum=self._values_ll_max)
            self.img_val = cv.circle(self.img_val, (pos[0], pos[1]), scale, val, -1)

        cv.putText(self.img_val, "Env ID: " + str(self.env_idx_render), (15, 50), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img_val, "Min Val: " + str(np.round(self._values_ll_min, 3)), (15, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img_val, "Max Val: " + str(np.round(self._values_ll_max, 3)), (15, 110), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


    def clear_values(self):
        self._values_ll = []
        self._states_ll = []


    def display_ego_state(self, states):
        states = states[self.env_idx_render, 0].cpu().numpy()

        h_diffs = 30
        h_start = 120  # 100
        cv.putText(self.img, "Ego state:", (470, h_start), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        names = ['px', 'py', 'th']
        for idx, (name, state) in enumerate(zip(names, states[:3])):
            strval = "{:+.3f}".format(state)
            cv.putText(self.img, name + ': ' + strval, (470, h_start + (idx+1) * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


    def draw_all_targets(self, all_targets_pos_world_env0, all_egoagent_pos_world_env0):
        all_targets = torch.stack(all_targets_pos_world_env0, dim=0).cpu().numpy()
        all_targets = all_targets[~(all_targets==0).all(1)]
        #project into camera frame%
        all_targets = self.cords2px_np(all_targets)
        # unique_idx_target = np.unique(all_targets, axis=0, return_index=True)[1]
        # all_targets = all_targets[np.sort(unique_idx_target)]

        all_positions = torch.stack(all_egoagent_pos_world_env0, dim=0).cpu().numpy()
        all_positions = all_positions[~(all_positions==0).all(1)]
        #project into camera frame%
        all_positions = self.cords2px_np(all_positions)
        # unique_idx_position = np.unique(all_positions, axis=0, return_index=True)[1]
        # all_positions = all_positions[np.sort(unique_idx_position)]
        # all_positions = np.concatenate((all_positions, 0 * np.expand_dims(all_positions[0], axis=0)), axis=0)  # skip first entry (want end of LL ep)

        idx_offset = 0
        if len(all_targets) > 3:
          idx_offset = len(all_targets) - 3
          all_targets = all_targets[-3:]
          all_positions = all_positions[-3:]

        for idx_iter, (target, position) in enumerate(zip(all_targets, all_positions[0:])):
            idx = idx_iter + idx_offset
            scale =  int(80/(self.scale_x))
            self.img = cv.circle(self.img, (target[0], target[1]), scale, (int(self.colors[-1]),  0, int(self.colors[-1])), -1)
            cv.putText(self.img, str(idx), (target[0]+5, target[1]-10), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            
            self.img = cv.circle(self.img, (position[0], position[1]), scale, (0,  255, 0), -1)
            cv.putText(self.img, str(idx), (position[0]+5, position[1]-10), self.font, 0.5,  (0,  255, 0), 1, cv.LINE_AA)

            # if idx < len(all_targets)-1:
            self.img  = cv.line(self.img, target, position, color=(255, 0, 0), thickness=1)

    def draw_track_progress(self, track_progress, active_tile):
        track_progress = track_progress[0, 0].cpu().numpy()
        active_tile = active_tile[0, 0].cpu().numpy()

        strval = "{:+.2f}".format(track_progress)
        cv.putText(self.img, "Progr=" + strval, (470, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "nTile=" + str(active_tile), (470, 110), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    def get_digit(self, number, n):
        return number // 10**n % 10

    def draw_reset_cause(self, reset_cause):
        reset_cause = reset_cause[0].cpu().numpy().astype(int)

        for idx, rc in enumerate(reset_cause):
            cause = ""
            if rc > 0:
                cause += "ID" + str(idx) + ": "
                cause += "O" if self.get_digit(rc, 0) > 0 else ""
                cause += "P" if self.get_digit(rc, 1) > 0 else ""
                cause += "T" if self.get_digit(rc, 2) > 0 else ""
                break

        cv.putText(self.img, "Reset=" + cause, (470, 50), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    def draw_tile_idx(self, tile_idx_car, tile_idx_trg):
        tile_idx_car = int(tile_idx_car[0, 0].cpu().numpy())
        tile_idx_trg = int(tile_idx_trg[0, 0].cpu().numpy())

        cv.putText(self.img, "CarIdx=" + str(tile_idx_car), (470, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "TrgIdx=" + str(tile_idx_trg), (470, 110), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    def draw_max_vel(self, max_vels, use_ppc_vec):

        max_vels = max_vels[0, :, 0].cpu().numpy()
        use_ppcs = use_ppc_vec[0, :, 0].cpu().numpy()
        w_diffs = 45
        w_start = 120
        cv.putText(self.img, "Max vel:  [", (w_start, 50), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        for idx, (max_vel, use_ppc) in enumerate(zip(max_vels, use_ppcs)):
            strvel = "{:.2f}".format(max_vel)
            if idx==len(max_vels)-1:
                strvel += "]"
            color = (255, 0, 0) if use_ppc else (int(self.colors[-1]),  0, int(self.colors[-1]))
            cv.putText(self.img, strvel, (w_start + (idx+2) * w_diffs, 50), self.font, 0.5, color, 1, cv.LINE_AA)

    def draw_curr_vel(self, curr_vels):

        max_vels = curr_vels[0, 0].cpu().numpy()
        h_start = 420
        h_diffs = 30
        cv.putText(self.img, "vx = " + str(round(max_vels[0], 3)), (15, h_start + 0 * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "vy = " + str(round(max_vels[1], 3)), (15, h_start + 1 * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    def draw_ranks(self, ranks):

        ranks = ranks[0, :].cpu().numpy()
        w_diffs = 45
        w_start = 120
        cv.putText(self.img, "Ranks:    [", (w_start, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        for idx, rank in enumerate(ranks):
            strvel = "  " + str(rank)
            if idx==len(ranks)-1:
                strvel += "]"
            cv.putText(self.img, strvel, (w_start + (idx+2) * w_diffs, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


    def draw_rew_terms(self, rew_terms):
        h_diffs = 30
        h_start = 120  # 100
        cv.putText(self.img, "Rewards/dt:", (15, h_start), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        for idx, (k, v) in enumerate(rew_terms.items()):
            vi = v[0, 0].cpu().numpy()
            ki = k[:3].title()
            strval = "{:+.2f}".format(vi)
            cv.putText(self.img, ki, (15, h_start + (idx+1) * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            cv.putText(self.img, "=", (48, h_start + (idx+1) * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            cv.putText(self.img, strval, (66, h_start + (idx+1) * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


    def draw_wheel_on_track_indicators(self, wheels_on_track, time_off_track):

        wheels_on_track = wheels_on_track[0, 0].cpu().numpy()
        time_off_track = time_off_track[0, 0].cpu().numpy()

        # # Specify indicator dimensions
        # id_width = 26  #17
        # id_height = 23
        # id0_start = (214, 33)  # ccw: LF
        # id1_start = (248, 33)  # ccw: RF
        # id2_start = (248, 63)  # ccw: RB
        # id3_start = (214, 63)  # ccw: LB
        # id0_color = (255 * (1 - int(wheels_on_track[0])), 255 * int(wheels_on_track[0]), 0)
        # id1_color = (255 * (1 - int(wheels_on_track[1])), 255 * int(wheels_on_track[1]), 0)
        # id2_color = (255 * (1 - int(wheels_on_track[2])), 255 * int(wheels_on_track[2]), 0)
        # id3_color = (255 * (1 - int(wheels_on_track[3])), 255 * int(wheels_on_track[3]), 0)

        # # Visualize
        # cv.rectangle(self.img, id0_start, (id0_start[0] + id_width, id0_start[1] + id_height), color=id0_color, thickness=-1)
        # cv.rectangle(self.img, id1_start, (id1_start[0] + id_width, id1_start[1] + id_height), color=id1_color, thickness=-1)
        # cv.rectangle(self.img, id2_start, (id2_start[0] + id_width, id2_start[1] + id_height), color=id2_color, thickness=-1)
        # cv.rectangle(self.img, id3_start, (id3_start[0] + id_width, id3_start[1] + id_height), color=id3_color, thickness=-1)
        
        # # Add description
        # cv.putText(self.img, "Wheels  FL  FR", (150, 50), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        # cv.putText(self.img, "on/off  BL  BR", (150, 80), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


        # Specify indicator dimensions
        x_rel0 = 510

        id_width = 26  #17
        id_height = 30
        id0_start = (x_rel0, self.height - 85)  # ccw: LF
        id1_start = (x_rel0 + 34, self.height - 85)  # ccw: RF
        id2_start = (x_rel0 + 34, self.height - 50)  # ccw: RB
        id3_start = (x_rel0, self.height - 50)  # ccw: LB
        id0_color = (255 * (1 - int(wheels_on_track[0])), 255 * int(wheels_on_track[0]), 0)
        id1_color = (255 * (1 - int(wheels_on_track[1])), 255 * int(wheels_on_track[1]), 0)
        id2_color = (255 * (1 - int(wheels_on_track[2])), 255 * int(wheels_on_track[2]), 0)
        id3_color = (255 * (1 - int(wheels_on_track[3])), 255 * int(wheels_on_track[3]), 0)

        # Visualize
        cv.rectangle(self.img, id0_start, (id0_start[0] + id_width, id0_start[1] - id_height), color=id0_color, thickness=-1)
        cv.rectangle(self.img, id1_start, (id1_start[0] + id_width, id1_start[1] - id_height), color=id1_color, thickness=-1)
        cv.rectangle(self.img, id2_start, (id2_start[0] + id_width, id2_start[1] - id_height), color=id2_color, thickness=-1)
        cv.rectangle(self.img, id3_start, (id3_start[0] + id_width, id3_start[1] - id_height), color=id3_color, thickness=-1)
        
        # # Add description
        cv.putText(self.img, "T off=" + str(int(time_off_track)), (x_rel0, self.height - 20), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "FL  FR", (x_rel0+3, id0_start[1] - 10), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "BL  BR", (x_rel0+3, id2_start[1] - 10), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)




    # def draw_action_distributions(self, hl_action_probs, ll_action_mean, ll_action_std):

    #     hl_action_probs = hl_action_probs[0, 0].cpu().numpy()
    #     ll_action_mean = ll_action_mean[0, 0].cpu().numpy()
    #     ll_action_std = ll_action_std[0, 0].cpu().numpy()
    #     # xlabel = np.linspace(1, hl_action_probs.shape[-1], num=hl_action_probs.shape[-1])

    #     bar_width = 10
    #     bar_scale = 50
    #     bar_bottom = self.height - 50
    #     separation = 3

    #     # ### High-level controller ###
    #     # Action 1
    #     x_start = 35  # 125
    #     for idx, action_prob in enumerate(hl_action_probs[0]):
    #         x_bar_start = x_start + idx * (bar_width + separation)
    #         x_bar_end = x_bar_start + bar_width
    #         cv.rectangle(self.img, (x_bar_start, bar_bottom), (x_bar_end, bar_bottom - int(action_prob * 50)), color=(255, 0, 0), thickness=-1)
    #     cv.putText(self.img, "Prob Gx", (x_start, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    #     # Action 2
    #     x_start = x_bar_end + 30
    #     for idx, action_prob in enumerate(hl_action_probs[1]):
    #         x_bar_start = x_start + idx * (bar_width + separation)
    #         x_bar_end = x_bar_start + bar_width
    #         cv.rectangle(self.img, (x_bar_start, bar_bottom), (x_bar_end, bar_bottom - int(action_prob * 50)), color=(255, 0, 0), thickness=-1)
    #     cv.putText(self.img, "Prob Gy", (x_start, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    #     # Action 3
    #     x_start = x_bar_end + 30
    #     for idx, action_prob in enumerate(hl_action_probs[2]):
    #         x_bar_start = x_start + idx * (bar_width + separation)
    #         x_bar_end = x_bar_start + bar_width
    #         cv.rectangle(self.img, (x_bar_start, bar_bottom), (x_bar_end, bar_bottom - int(action_prob * 50)), color=(255, 0, 0), thickness=-1)
    #     cv.putText(self.img, "Prob Ux", (x_start, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    #     # Action 4
    #     x_start = x_bar_end + 30
    #     for idx, action_prob in enumerate(hl_action_probs[3]):
    #         x_bar_start = x_start + idx * (bar_width + separation)
    #         x_bar_end = x_bar_start + bar_width
    #         cv.rectangle(self.img, (x_bar_start, bar_bottom), (x_bar_end, bar_bottom - int(action_prob * 50)), color=(255, 0, 0), thickness=-1)
    #     cv.putText(self.img, "Prob Uy", (x_start, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

    #     # ### Low-level controller ###
    #     # x = steering
    #     # y = throttle

    #     # Bounds
    #     x_act_min, x_act_max = -0.35, +0.35
    #     y_act_min, y_act_max = -0.30, +1.30

    #     def scale_pos_to_img(v, v_min, v_max, scale):
    #         return int(scale * (v - v_min) / (v_max - v_min))

    #     def scale_std_to_img(v, v_min, v_max, scale):
    #         return int(scale * (v) / (v_max - v_min))

    #     ellipse_scale = 70
    #     ellipse_x_pos = scale_pos_to_img(ll_action_mean[0], x_act_min, x_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_mean[0])
    #     ellipse_y_pos = scale_pos_to_img(ll_action_mean[1], y_act_min, y_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_mean[1])
    #     ellipse_x_len = scale_std_to_img(ll_action_std[0], x_act_min, x_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_std[0])
    #     ellipse_y_len = scale_std_to_img(ll_action_std[1], y_act_min, y_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_std[1])

    #     ellipse_x_pos = ellipse_x_pos + x_bar_end + 100
    #     ellipse_y_pos = bar_bottom - ellipse_y_pos


    #     # Relative and scaled
    #     x_act_rel_min = scale_pos_to_img(x_act_min, x_act_min, x_act_max, ellipse_scale)
    #     x_act_rel_max = scale_pos_to_img(x_act_max, x_act_min, x_act_max, ellipse_scale)

    #     y_act_rel_min = scale_pos_to_img(y_act_min, y_act_min, y_act_max, ellipse_scale)
    #     y_act_rel_max = scale_pos_to_img(y_act_max, y_act_min, y_act_max, ellipse_scale)

    #     rect_pos_start = (x_act_rel_min + x_bar_end + 100, bar_bottom - y_act_rel_min)
    #     rect_pos_end = (x_act_rel_max + x_bar_end + 100, bar_bottom - y_act_rel_max)

    #     cv.rectangle(self.img, rect_pos_start, rect_pos_end, color=(0, 0, 0), thickness=1)

    #     # Zero lines
    #     x_act_rel_zero = scale_pos_to_img(0.0, x_act_min, x_act_max, ellipse_scale)
    #     y_act_rel_zero = scale_pos_to_img(0.0, y_act_min, y_act_max, ellipse_scale)
        
    #     x_zero_start = (x_act_rel_zero + x_bar_end + 100, bar_bottom - y_act_rel_min)
    #     x_zero_end = (x_act_rel_zero + x_bar_end + 100, bar_bottom - y_act_rel_max)

    #     y_zero_start = (x_act_rel_min + x_bar_end + 100, bar_bottom - y_act_rel_zero)
    #     y_zero_end = (x_act_rel_max + x_bar_end + 100, bar_bottom - y_act_rel_zero)

    #     cv.line(self.img, x_zero_start, x_zero_end, color=(0, 0, 255), thickness=1)
    #     cv.line(self.img, y_zero_start, y_zero_end, color=(0, 0, 255), thickness=1)

    #     cv.ellipse(self.img, (ellipse_x_pos, ellipse_y_pos), (ellipse_x_len, ellipse_y_len), float(0.0), 0.0, 360.0, (255, 0, 0), 2)
    #     cv.circle(self.img, (ellipse_x_pos, ellipse_y_pos), 1, (255,0,0))

    #     # # Cutoff
    #     # cut_scale = 1.3

    #     # x_act_rel_min_cut = scale_to_img(cut_scale * x_act_min, x_act_min, x_act_max, ellipse_scale)
    #     # x_act_rel_max_cut = scale_to_img(cut_scale * x_act_max, x_act_min, x_act_max, ellipse_scale)

    #     # y_act_rel_min_cut = scale_to_img(cut_scale * y_act_min, y_act_min, y_act_max, ellipse_scale)
    #     # y_act_rel_max_cut = scale_to_img(cut_scale * y_act_max, y_act_min, y_act_max, ellipse_scale)

    #     # rect_pos_start_cut = (x_act_rel_min_cut + x_bar_end + 100, bar_bottom + y_act_rel_min_cut)
    #     # rect_pos_end_cut = (x_act_rel_max_cut + x_bar_end + 100, bar_bottom - y_act_rel_max_cut)

    #     # cv.rectangle(self.img, rect_pos_start_cut, rect_pos_end_cut, color=(255, 0, 0), thickness=1)

    #     cv.putText(self.img, "Steer vs. Throttle", (x_act_rel_min + x_bar_end + 65, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


    def draw_team_stats_debug(self, states, actions):
        states = states[0, :self.teamsize].cpu().numpy()
        actions = actions[0, :self.teamsize].cpu().numpy()

        dstates = states - np.expand_dims(states[0], axis=0)
        dactions = actions - np.expand_dims(actions[0], axis=0)

        ds_stat = np.mean(dstates**2, axis=-1)
        da_stat = np.mean(dactions**2, axis=-1)

        h_diffs = 30
        h_start = 400  # 380
        cv.putText(self.img, "Team debug:", (15, h_start), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

        ds_strval = "[" 
        for idx_s, ds_val in enumerate(ds_stat):
            ds_strval += "{:+.2f}".format(ds_val)
            if idx_s < len(ds_stat)-1:
                ds_strval += ","
        ds_strval += "]"
        cv.putText(self.img, "ds=" + ds_strval, (15, h_start + 1 * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

        da_strval = "[" 
        for idx_a, da_val in enumerate(da_stat):
            da_strval += "{:+.2f}".format(da_val)
            if idx_a < len(da_stat)-1:
                da_strval += ","
        da_strval += "]"
        cv.putText(self.img, "da=" + da_strval, (15, h_start + 2 * h_diffs), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


    def draw_action_distributions_new(self, hl_action_probs, ll_action_mean, ll_action_std, target_dist, actions, last_actions):

        hl_action_probs = hl_action_probs[0, 0].cpu().numpy()
        ll_action_mean = ll_action_mean[0, 0].cpu().numpy()
        ll_action_std = ll_action_std[0, 0].cpu().numpy()

        target_dist = target_dist[0, 0, :].cpu().numpy()
        actions = actions[0, 0, :].cpu().numpy()
        last_actions = last_actions[0, 0, :].cpu().numpy()
        # xlabel = np.linspace(1, hl_action_probs.shape[-1], num=hl_action_probs.shape[-1])

        bar_scale = 30
        bar_bottom = self.height - 50

        # width budget = 70
        # per bar = int(width budget / len(hl_action_probs[0]))
        # separation = int(max(1.0, per bar * 0.2))
        # bar width = per bar - separation

        width_budget = 65
        bar_budget = int(width_budget / len(hl_action_probs[0]))
        separation = int(max(1.0, bar_budget * 0.3))  # 3
        bar_width = bar_budget - separation  # 10

        # ### High-level controller ###
        # Action 1
        x_start = 230  # 150  # 35  # 125
        y_start = bar_bottom - 35
        for idx, action_prob in enumerate(hl_action_probs[0]):
            x_bar_start = x_start + idx * (bar_width + separation)
            x_bar_end = x_bar_start + bar_width
            cv.rectangle(self.img, (x_bar_start, y_start), (x_bar_end, y_start - int(action_prob * bar_scale)), color=(255, 0, 0), thickness=-1)
        # cv.putText(self.img, "Prob Gx", (x_start, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

        # Action 2
        x_start = x_start
        y_start = bar_bottom
        for idx, action_prob in enumerate(hl_action_probs[1]):
            x_bar_start = x_start + idx * (bar_width + separation)
            x_bar_end = x_bar_start + bar_width
            cv.rectangle(self.img, (x_bar_start, y_start), (x_bar_end, y_start - int(action_prob * bar_scale)), color=(255, 0, 0), thickness=-1)
        cv.putText(self.img, "Goal Pos", (x_start-5, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

        if len(hl_action_probs) > 2:
            # Action 3
            x_start = x_bar_end + 30
            y_start = bar_bottom - 35
            for idx, action_prob in enumerate(hl_action_probs[2]):
                x_bar_start = x_start + idx * (bar_width + separation)
                x_bar_end = x_bar_start + bar_width
                cv.rectangle(self.img, (x_bar_start, y_start), (x_bar_end, y_start - int(action_prob * bar_scale)), color=(255, 0, 0), thickness=-1)
            # cv.putText(self.img, "Prob Ux", (x_start, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

            # Action 4
            x_start = x_start
            y_start = bar_bottom
            for idx, action_prob in enumerate(hl_action_probs[3]):
                x_bar_start = x_start + idx * (bar_width + separation)
                x_bar_end = x_bar_start + bar_width
                cv.rectangle(self.img, (x_bar_start, y_start), (x_bar_end, y_start - int(action_prob * bar_scale)), color=(255, 0, 0), thickness=-1)
            cv.putText(self.img, "Goal Std", (x_start-5, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


        x_ref0 = 405
        cv.putText(self.img, "x=" + "{:.2f}".format(target_dist[0]), (x_ref0, bar_bottom - 35), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "y=" + "{:.2f}".format(target_dist[1]), (x_ref0, bar_bottom), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "HL dist", (x_ref0, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


        # ### Low-level controller ###
        # x = steering
        # y = throttle

        # Bounds
        # x_act_min, x_act_max = -1.0, +1.0  # -0.5, +0.5  # -0.35, +0.35
        # y_act_min, y_act_max = -1.0, +1.0  # -0.3, +0.5  # +1.30
        x_act_min = self.act_ll_off[0] - self.act_ll_scl[0]
        x_act_max = self.act_ll_off[0] + self.act_ll_scl[0]
        y_act_min = self.act_ll_off[1] - self.act_ll_scl[1]
        y_act_max = self.act_ll_off[1] + self.act_ll_scl[1]
        x_off_bar = 50
        x_ref0 = 25  # x_bar_end + x_off_bar

        def scale_pos_to_img(v, v_min, v_max, scale):
            return int(scale * (v - v_min) / (v_max - v_min))

        def scale_std_to_img(v, v_min, v_max, scale):
            return int(scale * (v) / (v_max - v_min))

        # ll_action_mean = (ll_action_mean - self.act_ll_off) / self.act_ll_scl
        # ll_action_std = ll_action_std / self.act_ll_scl

        ellipse_scale = 70
        ellipse_x_pos = scale_pos_to_img(ll_action_mean[0], x_act_min, x_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_mean[0])
        ellipse_y_pos = scale_pos_to_img(ll_action_mean[1], y_act_min, y_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_mean[1])
        ellipse_x_len = scale_std_to_img(ll_action_std[0], x_act_min, x_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_std[0])
        ellipse_y_len = scale_std_to_img(ll_action_std[1], y_act_min, y_act_max, ellipse_scale)  # int(ellipse_scale * ll_action_std[1])

        ellipse_x_pos = ellipse_x_pos + x_ref0
        ellipse_y_pos = bar_bottom - ellipse_y_pos

        last_actions_x_pos = scale_pos_to_img(last_actions[0], x_act_min, x_act_max, ellipse_scale)
        last_actions_y_pos = scale_pos_to_img(last_actions[1], y_act_min, y_act_max, ellipse_scale)
        last_actions_x_pos = last_actions_x_pos + x_ref0
        last_actions_y_pos = bar_bottom - last_actions_y_pos


        # Relative and scaled
        x_act_rel_min = scale_pos_to_img(x_act_min, x_act_min, x_act_max, ellipse_scale)
        x_act_rel_max = scale_pos_to_img(x_act_max, x_act_min, x_act_max, ellipse_scale)

        y_act_rel_min = scale_pos_to_img(y_act_min, y_act_min, y_act_max, ellipse_scale)
        y_act_rel_max = scale_pos_to_img(y_act_max, y_act_min, y_act_max, ellipse_scale)

        rect_pos_start = (x_act_rel_min + x_ref0, bar_bottom - y_act_rel_min)
        rect_pos_end = (x_act_rel_max + x_ref0, bar_bottom - y_act_rel_max)

        cv.rectangle(self.img, rect_pos_start, rect_pos_end, color=(0, 0, 0), thickness=1)

        # Zero lines
        x_act_rel_zero = scale_pos_to_img(0.0, x_act_min, x_act_max, ellipse_scale)
        y_act_rel_zero = scale_pos_to_img(0.0, y_act_min, y_act_max, ellipse_scale)
        
        x_zero_start = (x_act_rel_zero + x_ref0, bar_bottom - y_act_rel_min)
        x_zero_end = (x_act_rel_zero + x_ref0, bar_bottom - y_act_rel_max)

        y_zero_start = (x_act_rel_min + x_ref0, bar_bottom - y_act_rel_zero)
        y_zero_end = (x_act_rel_max + x_ref0, bar_bottom - y_act_rel_zero)

        cv.line(self.img, x_zero_start, x_zero_end, color=(0, 0, 255), thickness=1)
        cv.line(self.img, y_zero_start, y_zero_end, color=(0, 0, 255), thickness=1)

        cv.ellipse(self.img, (ellipse_x_pos, ellipse_y_pos), (ellipse_x_len, ellipse_y_len), float(0.0), 0.0, 360.0, (255, 0, 0), 2)
        cv.circle(self.img, (ellipse_x_pos, ellipse_y_pos), 1, (255,0,0), -1)
        cv.circle(self.img, (last_actions_x_pos, last_actions_y_pos), 2, (0,0,255), -1)



        x_sep1_start = (10,bar_bottom - 80)
        x_sep1_end = (590, bar_bottom - 80)
        cv.line(self.img, x_sep1_start, x_sep1_end, color=(0, 0, 0), thickness=2)

        x_sep2_start = (10,bar_bottom + 45)
        x_sep2_end = (590, bar_bottom + 45)
        cv.line(self.img, x_sep2_start, x_sep2_end, color=(0, 0, 0), thickness=2)

        y_sep1_start = (210, bar_bottom + 45)
        y_sep1_end = (210, bar_bottom - 80)
        cv.line(self.img, y_sep1_start, y_sep1_end, color=(0, 0, 0), thickness=2)

        y_sep2_start = (490, bar_bottom + 45)
        y_sep2_end = (490, bar_bottom - 80)
        cv.line(self.img, y_sep2_start, y_sep2_end, color=(0, 0, 0), thickness=2)

        y_sep3_start = (10, bar_bottom + 45)
        y_sep3_end = (10, bar_bottom - 80)
        cv.line(self.img, y_sep3_start, y_sep3_end, color=(0, 0, 0), thickness=2)

        y_sep4_start = (590, bar_bottom + 45)
        y_sep4_end = (590, bar_bottom - 80)
        cv.line(self.img, y_sep4_start, y_sep4_end, color=(0, 0, 0), thickness=2)

        # # Cutoff
        # cut_scale = 1.3

        # x_act_rel_min_cut = scale_to_img(cut_scale * x_act_min, x_act_min, x_act_max, ellipse_scale)
        # x_act_rel_max_cut = scale_to_img(cut_scale * x_act_max, x_act_min, x_act_max, ellipse_scale)

        # y_act_rel_min_cut = scale_to_img(cut_scale * y_act_min, y_act_min, y_act_max, ellipse_scale)
        # y_act_rel_max_cut = scale_to_img(cut_scale * y_act_max, y_act_min, y_act_max, ellipse_scale)

        # rect_pos_start_cut = (x_act_rel_min_cut + x_bar_end + 100, bar_bottom + y_act_rel_min_cut)
        # rect_pos_end_cut = (x_act_rel_max_cut + x_bar_end + 100, bar_bottom - y_act_rel_max_cut)

        # cv.rectangle(self.img, rect_pos_start_cut, rect_pos_end_cut, color=(255, 0, 0), thickness=1)

        cv.putText(self.img, "St vs. Th", (x_act_rel_min + x_ref0, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)

        x_ref0 = 125
        cv.putText(self.img, "s=" + "{:.2f}".format(actions[0]), (x_ref0, bar_bottom - 35), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "t=" + "{:.2f}".format(actions[1]), (x_ref0, bar_bottom), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
        cv.putText(self.img, "LL act", (x_ref0, bar_bottom + 30), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)


    def draw_svo_distribution(self, hl_svo_probs):

        hl_svo_probs = hl_svo_probs[0, 0, 0].cpu().numpy()

        bar_scale = 30
        bar_bottom = 170

        # width budget = 70
        # per bar = int(width budget / len(hl_action_probs[0]))
        # separation = int(max(1.0, per bar * 0.2))
        # bar width = per bar - separation

        width_budget = 65
        bar_budget = int(width_budget / len(hl_svo_probs))
        separation = int(max(1.0, bar_budget * 0.3))  # 3
        bar_width = bar_budget - separation  # 10

        # ### High-level controller ###
        # Action 1
        x_start = 490  # 150  # 35  # 125
        y_start = bar_bottom
        for idx, action_prob in enumerate(hl_svo_probs):
            x_bar_start = x_start + idx * (bar_width + separation)
            x_bar_end = x_bar_start + bar_width
            cv.rectangle(self.img, (x_bar_start, y_start), (x_bar_end, y_start - int(action_prob * bar_scale)), color=(255, 0, 0), thickness=-1)
        cv.putText(self.img, "SVO", (x_start+15, bar_bottom+20), self.font, 0.5, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)



    def _render_interactive(self):
        cv.imshow("dmaracing", self.img)
        key = cv.waitKey(1)
        #print(key)
        if key == 118: #toggle render on v
            if self.do_render:
                self.do_render = False
            else: 
                self.do_render = True
            print('[VIZ] render toggled to ', self.do_render)
        if key == 113 or key == 27: #quit on escape or q
            sys.exit()
        if self.do_render:
            if key == 114:
                self.env_idx_render = np.mod(self.env_idx_render+1, self.num_envs)
                self.draw_track()
                #print('[VIZ] env toggled to ', self.env_idx_render)
            if key == 116:
                self.env_idx_render = np.mod(self.env_idx_render-1, self.num_envs)
                self.draw_track()
                #print('[VIZ] env toggled to ', self.env_idx_render)
            if key == 119:
                self.y_offset -= 40
                self.draw_track()
            if key == 115:
                self.y_offset += 40
                self.draw_track()
            if key == 97:
                self.x_offset += 40
                self.draw_track()
            if key == 100:
                self.x_offset -= 40
                self.draw_track()
            if key == 46:
                self.scale_x *= 1.2
                self.scale_y *= 1.2
                self.draw_track()
            if key == 44:     
                self.scale_x /= 1.2
                self.scale_y /= 1.2
                self.draw_track()
        return key

    
    def draw_multiagent_rep(self, state):
            transl = state[self.env_idx_render, :, 0:2]
            theta = state[self.env_idx_render, :, 2]
            delta = state[self.env_idx_render, :, 3]
            self.R[0, 0, :] = torch.cos(theta)
            self.R[0, 1, :] = -torch.sin(theta)
            self.R[1, 0, :] = torch.sin(theta)
            self.R[1, 1, :] = torch.cos(theta)

            car_box_rot = torch.einsum ('ijl, jkl -> ikl', self.R, self.car_box_m)
            car_box_world = torch.transpose(car_box_rot + transl.T.unsqueeze(1).repeat(1,4,1), 0,1)
            
            self.R[0, 0, :] = torch.cos(theta+delta)
            self.R[0, 1, :] = -torch.sin(theta+delta)
            self.R[1, 0, :] = torch.sin(theta+delta)
            self.R[1, 1, :] = torch.cos(theta+delta)
            car_heading_rot = torch.einsum ('ijl, jkl -> ikl', self.R, self.car_heading_m)
            car_heading_world = torch.transpose(car_heading_rot + transl.T.unsqueeze(1).repeat(1,2,1), 0,1)

            px_car_box_world = self.cords2px(car_box_world)
            px_car_heading_world = self.cords2px(car_heading_world)
            px_car_cm_world = self.cords2px_np(transl[:,...].cpu().numpy())

            for idx in range(self.num_cars):
                px_x_number = (self.width/self.scale_x*transl[idx, 0] + self.width/2.0).cpu().numpy().astype(np.int32).item()
                px_y_number = (-self.height/self.scale_y*transl[idx, 1] + self.height/2.0).cpu().numpy().astype(np.int32).item()
                px_pts_car = px_car_box_world[..., idx].reshape(-1,1,2)
                px_pts_heading = px_car_heading_world[..., idx].reshape(-1,1,2)
                cv.polylines(self.img, [px_pts_car], isClosed = True, color = (255-int(self.colors[idx]),0,int(self.colors[idx])), thickness = self.thickness)
                cv.fillPoly(self.car_img, [px_pts_car], color = (255-int(self.colors[idx]),0,int(self.colors[idx]), 0.9))
                cv.polylines(self.img, [px_pts_heading], isClosed = True, color = (0, 0, 255), thickness = 2)
                if self.drag_reduced[self.env_idx_render, idx]:
                    cv.putText(self.img, str(idx)+' dr', (px_x_number+ self.x_offset, px_y_number + self.y_offset-10), self.font, 0.5, (int(self.colors[idx]),0,int(self.colors[idx])), 1, cv.LINE_AA)   
                else:
                    cv.putText(self.img, str(idx), (px_x_number+ self.x_offset, px_y_number + self.y_offset-10), self.font, 0.5, (int(self.colors[idx]),0,int(self.colors[idx])), 1, cv.LINE_AA)   
            self.img = cv.addWeighted(self.car_img, 0.5, self.img, 0.5, 0)
            

    def draw_singleagent_rep(self, state):
            transl = state[:, 0, 0:2]
            theta = state[:, 0, 2]
            delta = state[:, 0, 3]
            self.R[0, 0, :] = torch.cos(theta)
            self.R[0, 1, :] = -torch.sin(theta)
            self.R[1, 0, :] = torch.sin(theta)
            self.R[1, 1, :] = torch.cos(theta)

            car_box_rot = torch.einsum ('ijl, jkl -> ikl', self.R, self.car_box_m)
            car_box_world = torch.transpose(car_box_rot + transl.T.unsqueeze(1).repeat(1,4,1), 0,1)
            
            self.R[0, 0, :] = torch.cos(theta+delta)
            self.R[0, 1, :] = -torch.sin(theta+delta)
            self.R[1, 0, :] = torch.sin(theta+delta)
            self.R[1, 1, :] = torch.cos(theta+delta)
            car_heading_rot = torch.einsum ('ijl, jkl -> ikl', self.R, self.car_heading_m)
            car_heading_world = torch.transpose(car_heading_rot + transl.T.unsqueeze(1).repeat(1,2,1), 0,1)

            px_car_box_world = self.cords2px(car_box_world)
            px_car_heading_world = self.cords2px(car_heading_world)

            for idx in range(self.num_cars):
                px_x_number = (self.width/self.scale_x*transl[idx, 0] + self.width/2.0).cpu().numpy().astype(np.int32).item()
                px_y_number = (-self.height/self.scale_y*transl[idx, 1] + self.height/2.0).cpu().numpy().astype(np.int32).item()
                px_pts_car = px_car_box_world[..., idx].reshape(-1,1,2)
                px_pts_heading = px_car_heading_world[..., idx].reshape(-1,1,2)
                cv.polylines(self.img, [px_pts_car], isClosed = True, color = (int(self.colors[idx]),0,int(self.colors[idx])), thickness = self.thickness)
                cv.fillPoly(self.car_img, [px_pts_car], color = (int(255-self.colors[idx]),0,int(self.colors[idx]), 0.9))
                cv.polylines(self.img, [px_pts_heading], isClosed = True, color = (int(self.colors[idx]),0,int(self.colors[idx])), thickness = self.thickness)
                cv.putText(self.img, str(idx), (px_x_number+ self.x_offset, px_y_number + self.y_offset -10), self.font, 0.5, (int(self.colors[idx]),0,int(self.colors[idx])), 1, cv.LINE_AA)
                
            self.img = cv.addWeighted(self.car_img, 0.5, self.img, 0.5, 0)

    def add_slip_markers(self,):
        idx_ag, idx_wheel = torch.where(self.slip[self.env_idx_render, : , :])
        locations = self.wheel_locs[self.env_idx_render,idx_ag, idx_wheel, : ]
        if len(locations):
            slip_locations = locations.cpu().numpy()
            #print(slip_locations)
            self.slip_markers.append(slip_locations)
            if len(self.slip_markers) > 10:  # 140:
                del self.slip_markers[0]

    def add_lines(self, endpoints, color = (0,0,0), thickness = 1):
        #endpoints shape should be (nlines*2points, 2 corrds)
        self.lines.append([endpoints, color, thickness])

    def clear_lines(self):
        self.lines = []

    def draw_lines(self, img):
        for endpoints, color, thickness in self.lines:
            coords = self.cords2px_np(endpoints)
            img  = cv.polylines(img, [coords.reshape(-1,1,2)], isClosed = False, color = color, thickness = thickness)
                

    def cords2px(self, pts):
        pts = pts.cpu().numpy()
        pts[:, 0, :] = self.width/self.scale_x*pts[:, 0, :] + self.width/2.0 + self.x_offset
        pts[:, 1, :] = -self.height/self.scale_y*pts[:, 1, :] + self.height/2.0 + self.y_offset
        return pts.astype(np.int32)

    def cords2px_np(self, pts):
        pts[:, 0] = self.width/self.scale_x*pts[:, 0] + self.width/2.0 + self.x_offset
        pts[:, 1] = -self.height/self.scale_y*pts[:, 1] + self.height/2.0 + self.y_offset
        return pts.astype(np.int32)

    def length2px_np(self, pts):
        pts[:, 0] = self.width/self.scale_x*pts[:, 0]
        pts[:, 1] = self.height/self.scale_y*pts[:, 1]
        return pts.astype(np.int32)

    def cords2px_np_copy(self, pts):
        a = pts.copy()
        a[:, 0] = self.width/self.scale_x*a[:, 0] + self.width/2.0 + self.x_offset
        a[:, 1] = -self.height/self.scale_y*a[:, 1] + self.height/2.0 + self.y_offset
        return a.astype(np.int32)

    def add_point(self, cords, radius, color, thickness):
        cd = cords.copy()
        cd = self.cords2px_np(cd)
        self.points.append([cd, radius, color, thickness])

    def clear_markers(self,):
        self.points = []   

    def draw_points(self, img):
        for group in self.points: 
            for idx in range(len(group[0])):
                img = cv.circle(img, (group[0][idx, 0], group[0][idx, 1]), group[1], group[2], group[3])

    def reset_slip_markers(self,):
        self.slip_markers = []

    def draw_slip_markers(self):
        #project into camera frame
        current_markers = []
        for markergroup in self.slip_markers:
            current_markers.append(self.cords2px_np_copy(markergroup))
        scale =  int(50/(self.scale_x))
        for group in current_markers:
            for loc in group.tolist():
                self.img = cv.circle(self.img, (loc[0], loc[1]), scale, (30,30,30), -1)

    def add_string(self, string):
        self.msg.append(string) 

    def clear_string(self,):
        self.msg = []

    def draw_string(self, img):
        if len(self.msg):
            idx = 0
            for msg in self.msg:
                cv.putText(img, msg, (10, 100 + 40*idx), self.font, 1, (0, 0, 0), 1, cv.LINE_AA)
                idx+=1

    def mark_env(self, idx):
        self.marked_env = idx

    def draw_marked_agents(self, img):
        if self.marked_env is not None:
            pos = self.state[self.env_idx_render, self.marked_env, 0:2].view(1,-1).cpu().numpy()
            px = self.cords2px_np(pos)
            cv.circle(img, (px[0,0],px[0,1]), 30, (250,150,0))
    
    def draw_track(self,):
        self.track_canvas = draw_track(self.track_canvas,
                                       self.track_centerlines[self.active_track_ids[self.env_idx_render]].copy(),
                                       self.track_poly_verts[self.active_track_ids[self.env_idx_render]].copy(),
                                       self.track_border_poly_verts[self.active_track_ids[self.env_idx_render]].copy(),
                                       self.track_border_poly_cols[self.active_track_ids[self.env_idx_render]].copy(),
                                       self.track_tile_counts[self.active_track_ids[self.env_idx_render]].copy(),
                                       self.cords2px_np, 
                                       self.cfg['track']['draw_centerline'])
    def draw_track_reset(self,):
        self.reset_slip_markers()
        self.draw_track()

    def save_frame(self, path):
        cv.imwrite(path, self.img)
    
    def draw_lookahead_markers(self, lookahead, bounds):
        points = lookahead[self.env_idx_render, 0, :, :].cpu().numpy()

        #project into camera frame%
        points = self.cords2px_np(points)
        scale =  int(50/(self.scale_x))
        for point in points:
            self.img = cv.circle(self.img, (point[0], point[1]), scale, (255, 0,0), -1)

        rbounds = bounds[self.env_idx_render, 0, :, [0, 1]].cpu().numpy()
        #project into camera frame%
        rbounds = self.cords2px_np(rbounds)
        for rbound in rbounds:
            self.img = cv.circle(self.img, (rbound[0], rbound[1]), scale, (0, 0, 0), -1)

        lbounds = bounds[self.env_idx_render, 0, :, [2, 3]].cpu().numpy()
        #project into camera frame%
        lbounds = self.cords2px_np(lbounds)
        for lbound in lbounds:
            self.img = cv.circle(self.img, (lbound[0], lbound[1]), scale, (0, 0, 0), -1)

    def draw_target_marker(self, targets, targets_rew01, targets_angle):
        target = targets[self.env_idx_render, 0, :].cpu().numpy()
        target_rew01 = targets_rew01[self.env_idx_render, 0, :].cpu().numpy()
        target_angle = targets_angle[self.env_idx_render, 0].cpu().numpy()

        #project into camera frame%
        target = self.cords2px_np(target[None, :])[0]
        scale =  int(50/(self.scale_x))
        target_rew01 = self.length2px_np(target_rew01[None, :])[0]
        target_angle = 360.0 - 180.0 / np.pi * target_angle

        self.img = cv.circle(self.img, (target[0], target[1]), scale, (0, 0, 255), -1)
        self.img = cv.ellipse(self.img, (target[0], target[1]), (target_rew01[0], target_rew01[1]), float(target_angle), 0.0, 360.0, (0, 0, 255), 1)

    def update_attention(self, attention=None):
        if attention is not None:
            self.attention = attention.detach().cpu().numpy().mean(axis=0)
        else:
            self.attention = None

    def _draw_attention(self, states):
        state = states[self.env_idx_render, :, 0:2].cpu().numpy()
        for ado_id in range(self.num_agents-1):
            endpoints = np.array([state[0], state[ado_id+1]])
            if self.attention[ado_id].item() >= 0.01:
                color = self.rgb_convert(self.attention[ado_id])
                self.add_lines(endpoints=endpoints.squeeze(), color = color, thickness= int(5*self.attention[ado_id].item()))

    def rgb_convert(self, value, minimum=0.0, maximum=1.0):
        ratio = 2 * (value-minimum) / (maximum - minimum)
        b = int(max(0*ratio, 255*(1 - ratio)))
        r = int(max(0, 255*(ratio - 1)))
        g = 255 - b - r
        return r, g, b
