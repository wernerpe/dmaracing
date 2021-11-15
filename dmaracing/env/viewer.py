import cv2 as cv
import torch
import numpy as np
import sys

from dmaracing.utils.trackgen import draw_track


class Viewer:
    def __init__(self, cfg, track):
        self.device = 'cuda:0'
        self.cfg = cfg

        #load cfg
        self.width = self.cfg['viewer']['width']
        self.height = self.cfg['viewer']['height']
        self.scale_x = self.cfg['viewer']['scale']
        self.scale_y = self.height/self.width*self.scale_x
        self.thickness = self.cfg['viewer']['linethickness']
        self.draw_multiagent = self.cfg['viewer']['multiagent']
        self.max_agents = self.cfg['viewer']['maxAgents']
        self.num_agents = self.cfg['sim']['numAgents']
        self.num_envs = self.cfg['sim']['numEnv']
        
        if self.draw_multiagent:
            self.num_cars = self.num_agents
        else:
            self.num_cars = min(self.max_agents, self.num_envs)
        
        #bounding box of car in model frame
        w = self.cfg['model']['W']
        lf = self.cfg['model']['lf']  
        lr = self.cfg['model']['lr']
        self.car_box_m = torch.tensor([[lf, -w/2],[lf, w/2],[-lr, w/2], [-lr, -w/2]], device = self.device)
        self.car_box_m = self.car_box_m.unsqueeze(2).repeat(1, 1, self.num_cars)
        self.car_box_m = torch.transpose(self.car_box_m, 0,1) 
        self.car_heading_m = torch.tensor([[0, 0],[lf, 0]], device = self.device)
        self.car_heading_m = self.car_heading_m.unsqueeze(2).repeat(1, 1, self.num_cars)
        self.car_heading_m = torch.transpose(self.car_heading_m, 0,1)
        self.R = torch.zeros((2, 2, self.num_cars), device=self.device, dtype = torch.float)
        self.img = 255*np.ones((self.height, self.width, 3), np.uint8)
        self.track_canvas = self.img.copy()
        self.colors = 255.0/self.num_cars*np.arange(self.num_cars) 
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.do_render = True
        self.env_idx_render = 0
        self.track = track
        self.x_offset = -50
        self.y_offset = 0
        self.points = []
        self.msg = []
        self.marked_env = None
        self.state = []
        self.draw_track()
        cv.imshow('dmaracing', self.track_canvas)

    def center_cam(self, state):
        self.scale_x /= 0.7
        self.scale_y /= 0.7 
        #self.x_offset = int(-self.width/self.scale_x*state[0,0,0])
        #self.y_offset = int(self.height/self.scale_y*state[0,0,1])
        self.draw_track()
        

    def render(self, state):
        self.state = state.clone()
        if self.do_render:
            #do drawing
            #listen for keypressed events
            self.img = self.track_canvas.copy()
            self.car_img = self.img.copy()

            if self.draw_multiagent:
                self.draw_multiagent_rep(state)
            else:
                self.draw_singleagent_rep(state[:self.num_cars])
            cv.putText(self.img, "env:" + str(self.env_idx_render), (50, 50), self.font, 2, (int(self.colors[-1]),  0, int(self.colors[-1])), 1, cv.LINE_AA)
            self.draw_points()
            self.draw_string()
            self.draw_marked_agents()

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
                #print('[VIZ] env toggled to ', self.env_idx_render)
            if key == 116:
                self.env_idx_render = np.mod(self.env_idx_render-1, self.num_envs)
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

            for idx in range(self.num_cars):
                px_x_number = (self.width/self.scale_x*transl[idx, 0] + self.width/2.0).cpu().numpy().astype(np.int32).item()
                px_y_number = (-self.height/self.scale_y*transl[idx, 1] + self.height/2.0).cpu().numpy().astype(np.int32).item()
                px_pts_car = px_car_box_world[..., idx].reshape(-1,1,2)
                px_pts_heading = px_car_heading_world[..., idx].reshape(-1,1,2)
                cv.polylines(self.img, [px_pts_car], isClosed = True, color = (int(self.colors[idx]),0,int(self.colors[idx])), thickness = self.thickness)
                cv.fillPoly(self.car_img, [px_pts_car], color = (int(self.colors[idx]),0,int(self.colors[idx]), 0.9))
                cv.polylines(self.img, [px_pts_heading], isClosed = True, color = (0, 0, 255), thickness = 2)
                cv.putText(self.img, str(idx), (px_x_number+ self.x_offset, px_y_number + self.y_offset-10), self.font, 0.5, (int(self.colors[idx]),0,int(self.colors[idx])), 1, cv.LINE_AA)   
            self.img = cv.addWeighted(self.car_img, 0.3, self.img, 0.7, 0)

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
                cv.fillPoly(self.car_img, [px_pts_car], color = (int(self.colors[idx]),0,int(self.colors[idx]), 0.9))
                cv.polylines(self.img, [px_pts_heading], isClosed = True, color = (int(self.colors[idx]),0,int(self.colors[idx])), thickness = self.thickness)
                cv.putText(self.img, str(idx), (px_x_number+ self.x_offset, px_y_number + self.y_offset -10), self.font, 0.5, (int(self.colors[idx]),0,int(self.colors[idx])), 1, cv.LINE_AA)
            self.img = cv.addWeighted(self.car_img, 0.3, self.img, 0.7, 0)

    def cords2px(self, pts):
        pts = pts.cpu().numpy()
        pts[:, 0, :] = self.width/self.scale_x*pts[:, 0, :] + self.width/2.0 + self.x_offset
        pts[:, 1, :] = -self.height/self.scale_y*pts[:, 1, :] + self.height/2.0 + self.y_offset
        return pts.astype(np.int32)

    def cords2px_np(self, pts):
        pts[:, 0] = self.width/self.scale_x*pts[:, 0] + self.width/2.0 + self.x_offset
        pts[:, 1] = -self.height/self.scale_y*pts[:, 1] + self.height/2.0 + self.y_offset
        return pts.astype(np.int32)

    def add_point(self, cords, radius, color):
        cd = cords.copy()
        cd = self.cords2px_np(cd)
        self.points.append([cd, radius, color])

    def clear_markers(self,):
        self.points = []   

    def draw_points(self,):
        for group in self.points: 
            for idx in range(len(group[0])):
                cv.circle(self.img, (group[0][idx, 0], group[0][idx, 1]), group[1], group[2])

    def add_string(self, string):
        self.msg.append(string) 

    def clear_string(self,):
        self.msg = []

    def draw_string(self,):
        if len(self.msg):
            idx = 0
            for msg in self.msg:
                cv.putText(self.img, msg, (10, 100 + 40*idx), self.font, 1, (0, 0, 0), 1, cv.LINE_AA)
                idx+=1

    def mark_env(self, idx):
        self.marked_env = idx

    def draw_marked_agents(self):
        if self.marked_env is not None:
            pos = self.state[self.marked_env, 0, 0:2].view(1,-1).cpu().numpy()
            px = self.cords2px_np(pos)
            cv.circle(self.img, (px[0,0],px[0,1]), 100, (250,150,0))
    
    def draw_track(self,):
        draw_track(self.track_canvas, self.track, self.cords2px_np, self.cfg['track']['draw_centerline'])