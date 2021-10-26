import cv2 as cv
from numpy.core.numeric import isclose
import torch
import numpy as np
import sys
from dmaracing.utils.trackgen import get_track

class Viewer:
    def __init__(self, cfg):
        self.device = 'cuda:0'
        self.cfg = cfg

        #load cfg
        self.width = self.cfg['viewer']['width']
        self.height = self.cfg['viewer']['height']
        self.scale_x = self.cfg['viewer']['scale']
        self.scale_y = self.height/self.width*self.scale_x
        self.thickness = self.cfg['viewer']['linethickness']
        self.num_agents = self.cfg['sim']['numAgents']
        self.num_envs = self.cfg['sim']['numEnv']

        #bounding box of car in model frame
        w = self.cfg['model']['W']
        lf = self.cfg['model']['lf']  
        lr = self.cfg['model']['lr']
        self.car_box_m = torch.tensor([[lf, -w/2],[lf, w/2],[-lr, w/2], [-lr, -w/2]], device = self.device)
        self.car_box_m = self.car_box_m.unsqueeze(2).repeat(1, 1, self.num_agents)
        self.car_box_m = torch.transpose(self.car_box_m, 0,1) 
        self.car_heading_m = torch.tensor([[0, 0],[lf, 0]], device = self.device)
        self.car_heading_m = self.car_heading_m.unsqueeze(2).repeat(1, 1, self.num_agents)
        self.car_heading_m = torch.transpose(self.car_heading_m, 0,1)
        self.R = torch.zeros((2, 2, self.num_agents), device=self.device, dtype = torch.float)
        self.img = 255*np.ones((self.height, self.width, 3), np.uint8)
        self.colors = 255.0/self.num_agents*np.arange(self.num_agents) 
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.do_render = True
        self.env_idx_render = 0
        self.track = get_track(2)
        cv.imshow('dmaracing', self.img)

    def render(self, state):
        if self.do_render:
            #do drawing
            #listen for keypressed events
            self.img = self.track.copy()#255*np.ones((self.height, self.width, 3), np.uint8)
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

            for idx in range(self.num_agents):
                px_x_number = (self.width/self.scale_x*transl[idx, 0] + self.width/2.0).cpu().numpy().astype(np.int32).item()
                px_y_number = (-self.height/self.scale_y*transl[idx, 1] + self.height/2.0).cpu().numpy().astype(np.int32).item()
                px_pts_car = px_car_box_world[..., idx].reshape(-1,1,2)
                px_pts_heading = px_car_heading_world[..., idx].reshape(-1,1,2)
                cv.polylines(self.img, [px_pts_car], isClosed = True, color = (int(self.colors[idx]),0,int(self.colors[idx])), thickness = self.thickness)
                cv.polylines(self.img, [px_pts_heading], isClosed = True, color = (int(self.colors[idx]),0,int(self.colors[idx])), thickness = self.thickness)
                cv.putText(self.img, str(idx), (px_x_number, px_y_number), self.font, 0.5, (int(self.colors[idx]),0,int(self.colors[idx])), 1, cv.LINE_AA)
            cv.putText(self.img, "env:" + str(self.env_idx_render), (50, 50), self.font, 2, (int(self.colors[idx]),  0, int(self.colors[idx])), 1, cv.LINE_AA)
            
        cv.imshow("dmaracing", self.img)
        key = cv.waitKey(1)
        if key == 118: #toggle render on v
            if self.do_render:
                self.do_render = False
            else: 
                self.do_render = True
            print('[VIZ] render toggled to ', self.do_render)
        if key == 113 or key == 27: #quit on escape or q
            sys.exit()
        if key == 114:
            self.env_idx_render = np.mod(self.env_idx_render+1, self.num_envs)
            #print('[VIZ] env toggled to ', self.env_idx_render)
        if key == 116:
            self.env_idx_render = np.mod(self.env_idx_render-1, self.num_envs)
            #print('[VIZ] env toggled to ', self.env_idx_render)
        return key

    def cords2px(self, pts):
        pts = pts.cpu().numpy()
        pts[:, 0, :] = self.width/self.scale_x*pts[:, 0, :] + self.width/2.0
        pts[:, 1, :] = -self.height/self.scale_y*pts[:, 1, :] + self.height/2.0
        return pts.astype(np.int32)