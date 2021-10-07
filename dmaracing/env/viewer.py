import cv2 as cv
from numpy.core.numeric import isclose
import torch
import numpy as np

class Viewer:
    def __init__(self, cfg):
        self.cfg = cfg
        #load cfg
        self.width = self.cfg['viewer']['width']
        self.height = self.cfg['viewer']['height']
        self.scale_x = self.cfg['viewer']['scale']
        self.scale_y = self.height/self.width*self.scale_x
        self.thickness = self.cfg['viewer']['linethickness']
        
        #bounding box of car in model frame
        w = self.cfg['model']['w']
        lf = self.cfg['model']['lf']  
        lr = self.cfg['model']['lr']
        self.car_box_m = np.array([[lf, -w/2],[lf, w/2],[-lr, w/2], [-lr, -w/2]])
        self.car_heading_m = np.array([[0, 0],[lf, 0]])
        


        self.canvas = 255*np.ones((self.height, self.width, 3), np.uint8)
        cv.imshow('dmaracing', self.canvas)

    def render(self, state):
        #do drawing
        #listen for keypressed events
        img = 255*np.ones((self.height, self.width, 3), np.uint8)
        state = state.cpu().numpy()
        transl = state[0,0,0:2]
        theta = state[0,0,2]
        R = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        car_box_world = (R@self.car_box_m.T + np.tile(transl.reshape(2,1), (1,4))).T 
        car_heading_world = (R@self.car_heading_m.T + np.tile(transl.reshape(2,1), (1,2))).T 
        px_car_box_world = self.cords2px(car_box_world).reshape(-1,1,2)
        px_car_heading_world = self.cords2px(car_heading_world).reshape(-1,1,2)
        cv.polylines(img, [px_car_box_world], isClosed = True, color = (0,0,0), thickness = self.thickness)
        cv.polylines(img, [px_car_heading_world], isClosed = False, color = (0,0,0), thickness = self.thickness)
        cv.imshow("dmaracing", img)
        key = cv.waitKey(1)
        return key

    def cords2px(self, pts):
        pts[:, 0] = (self.width/self.scale_x*pts[:, 0] + self.width/2.0).astype(np.int)
        pts[:, 1] = (self.height/self.scale_y*pts[:, 1] + self.height/2.0).astype(np.int)
        return pts.astype(np.int32)