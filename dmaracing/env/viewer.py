import cv2 as cv
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

        self.canvas = 255*np.ones((self.height, self.width, 3), np.uint8)
        cv.imshow('dmaracing', self.canvas)

    def render(self, state):
        #do drawing
        #listen for keypressed events
        self.canvas[35:45,35:45, :] = self.canvas[35:45,35:45, :] + 10 
        cv.imshow("dmaracing", self.canvas)
        key = cv.waitKey(1)
        return key