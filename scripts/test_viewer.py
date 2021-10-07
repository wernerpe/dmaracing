import torch
from torch._C import dtype
from dmaracing.env.viewer import Viewer
from dmaracing.utils.helpers import getcfg
import os
import numpy as np



path_cfg = os.getcwd() + '/cfg'
cfg, cfg_train = getcfg(path_cfg)

viewer = Viewer(cfg)
state_dummy = torch.zeros((2,2,7), dtype= torch.float, requires_grad=False, device='cuda:0')
state_dummy[:, 1, 0] = 0.5
state_dummy[:, :, 2] = np.pi/6

while True:
    evnt = viewer.render(state_dummy[:,:,0:3])
    if evnt == 118:
        print('toggle viewer')
    if evnt == 113 or evnt == 27:
        break