import torch
from torch._C import dtype
from dmaracing.env.viewer import Viewer
from dmaracing.utils.helpers import getcfg
import os
import numpy as np



path_cfg = os.getcwd() + '/cfg'
cfg, cfg_train = getcfg(path_cfg)

viewer = Viewer(cfg)
state_dummy = torch.zeros((2,cfg['sim']['numAgents'],7), dtype= torch.float, requires_grad=False, device='cuda:0')
state_dummy[0, :, 0] = -1 + 2.0/cfg['sim']['numAgents']*torch.arange(cfg['sim']['numAgents'])
state_dummy[0, :, 2] = 2.0*np.pi/cfg['sim']['numAgents']*torch.arange(cfg['sim']['numAgents'])

while True:
    evnt = viewer.render(state_dummy[:,:,0:3])
    if evnt == -100:
        break