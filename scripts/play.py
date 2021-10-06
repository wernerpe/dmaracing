import torch
from dmaracing.env.racing_sim import DmarEnv
from dmaracing.utils.helpers import *
import os

path_cfg = os.getcwd() + 'cfg'
cfg, cfg_train = getcfg(path_cfg)

