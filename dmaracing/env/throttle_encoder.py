import torch
import torch.nn as nn
from collections import OrderedDict

encoder_dict = OrderedDict([('0.weight', torch.tensor([[-1.4911,  0.0604,  1.2223],
        [ 0.2872, -0.1739,  1.5631],
        [ 0.4241, -0.0029, -1.9037],
        [ 0.4564,  0.0474,  2.2591]])), ('0.bias', torch.tensor([0.0498, 1.1539, 0.7325, 0.0019])), ('2.weight', torch.tensor([[-0.4032, -0.3328, -0.3307,  0.0975],
        [ 2.8774,  0.7871, -0.9816,  0.4903],
        [-0.0559,  0.3570, -0.5179, -0.5398],
        [ 0.3623,  0.4138,  1.7321, -0.8834]])), ('2.bias', torch.tensor([-0.4956,  0.8397, -0.2425,  1.0157])), ('4.weight', torch.tensor([[ 0.2481,  1.6715, -0.1775, -1.1801]])), ('4.bias', torch.tensor([0.9727]))])

def get_throttle_encoder():
    enc = nn.Sequential(nn.Linear(3, 4),
                nn.ReLU(),
                nn.Linear(4, 4),
                nn.ReLU(),
                nn.Linear(4, 1))
    enc.load_state_dict(encoder_dict)
    return enc