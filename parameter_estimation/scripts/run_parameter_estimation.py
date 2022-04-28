from functools import partial
import torch
import numpy as np

from torch.autograd import Variable

from parameter_estimation.utils import mlefit


iz = Variable(torch.tensor(0.030), requires_grad=True)
bf = Variable(torch.tensor(0.711), requires_grad=True)
cf = Variable(torch.tensor(1.414), requires_grad=True)
df = Variable(torch.tensor(0.892), requires_grad=True)
br = Variable(torch.tensor(2.482), requires_grad=True)
cr = Variable(torch.tensor(1.343), requires_grad=True)
dr = Variable(torch.tensor(0.892), requires_grad=True)

def residual(iz, bf, cf, df, br, cr, dr, state, control, dt):

    state_tm1 = state[:-1]
    control_tm1 = control[:-1]

    # state
    vel_x = state_tm1[..., 3]     # vel_x
    vel_y = state_tm1[..., 4]     # vel_y
    phi = state_tm1[..., 2]       # pos_theta
    omega = state_tm1[..., 5]    # acc_theta

    # control
    delta = control_tm1[..., 0]
    acc = control_tm1[..., 1] 

    # params
    m = 3.3
    lf = 0.19
    lr = 0.14

    alf = -torch.arctan((omega*lf + vel_y)/vel_x) + delta
    alr = torch.arctan((omega*lr - vel_y)/vel_x)

    Ffy = df * torch.sin(cf * torch.arctan(bf * alf))
    Fry = dr * torch.sin(cr * torch.arctan(br * alr))
    Frx = acc
    
    dx = vel_x * torch.cos(phi) - vel_y * torch.sin(phi)
    dy = vel_x * torch.sin(phi) + vel_y * torch.cos(phi)
    dphi = omega
    dvel_x = 1./m * (Frx - Ffy * torch.sin(delta) + m * vel_y * omega)
    dvel_y = 1./m * (Fry + Ffy * torch.cos(delta) - m * vel_x * omega)
    domega = 1./iz * (Ffy * lf * torch.cos(delta) - Fry * lr)

    dstate = torch.stack([dx, dy, dphi, dvel_x, dvel_y, domega], axis=-1)
    state_est = state_tm1 + dt * dstate

    return state[1:] - state_est



residual_param = partial(residual, iz, bf, cf, df, br, cr, dr)

data = np.load('parameter_estimation/data/racecar_3_2022-03-22-10-34-20.npz')

states = data['states']
controls = data['controls']
time = data['time']
dtime = time[1:] - time[:-1]

mlefit(residual_param, [iz, bf, cf, df, br, cr, dr], (states, controls, dtime[:-1]))



pass





    
