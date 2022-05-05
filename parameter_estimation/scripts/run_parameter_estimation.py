from functools import partial
import torch
import numpy as np

from torch.autograd import Variable

from parameter_estimation.utils import forcefit, mlefit

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


iz_prior = 0.030
bf_prior = 0.711
cf_prior = 1.414
df_prior = 0.892
br_prior = 2.482
cr_prior = 1.343
dr_prior = 0.892


iz = Variable(torch.tensor(iz_prior, dtype=torch.float32), requires_grad=True)
bf = Variable(torch.tensor(bf_prior, dtype=torch.float32), requires_grad=True)
cf = Variable(torch.tensor(cf_prior, dtype=torch.float32), requires_grad=True)
df = Variable(torch.tensor(df_prior, dtype=torch.float32), requires_grad=True)
br = Variable(torch.tensor(br_prior, dtype=torch.float32), requires_grad=True)
cr = Variable(torch.tensor(cr_prior, dtype=torch.float32), requires_grad=True)
dr = Variable(torch.tensor(dr_prior, dtype=torch.float32), requires_grad=True)
# iz = Variable(torch.tensor(0.), requires_grad=True)
# bf = Variable(torch.tensor(0.), requires_grad=True)
# cf = Variable(torch.tensor(0.), requires_grad=True)
# df = Variable(torch.tensor(0.), requires_grad=True)
# br = Variable(torch.tensor(0.), requires_grad=True)
# cr = Variable(torch.tensor(0.), requires_grad=True)
# dr = Variable(torch.tensor(0.), requires_grad=True)

def residual(iz, bf, cf, df, br, cr, dr, state_tm1, control_tm1, state_t, dt):

    # state
    pos_x = state_tm1[..., 0]
    pos_y = state_tm1[..., 1]
    phi = state_tm1[..., 2]
    vel_x = state_tm1[..., 3]     # vel_x
    vel_y = state_tm1[..., 4]     # vel_y
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

    return state_est



residual_param = partial(residual, iz, bf, cf, df, br, cr, dr)

data = np.load('parameter_estimation/data/racecar_3_2022-03-22-10-34-20.npz')

states = np.transpose(data['states'], [1, 0])
controls = np.transpose(data['controls'], [1, 0])
time = data['time']
dtime = time[1:] - time[:-1]

states = torch.tensor(states[..., :6], requires_grad=False, dtype=torch.float32)
controls = torch.tensor(controls, requires_grad=False, dtype=torch.float32)
dtime = torch.tensor(dtime, requires_grad=False, dtype=torch.float32).unsqueeze(dim=1)

states_tm1 = states[:-1]
controls_tm1 = controls[:-1]
states_t = states[1:]

# mlefit(residual_param, [iz, bf, cf, df, br, cr, dr], (states, controls, dtime))
params = forcefit(residual_param, [iz, bf, cf, df, br, cr, dr], (states_tm1, controls_tm1, states_t, dtime), iter=10000)


iz_fitted = params[0]
bf_fitted = params[1]
cf_fitted = params[2]
df_fitted = params[3]
br_fitted = params[4]
cr_fitted = params[5]
dr_fitted = params[6]

names = ['iz', 'bf', 'cf', 'df', 'br', 'cr', 'dr']
prior = [iz_prior, bf_prior, cf_prior, df_prior, br_prior, cr_prior, dr_prior]
fitted = [iz_fitted, bf_fitted, cf_fitted, df_fitted, br_fitted, cr_fitted, dr_fitted]

for name, old, new in zip(names, prior, fitted):
    print(name + ": " + str(old) + " ---> " + str(new.detach().numpy()))


# Test
test = np.load('parameter_estimation/data/racecar_3_2022-03-22-11-11-48.npz')

test_states = np.transpose(test['states'], [1, 0])
test_controls = np.transpose(test['controls'], [1, 0])
test_time = test['time']
test_dtime = test_time[1:] - test_time[:-1]

test_states = torch.tensor(test_states[..., :6], requires_grad=False, dtype=torch.float32)
test_controls = torch.tensor(test_controls, requires_grad=False, dtype=torch.float32)
test_dtime = torch.tensor(test_dtime, requires_grad=False, dtype=torch.float32).unsqueeze(dim=1)

test_states_tm1 = test_states[:-1]
test_controls_tm1 = test_controls[:-1]
test_states_t = test_states[1:]

predicted = residual_param(test_states_tm1, test_controls_tm1, test_states_t, test_dtime)
error = test_states_t - predicted
print("Error mean = " + str(error.abs().mean()))

pass





    
