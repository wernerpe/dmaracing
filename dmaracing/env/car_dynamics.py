import torch
from typing import Dict, List, Tuple
import numpy as np

from dmaracing.env.car_dynamics_utils import resolve_collsions
# Add Dynamics directory to python path. Change this path to match that of your local system!
# import sys
# sys.path.insert(1, '/home/peter/git/dynamics_model_learning/scripts')
# Import Dynamics encoder from TRI dynamics library.
from dynamics_lib import DynamicsEncoder

from dmaracing.env.car_dynamics_utils import SwitchedBicycleKinodynamicModel

#@torch.jit.script
def step_cars(
    state: torch.Tensor,
    actions: torch.Tensor,
    drag_reduced: torch.Tensor,
    wheel_locations: torch.Tensor,
    R: torch.Tensor,
    contact_wrenches: torch.Tensor,
    shove: torch.Tensor,
    mod_par: Dict[str, float],
    sim_par: Dict[str, float],
    vn: Dict[str, int],
    collision_pairs: List[List[int]],
    collision_verts: torch.Tensor,
    P_tot: torch.Tensor,
    D_tot: torch.Tensor,
    S_mat: torch.Tensor,
    Repf_mat: torch.Tensor,
    Ds: List[int],
    num_envs: int,
    zero_pad: torch.Tensor,
    collide: int,
    wheels_on_track_segments: torch.Tensor,
    active_track_mask: torch.Tensor,
    A_track: torch.Tensor,
    b_track: torch.Tensor,
    S_track: torch.Tensor,
    dyn_model: SwitchedBicycleKinodynamicModel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    dyn_state, dyn_control, slip, wheel_locations_world = get_state_control_tensors(
        state,
        actions,
        #drag_reduced,
        wheel_locations,
        R,
        #mod_par,
        #sim_par,
        vn,
        num_envs,
        wheels_on_track_segments,
        active_track_mask,
        A_track,
        b_track,
        S_track,
    )
    # TODO: PETER Fix this with new dynamics model
    #dyn_state_shape = dyn_state.shape
    #contact_wrenches_shape = contact_wrenches.shape
    #input_state_shape = [dyn_state_shape[0] * dyn_state_shape[1], dyn_state_shape[2]]
    #dyn_control_shape = dyn_control.shape
    #input_control_shape = [dyn_control_shape[0] * dyn_control_shape[1], dyn_control_shape[2]]
    #col_shape = [contact_wrenches_shape[0] * contact_wrenches_shape[1], contact_wrenches_shape[2]]
    
    # stepped_state = dyn_model.dynamics_integrator.step_state(
    #     dyn_state.reshape(input_state_shape).to(dyn_model.device), dyn_control.reshape(input_control_shape).to(dyn_model.device),
    #     contact_wrenches.reshape(col_shape).to(dyn_model.device), shove.reshape(col_shape).to(dyn_model.device)
    # )
    print('dyn pre', dyn_state[0,0,3])
    stepped_state = dyn_model.forward(dyn_state.clone(), dyn_control, contact_wrenches, shove)
    #print(dyn_control[0,0,:])
    print('dyn_post',stepped_state[0,0,3])
    #print(stepped_state[0,0,0])
    slip *= (dyn_model.time_since_col<dyn_model.col_decay_time).view(-1, state.shape[1], 1)
    
    #TODO Check what state goes where whithout intermediate layer

    # stepped_state = stepped_state.reshape(dyn_state_shape)
    # new_state[..., vn["S_X"]] = stepped_state[...,0]
    # new_state[..., vn["S_Y"]] = stepped_state[...,1]
    # new_state[..., vn["S_THETA"]] = stepped_state[...,2]
    # new_state[..., vn["S_DX"]] = stepped_state[...,4]
    # new_state[..., vn["S_DY"]] = stepped_state[...,5]
    # new_state[..., vn["S_DTHETA"]] = stepped_state[...,6]
    # #new_state[..., vn["S_W0"]] = stepped_state[...,3] # Roll
    # new_state[..., vn["S_STEER"]] = state[...,vn["S_STEER"]]
    # #new_state[..., vn["S_GAS"]] = state[...,vn["S_GAS"]]
    new_state = stepped_state.detach()
    
    if collide:
        # resolve collisions using a combination of spring force and "shove" which pushes the vertex
        # in collision out of collision
        contact_wrenches, shove = resolve_collsions(
            contact_wrenches,
            shove,
            new_state,
            collision_pairs,
            mod_par['lf'],
            collision_verts,
            R[:, 0, :, :],
            P_tot,
            D_tot,
            S_mat,
            Repf_mat,
            Ds,
            zero_pad,
            sim_par["collisionstiffness"],
            dyn_model.Iz,
        )

    return new_state, contact_wrenches, shove, wheels_on_track_segments, slip, wheel_locations_world

@torch.jit.script
def get_state_control_tensors(
    state: torch.Tensor,
    actions: torch.Tensor,
    #drag_reduced: torch.Tensor,
    wheel_locations: torch.Tensor,
    R: torch.Tensor,
    #mod_par: Dict[str, float],
    #sim_par: Dict[str, float],
    vn: Dict[str, int],
    num_envs: int,
    wheels_on_track_segments: torch.Tensor,
    active_track_mask: torch.Tensor,
    A_track: torch.Tensor,
    b_track: torch.Tensor,
    S_track: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    theta = state[:, :, vn["S_THETA"]]
    #delta = state[:, :, vn["S_STEER"]]

    state[:, :, vn["S_STEER"]] = torch.clamp(actions[:, :, vn["A_STEER"]], -0.35 , .35)

    R[:, :, 0, 0] = torch.cos(theta)
    R[:, :, 0, 1] = -torch.sin(theta)
    R[:, :, 1, 0] = torch.sin(theta)
    R[:, :, 1, 1] = torch.cos(theta)
    # (num_envs, num_agents, 2, 2) x (num_envs, num_agents, num_wheels, 2)
    wheel_locations_bodycentric_world = torch.einsum("klij, dj->kldi", R, wheel_locations)
    wheel_locations_world = wheel_locations_bodycentric_world + torch.tile(
        state[:, :, vn["S_X"] : vn["S_Y"] + 1].unsqueeze(2), (1, 1, 4, 1)
    )
    wheel_locations_bodycentric_world = torch.nn.functional.pad(wheel_locations_bodycentric_world, (0, 1))

    slip = 0*wheels_on_track_segments[..., 0] < 10000

    wheels_on_track_segments_concat = 1.0 * (
        torch.einsum("es, stc, eawc  -> eawt", active_track_mask, A_track, wheel_locations_world)
        - torch.einsum("es, st -> et", active_track_mask, b_track).view(num_envs, 1, 1, -1)
        + 0.01
        >= 0
    )
    wheels_on_track_segments[:] = torch.einsum("jt, eawt -> eawj", S_track, wheels_on_track_segments_concat) >= 3.5
    
    # yaw = state[:, :, vn["S_THETA"]]  # atan2(2 * q1 * q2 + 2 * q0 * q3, q1 * q1 + q0 * q0 - q3 * q3 - q2 * q2)
    # #roll = torch.atan2(2 * q2 * q3 + 2 * q0 * q1, q3 * q3 - q2 * q2 - q1 * q1 + q0 * q0)
    # roll = 0*state[:, :, vn["S_THETA"]]
    # yaw_rate = state[:, :, vn["S_DTHETA"]]

    dyn_state = state
    # =torch.cat(
    #     [
    #         state[:, :, vn["S_X"] : vn["S_Y"] + 1],
    #         yaw.unsqueeze(2),
    #         roll.unsqueeze(2), #3
    #         state[:, :, vn["S_DX"] : vn["S_DY"] + 1], #45
    #         yaw_rate.unsqueeze(2),
    #     ],
    #     dim=2,
    # )  # x and y in world frame, dx and dy need to be rotated by theta to be in body frame
    # clipped_max = (state[:, :, vn["S_DX"]] < 5.0) * 1.0
    # clipped_min = (state[:, :, vn["S_DX"]] > -0.1) * -1.0  # 0.0
    dyn_control = torch.cat([torch.clamp(actions[:, :, vn["A_STEER"]], -0.35 , .35).unsqueeze(2), 
                             actions[:, :, vn["A_GAS"]].unsqueeze(2)], dim=2)
   
    return dyn_state, dyn_control, slip, wheel_locations_world
