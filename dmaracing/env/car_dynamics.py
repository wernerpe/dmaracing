import torch
from typing import Dict, List, Tuple
import numpy as np

from dmaracing.env.car_dynamics_utils import resolve_collsions
from dmaracing.env.car_dynamics_utils import SwitchedBicycleKinodynamicModel

# @torch.jit.script
def step_cars(
    state: torch.Tensor,
    actions: torch.Tensor,
    drag_reduced: torch.Tensor,
    wheel_locations: torch.Tensor,
    R: torch.Tensor,
    contact_wrenches: torch.Tensor,
    is_rear_end: torch.Tensor,
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
    new_state = dyn_model.forward(dyn_state.clone(), dyn_control, contact_wrenches, shove).detach()
    slip *= (dyn_model.time_since_col<dyn_model.col_decay_time).view(-1, state.shape[1], 1)
    
    if collide:
        # resolve collisions using a combination of spring force and "shove" which pushes the vertex
        # in collision out of collision
        contact_wrenches, shove, is_rear_end = resolve_collsions(
            contact_wrenches,
            is_rear_end,
            shove,
            new_state,
            collision_pairs,
            mod_par['L_coll']/2,
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
    wheel_locations: torch.Tensor,
    R: torch.Tensor,
    vn: Dict[str, int],
    num_envs: int,
    wheels_on_track_segments: torch.Tensor,
    active_track_mask: torch.Tensor,
    A_track: torch.Tensor,
    b_track: torch.Tensor,
    S_track: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    theta = state[:, :, vn["S_THETA"]]
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
    dyn_state = state
    dyn_control = torch.cat([torch.clamp(actions[:, :, vn["A_STEER"]], -0.35 , .35).unsqueeze(2), 
                             actions[:, :, vn["A_GAS"]].unsqueeze(2)], dim=2)
   
    return dyn_state, dyn_control, slip, wheel_locations_world
