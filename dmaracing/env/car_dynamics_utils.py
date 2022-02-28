import torch
import numpy as np
from typing import List, Tuple, Dict

from torch import device

def get_varnames()->Dict[str, int]:
    varnames = {}
    varnames['S_X'] = 0
    varnames['S_Y'] = 1
    varnames['S_THETA'] = 2
    varnames['S_DX'] = 3
    varnames['S_DY'] = 4
    varnames['S_DTHETA'] = 5
    varnames['S_W0'] = 6
    varnames['S_W1'] = 7
    varnames['S_W2'] = 8
    varnames['S_W3'] = 9
    varnames['S_STEER'] = 10
    varnames['S_GAS'] = 11
    
    varnames['A_STEER'] = 0
    varnames['A_GAS'] = 1
    varnames['A_BRAKE'] = 2
    return varnames

def allocate_car_dynamics_tensors(task):
    task.R = torch.zeros((task.num_envs, task.num_agents, 2, 2), dtype = torch.float, device = task.device, requires_grad=False)
    task.zero_pad = torch.zeros((task.num_envs,4,1), device =task.device, requires_grad=False) 

    task.P_tot, task.D_tot, task.S_mat, task.Repf_mat, task.Ds = build_col_poly_eqns(task.modelParameters['W'], task.modelParameters['lr'] + task.modelParameters['lf'], task.device, task.num_envs)
    task.collision_pairs = get_collision_pairs(task.num_agents)
    task.collision_verts = get_car_vert_mat(task.modelParameters['W'], 
                                            task.modelParameters['lr'] + task.modelParameters['lf'], 
                                            task.num_envs, 
                                            task.device)
    L = task.modelParameters['L']
    W = task.modelParameters['W']
    task.wheel_locations = torch.zeros((4,2), device = task.device, dtype=torch.float, requires_grad=False)
    task.wheel_locations[0, 0] = L/2.0 
    task.wheel_locations[0, 1] = W/2.0 
    task.wheel_locations[1, 0] = L/2.0 
    task.wheel_locations[1, 1] = -W/2.0 
    task.wheel_locations[2, 0] = -L/2.0 
    task.wheel_locations[2, 1] = -W/2.0 
    task.wheel_locations[3, 0] = -L/2.0 
    task.wheel_locations[3, 1] = W/2.0 

    task.wheels_on_track_segments = torch.zeros((task.num_envs, task.num_agents, 4, task.max_track_num_tiles), requires_grad=False, device=task.device)>1
    task.slip = torch.zeros((task.num_envs, task.num_agents, 4), requires_grad=False, device=task.device)>1
    task.wheel_locations_world = torch.zeros((task.num_envs, task.num_agents, 4, 2), requires_grad=False, device=task.device)
    task.drag_reduction_points = torch.zeros((task.num_envs, task.num_agents*30, 2), requires_grad=False, device=task.device)
    task.drag_reduction_write_idx = 0
    task.drag_reduced = torch.zeros((task.num_envs, task.num_agents), requires_grad=False, device=task.device) > 1.0
    

def set_dependent_params(mod_par):
    SIZE = mod_par['SIZE']
    mod_par['ENGINE_POWER'] = mod_par['ENGINE_POWER_SCALE']*SIZE**2
    mod_par['WHEEL_MOMENT_OF_INERTIA'] = mod_par['WHEEL_MOMENT_OF_INERTIA_SCALE']*SIZE**2
    mod_par['FRICTION_LIMIT'] = mod_par['FRICTION_LIMIT_SCALE'] * SIZE * SIZE
    mod_par['WHEEL_R'] = SIZE*mod_par['WHEEL_R_SCALE']
    L = 160.0 *SIZE
    W = L/2
    M = L*W *mod_par['MASS_SCALE']
    mod_par['M'] = M
    mod_par['L'] = L
    mod_par['W'] = W 
    mod_par['I'] = mod_par['MOMENT_OF_INERTIA_SCALE']*M*(L**2 + W**2 )/12.0
    mod_par['lf'] = L/2
    mod_par['lr'] = L/2
    
#only used for collsion checking, only one car per env needed for pairwise checking
def get_car_vert_mat(w, l, num_envs, device):
    verts = torch.zeros((num_envs, 4, 2), device=device, dtype=torch.float, requires_grad=False)
    #top lft , ccw
    # 3 x-------x 0
    #   |       |
    # 2 X-------X 1
    #
    # ^ y 
    # ->x
    verts[:, 0, 0] = l/2
    verts[:, 0, 1] = w/2
    verts[:, 1, 0] = l/2
    verts[:, 1, 1] = -w/2
    verts[:, 2, 0] = -l/2
    verts[:, 2, 1] = -w/2
    verts[:, 3, 0] = -l/2
    verts[:, 3, 1] = w/2
    return verts

def get_collision_pairs(num_agents):
    #naively check all collision pairs
    ls = np.arange(num_agents)
    pairs = [[b, a] for idx, a in enumerate(ls) for b in ls[idx + 1:]]
    return pairs


def build_col_poly_eqns(w,
                        l, 
                        device, 
                        num_envs):

    P_tot = torch.zeros((16,2), device = device, dtype= torch.float, requires_grad=False)
    D_tot = torch.zeros((16,1), device = device, dtype= torch.float, requires_grad=False)
    #P1
    P_tot[0,1] = -1
    P_tot[1,1] = 1
    P_tot[2,0] = -1/np.sqrt(2)
    P_tot[2,1] = -1/np.sqrt(2)
    P_tot[3,0] = 1/np.sqrt(2)
    P_tot[3,1] = -1/np.sqrt(2)
    #P2
    P_tot[4:8, :] = P_tot[0:4, :]
    P_tot[4:8, 1] = -P_tot[4:8, 1]

    #P3
    #diagonal lines dont have to be normalzed, because not used for computing foce
    P_tot[8, 0] = -1
    P_tot[9, 0] = 1
    P_tot[10, 0] = 1
    P_tot[10, 1] = -1
    P_tot[11, 0] = 1
    P_tot[11, 1] = 1

    #P4
    P_tot[12:16, :] = P_tot[8:12, :]
    P_tot[12:16, 0] = -P_tot[8:12, 0]
    
    #D1
    D_tot[1,0] = w/2
    D_tot[2,0] = l/(2*np.sqrt(2)) - w/(2*np.sqrt(2))
    D_tot[3,0] = l/(2*np.sqrt(2)) - w/(2*np.sqrt(2))
    #D2
    D_tot[4:8, 0] = D_tot[0:4, 0]

    #D3
    D_tot[8] = l/2
    D_tot[9] = 0
    D_tot[10] = (-l+w)/2
    D_tot[11] = (-l+w)/2

    #D4
    D_tot[12:16, 0] = D_tot[8:12, 0]

    #summation template for polygon checking 
    S_mat = torch.eye(4, device=device, dtype=torch.float, requires_grad = False)
    tmp = torch.ones((1,4), device=device, dtype=torch.float, requires_grad = False)
    S_mat = torch.kron(S_mat, tmp)

    #repulsion force direction
    # colision polygon order
    #
    #            ^ y
    #        l   |
    # 3 x----------------x 0
    #   | \      2   /   | 
    #   | 4    ----   3  | w ->x
    #   | /     1    \   |
    # 2 X----------------X 1
    #
    #  repforce mat = [n1, n2, n3, n4] ^t
    Rep_froce_dir = torch.zeros((num_envs, 4,2), device= device, dtype=torch.float, requires_grad=False)
    Rep_froce_dir[:,0,1] = -1.0
    Rep_froce_dir[:,1,1] = 1.0
    Rep_froce_dir[:,2,0] = 1.0
    Rep_froce_dir[:,3,0] = -1.0
    
    Depth_selector = [1, 1 + 4, 0 + 8, 0 + 12]
    return P_tot, D_tot, S_mat, Rep_froce_dir, Depth_selector

@torch.jit.script
def transform_col_verts(rel_trans_global : torch.Tensor, 
                        theta_A : torch.Tensor, 
                        theta_B : torch.Tensor, 
                        verts: torch.Tensor,
                        R: torch.Tensor) ->torch.Tensor:
    '''
    collider -> (B) Vertex rep
    collidee -> (A) Poly rep

    input: 
    rel_trans_global (num_envs, 2) :relative translation pB-pA in global frame 
    theta_A (num_envs) : relative rotation of A to world
    theta_B (num_envs) : relative rotation of B to world
    verts : vertex tensor (numenvs, 4 car vertices, 2 cords)
    R: allocated rot mat (numenvs, 2, 2) 

    output:
    verts_rot_shift (num_envs, 4 car vertices, 2)
    '''
    theta_rel = theta_B - theta_A
    R[:, 0, 0 ] = torch.cos(theta_rel)
    R[:, 0, 1 ] = -torch.sin(theta_rel)
    R[:, 1, 0 ] = torch.sin(theta_rel)
    R[:, 1, 1 ] = torch.cos(theta_rel)
    verts_rot = torch.einsum('kij, klj->kli', R, verts)
    #rotate global translation into A frame
    R[:, 0, 0 ] = torch.cos(theta_A)
    R[:, 0, 1 ] = torch.sin(theta_A)
    R[:, 1, 0 ] = -torch.sin(theta_A)
    R[:, 1, 1 ] = torch.cos(theta_A)
    trans_rot = torch.einsum('kij, kj->ki', R, rel_trans_global)

    verts_rot_shift = verts_rot
    verts_rot_shift[:,0,:] += trans_rot[:]
    verts_rot_shift[:,1,:] += trans_rot[:]
    verts_rot_shift[:,2,:] += trans_rot[:]
    verts_rot_shift[:,3,:] += trans_rot[:]
    return verts_rot_shift

@torch.jit.script
def rotate_vec(vec : torch.Tensor, 
               theta : torch.Tensor,  
               R: torch.Tensor) ->torch.Tensor:
    '''
    collider -> (B) Vertex rep
    collidee -> (A) Poly rep

    input: 
    vec (num_envs, 2) : vector to rotate 
    theta (num_envs) : angle around z-axis to rotate
    R: allocated rot mat (numenvs, 2, 2) 

    output:
    verts_rot (num_envs, 2)
    '''
    R[:, 0, 0 ] = torch.cos(theta)
    R[:, 0, 1 ] = -torch.sin(theta)
    R[:, 1, 0 ] = torch.sin(theta)
    R[:, 1, 1 ] = torch.cos(theta)
    vec_rot = torch.einsum('kij, kj->ki', R, vec)
    return vec_rot


@torch.jit.script
def get_contact_wrenches(P_tot : torch.Tensor, 
                         D_tot : torch.Tensor,
                         S_mat : torch.Tensor,
                         Repf_mat : torch.Tensor, 
                         Depth_selector : List[int],
                         verts_tf : torch.Tensor,
                         num_envs : int,
                         zero_pad : torch.Tensor,
                         stiffness : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #evaluate polygon equations on vertices of collider in bodyframe of collidee
    vert_poly_dists = torch.einsum('ij, lkj->lki', P_tot, verts_tf) + torch.tile(D_tot.squeeze(), (num_envs, 4, 1)) 
    #check if in polygon
    in_poly = torch.einsum('ij, lkj -> lki', S_mat, 1.0*(vert_poly_dists+1e-4>=0)) > 3 #watch for numerics here #!$W%!$#!
    #inpoly (numenvs, num_verts, num_poly) Repforcedir (num_envs, num_poly, coords) -> num_env, num_verts, forcecoords
    force_dir = torch.einsum('ijk, ikl -> ijl', 1.0*in_poly, Repf_mat)
    #inpoly (numenvs, num_verts, num_poly) #vert_poly_dists[:,:, Ds] (num_env, num_verts, num_polygons)
    #use fact that we precomputed all distances to polygon walls
    depths = torch.einsum('ijk, ijk -> ij', 1.0*in_poly, vert_poly_dists[:,:, Depth_selector])
    magnitude = stiffness*depths
    forces = force_dir
    forces[:, :, 0] *= magnitude
    forces[:, :, 1] *= magnitude

    dirs_vert = torch.cat((verts_tf - torch.tile(torch.mean(verts_tf, dim = 1).unsqueeze(1), (1,4,1)), zero_pad ), dim = 2)
    dirs_force = torch.cat((forces[:, :, :], zero_pad), dim = 2)

    torque_B = torch.sum(torch.cross(dirs_vert, dirs_force, dim = 2)[:,:, 2], dim = 1)
    dirs_vert_A = torch.cat((verts_tf, zero_pad), dim = 2)
    torque_A = torch.sum(torch.cross(dirs_vert_A, -dirs_force, dim = 2)[:,:, 2], dim = 1)
    
    #sum over contact forces acting at the individual vertices
    forces = torch.sum(forces, dim = 1)

    return forces, torque_A, torque_B


@torch.jit.script
def resolve_collsions(contact_wrenches : torch.Tensor,
                      shove : torch.Tensor,
                      states : torch.Tensor,
                      collision_pairs : List[List[int]],
                      lf : float,
                      collision_verts : torch.Tensor,
                      R : torch.Tensor,
                      P_tot : torch.Tensor,
                      D_tot : torch.Tensor,
                      S_mat : torch.Tensor,
                      Repf_mat : torch.Tensor,
                      Ds : List[int],
                      zero_pad : torch.Tensor,
                      stiffness : float
                      ) -> Tuple[torch.Tensor, torch.Tensor]:

    contact_wrenches[:,:,:] =0.0
    shove[:,:,:] = 0.0
    #switch = torch.rand(len(collision_pairs))>0.5
    if len(collision_pairs):
        for id, colp in enumerate(collision_pairs):
            #colp = [colp[switch[id]*1], colp[~switch[id]*1]]
            idx_comp = torch.where(torch.norm(states[:, colp[0], 0:2] -  states[:, colp[1], 0:2], dim =1)<=2.3*lf)[0]
            
            if  len(idx_comp):
                states_A = states[idx_comp, colp[0], 0:3]
                states_B = states[idx_comp, colp[1], 0:3]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] -states_A[:,:2]
                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], R[idx_comp,:,:])
                force_B_0, torque_A_0, torque_B_0 = get_contact_wrenches(P_tot, 
                                                                         D_tot, 
                                                                         S_mat, 
                                                                         Repf_mat[idx_comp, ...],
                                                                         Ds,
                                                                         verts_tf,
                                                                         len(idx_comp),
                                                                         zero_pad[idx_comp, ...],
                                                                         stiffness
                                                                         )

                #rotate forces into global frame from frame A
                force_B_0 = rotate_vec(force_B_0, states_A[:, 2], R[idx_comp,:,:])
                contact_wrenches[ idx_comp, colp[0], :2] += -force_B_0
                contact_wrenches[ idx_comp, colp[1], :2] += force_B_0
                contact_wrenches[ idx_comp, colp[0], 2] += torque_A_0
                contact_wrenches[ idx_comp, colp[1], 2] += torque_B_0

                #flip and check other direction
                states_A = states[idx_comp, colp[1], 0:3]
                states_B = states[idx_comp, colp[0], 0:3]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] - states_A[:,:2]
                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], R[idx_comp,:,:])

                force_B_1, torque_A_1, torque_B_1 = get_contact_wrenches(P_tot, 
                                                                         D_tot, 
                                                                         S_mat, 
                                                                         Repf_mat[idx_comp, ...],
                                                                         Ds,
                                                                         verts_tf,
                                                                         len(idx_comp),
                                                                         zero_pad[idx_comp, ...],
                                                                         stiffness
                                                                         )

                #rotate forces into global frame from frame A
                col_active = torch.abs(contact_wrenches[ idx_comp, colp[1], 2]) > 0.01
                new_col = (1.0*~col_active).view(-1,1)
                already_col = (col_active).view(-1,1)
                force_B_1 = rotate_vec(force_B_1, states_A[:, 2], R[idx_comp,:,:])
                new_col_active = (torch.abs(torque_B_1) > 0.01).view(-1,1)

                contact_wrenches[ idx_comp, colp[1], :2] += -force_B_1 * (1.0 * new_col + 0.5*already_col) + (1.0*already_col-0.5*new_col_active)*force_B_0
                contact_wrenches[ idx_comp, colp[0], :2] += force_B_1 * (1.0 * new_col + 0.5*already_col) - (1.0*already_col-0.5*new_col_active)*force_B_0
                contact_wrenches[ idx_comp, colp[1], 2] += (torque_A_1.view(-1,1)*(1.0 * new_col + 0.5*already_col) + (1.0*already_col-0.5*new_col_active)*torque_B_0.view(-1,1)).view(-1)
                contact_wrenches[ idx_comp, colp[0], 2] += (torque_B_1.view(-1,1)*(1.0 * new_col + 0.5*already_col) + (1.0*already_col-0.5*new_col_active)*torque_A_0.view(-1,1)).view(-1)
                
                #shove[idx_comp, colp[0], :2] += 0.6*contact_wrenches[ idx_comp, colp[0], :2]/stiffness
                #shove[idx_comp, colp[1], :2] += 0.6*contact_wrenches[ idx_comp, colp[1], :2]/stiffness
    shove = 0.6*contact_wrenches[:,:,:2]/stiffness            
    return contact_wrenches, shove

