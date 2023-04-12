import torch
import numpy as np
from typing import List, Tuple, Dict
from dmaracing.env.throttle_encoder import get_throttle_encoder 
import torch.nn as nn 
from torch import device

def torch_unif_rand(shape, min, max, device):
    rnd = torch.rand(shape, device=device)
    return (max-min) * rnd + min

def get_varnames()->Dict[str, int]:
    varnames = {}
    varnames['S_X'] = 0
    varnames['S_Y'] = 1
    varnames['S_THETA'] = 2
    varnames['S_DX'] = 3
    varnames['S_DY'] = 4
    varnames['S_DTHETA'] = 5
    varnames['S_STEER'] = 6
    #varnames['S_W0'] = 6
    #varnames['S_W1'] = 7
    #varnames['S_W2'] = 8
    #varnames['S_W3'] = 9
    # varnames['S_GAS'] = 7
    
    varnames['A_STEER'] = 0
    varnames['A_GAS'] = 1
    #varnames['A_BRAKE'] = 2
    return varnames

def allocate_car_dynamics_tensors(task):
    task.R = torch.zeros((task.num_envs, task.num_agents, 2, 2), dtype = torch.float, device = task.device, requires_grad=False)
    task.zero_pad = torch.zeros((task.num_envs, 5,1), device =task.device, requires_grad=False) 

    task.P_tot, task.D_tot, task.S_mat, task.Repf_mat, task.Ds = build_col_poly_eqns(1.1*task.modelParameters['W_coll'], 1.1*(task.modelParameters['L_coll']), task.device, task.num_envs)
    task.collision_pairs = get_collision_pairs(task.num_agents, task.cfg['sim']['filtercollisionstoego'], task.team_size)
    task.collision_verts = get_car_vert_mat(task.modelParameters['W_coll'], 
                                            #task.modelParameters['lf']+task.modelParameters['lr'], 
                                            task.modelParameters['L_coll'],
                                            task.num_envs, 
                                            task.device)
    L = task.modelParameters['L_coll']
    W = task.modelParameters['W_coll']
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
    mod_par['L'] = mod_par['L_coll'] 
    
#only used for collsion checking, only one car per env needed for pairwise checking
def get_car_vert_mat(w, l, num_envs, device):
    verts = torch.zeros((num_envs, 5, 2), device=device, dtype=torch.float, requires_grad=False)
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
    verts[:, 4, 0] = l/2
    verts[:, 4, 1] = 0
    return verts

def get_collision_pairs(num_agents, filtercollisionpairstoego=False, team_size=None):
    if filtercollisionpairstoego:
        ls = np.arange(num_agents)
        # pairs = [[0, a] for a in ls[1:]]
        pairs = [[a, b] for idx, a in enumerate(ls[:team_size]) for b in ls[idx + 1:]] 
    else:
    #naively check all collision pairs
        ls = np.arange(num_agents)
        pairs = [[a, b] for idx, a in enumerate(ls) for b in ls[idx + 1:]]
    return pairs

def get_collision_pairs2(num_agents):
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
def transform_vel_verts(rel_vel_global : torch.Tensor, 
                        theta_A : torch.Tensor, 
                        theta_B : torch.Tensor, 
                        verts: torch.Tensor,
                        zero_pad: torch.Tensor,
                        R: torch.Tensor) ->torch.Tensor:
    '''
    collider -> (B) Vertex rep
    collidee -> (A) Poly rep
    input: 
    rel_vels_global (num_envs, 2) :relative velocities pB-pA in global frame 
    theta_A (num_envs) : relative rotation of A to world
    theta_B (num_envs) : relative rotation of B to world
    verts : vertex tensor (numenvs, 4 car vertices, 2 cords)
    R: allocated rot mat (numenvs, 2, 2) 
    output:
    vel_verts_rot_shift (num_envs, 4 car vertices, 2): velocities of vertices of B in A frame
    '''
    omega = torch.cat((zero_pad[:,0,:], zero_pad[:,0,:], rel_vel_global[..., -1].reshape(-1,1)), dim =-1)
    vel_verts_global = torch.cross(torch.tile(omega.unsqueeze(1), (1,5,1)), torch.cat((verts, zero_pad), dim = 2))[..., 0:2] + rel_vel_global[..., 0:2].reshape(-1,1,2) 
    
    theta_rel = theta_B - theta_A
    R[:, 0, 0 ] = torch.cos(theta_rel)
    R[:, 0, 1 ] = -torch.sin(theta_rel)
    R[:, 1, 0 ] = torch.sin(theta_rel)
    R[:, 1, 1 ] = torch.cos(theta_rel)
    vel_verts_Aframe = torch.einsum('kij, klj->kli', R, vel_verts_global)
   
    return vel_verts_Aframe

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
    verts_rot_shift[:,4,:] += trans_rot[:]
    
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
                         rel_vels : torch.Tensor,
                         num_envs : int,
                         zero_pad : torch.Tensor,
                         stiffness : float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #evaluate polygon equations on vertices of collider in bodyframe of collidee
    vert_poly_dists = torch.einsum('ij, lkj->lki', P_tot, verts_tf) + torch.tile(D_tot.squeeze(), (num_envs, 5, 1)) 
    #check if in polygon
    in_poly = torch.einsum('ij, lkj -> lki', S_mat, 1.0*(vert_poly_dists+1e-4>=0)) > 3 #watch for numerics here #!$W%!$#!
    #inpoly (numenvs, num_verts, num_poly) Repforcedir (num_envs, num_poly, coords) -> num_env, num_verts, forcecoords
    force_dir = torch.einsum('ijk, ikl -> ijl', 1.0*in_poly, Repf_mat)
    tangential_force_dir =torch.einsum('ijk, ikl -> ijl', 1.0*in_poly, Repf_mat[:,[2,3,1,0], :]) 
    
    #relative velocity projected onto tangential direction
    fric_vels = torch.einsum('evd, evd -> ev', tangential_force_dir, rel_vels).view(-1, 5,1)
    pos_fric_vels = 1.0*(fric_vels>0)
    neg_fric_vels = -1.0*(fric_vels<0)
    
    fric_forces = -tangential_force_dir*(pos_fric_vels + neg_fric_vels)*0.05
    #inpoly (numenvs, num_verts, num_poly) #vert_poly_dists[:,:, Ds] (num_env, num_verts, num_polygons)
    #use fact that we precomputed all distances to polygon walls
    depths = torch.einsum('ijk, ijk -> ij', 1.0*in_poly, vert_poly_dists[:,:, Depth_selector])
    magnitude = stiffness*depths
    forces = force_dir
    forces[:, :, 0] *= magnitude
    forces[:, :, 1] *= magnitude
    forces += fric_forces

    dirs_vert = torch.cat((verts_tf - torch.tile(torch.mean(verts_tf, dim = 1).unsqueeze(1), (1,5,1)), zero_pad ), dim = 2)
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
                      stiffness : float,
                      Iz : float
                      ) -> Tuple[torch.Tensor, torch.Tensor]:

    contact_wrenches[:,:,:] =0.0
    shove[:,:,:] = 0.0
    #switch = torch.rand(len(collision_pairs))>0.5
    if len(collision_pairs):
        for id, colp in enumerate(collision_pairs):
            #colp = [colp[switch[id]*1], colp[~switch[id]*1]]
            idx_comp = torch.where(torch.norm(states[:, colp[0], 0:2] -  states[:, colp[1], 0:2], dim =1)<=2.4*lf)[0]
            
            if  len(idx_comp):
                states_A = states[idx_comp, colp[0], 0:3]
                states_B = states[idx_comp, colp[1], 0:3]

                vels_A = states[idx_comp, colp[0], 3:6]
                vels_B = states[idx_comp, colp[1], 3:6]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] -states_A[:,:2]
                rel_vels = vels_B - vels_A

                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], R[idx_comp,:,:])
                vels_verts_tf = transform_vel_verts(rel_vels, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], zero_pad[idx_comp, :, :], R[idx_comp,:,:])

                force_B_0, torque_A_0, torque_B_0 = get_contact_wrenches(P_tot, 
                                                                         D_tot, 
                                                                         S_mat, 
                                                                         Repf_mat[idx_comp, ...],
                                                                         Ds,
                                                                         verts_tf,
                                                                         vels_verts_tf,
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

                vels_A = states[idx_comp, colp[1], 3:6]
                vels_B = states[idx_comp, colp[0], 3:6]

                #get contact wrenches for collision pair candidate
                rel_trans = states_B[:,:2] - states_A[:,:2]
                rel_vels = vels_B - vels_A

                verts_tf = transform_col_verts(rel_trans, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], R[idx_comp,:,:])
                vels_verts_tf = transform_vel_verts(rel_vels, states_A[:,2], states_B[:,2], collision_verts[idx_comp, :, :], zero_pad[idx_comp, :, :], R[idx_comp,:,:])

                force_B_1, torque_A_1, torque_B_1 = get_contact_wrenches(P_tot, 
                                                                         D_tot, 
                                                                         S_mat, 
                                                                         Repf_mat[idx_comp, ...],
                                                                         Ds,
                                                                         verts_tf,
                                                                         vels_verts_tf,
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

                contact_wrenches[ idx_comp, colp[1], :2] += -force_B_1 * (1.0 * new_col + 0.5*already_col) + (-0.5*new_col_active)*force_B_0
                contact_wrenches[ idx_comp, colp[0], :2] += force_B_1 * (1.0 * new_col + 0.5*already_col) - (-0.5*new_col_active)*force_B_0
                contact_wrenches[ idx_comp, colp[1], 2] += (torque_A_1.view(-1,1)*(1.0 * new_col + 0.5*already_col) + (-0.5*new_col_active)*torque_B_0.view(-1,1)).view(-1)
                contact_wrenches[ idx_comp, colp[0], 2] += (torque_B_1.view(-1,1)*(1.0 * new_col + 0.5*already_col) + (-0.5*new_col_active)*torque_A_0.view(-1,1)).view(-1)
            
    shove[:, :, :2] = 1.0 * contact_wrenches[:,:,:2] / stiffness
    shove[:, :, 2]  = .5 * contact_wrenches[:,:,2] / (stiffness*Iz)          
    return contact_wrenches, shove

class SwitchedBicycleKinodynamicModel(nn.Module):
    def __init__(self,
                 sim_cfg,
                 model_cfg,
                 vn = None,
                 device = 'cuda:0'
                 ):

        super().__init__()
        self.num_envs = sim_cfg['numEnv']
        self.num_states = sim_cfg['numStates']
        self.num_actions = sim_cfg['numActions']
        self.num_agents = sim_cfg['numAgents']
        self.device = device
        #self.parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float))
        self.dt = sim_cfg['dt']
        self.max_steer_rate = model_cfg['max_steer_rate']#2.1 #rad/s
        self.max_d_steer = self.dt*self.max_steer_rate
        self.max_vel = model_cfg['max_vel']
        self.M = model_cfg['mass']
        self.lf = model_cfg['lf']
        self.lr = model_cfg['lr']
        self.Iz = model_cfg['Iz']
        self.L = self.lf + self.lr
        self.lr_noise = 0
        self.lf_noise = 0
        self.gas_noise = 0
        self.steering_offset_noise = 0
        self.max_vel_vec = self.max_vel
        
        self.throttle_encoder = get_throttle_encoder()
        self.throttle_encoder.to(self.device)
        self.test = False
        self.use_collisions = False
        self.Bf = model_cfg['Bf']
        self.Br = model_cfg['Br']
        self.Cf = model_cfg['Cf']
        self.Cr = model_cfg['Cr']
        self.Df = model_cfg['Df']
        self.Dr = model_cfg['Dr']

        self.vm_noise_scale_ego = model_cfg['vm_noise_scale_ego']
        self.vm_noise_scale_ado = model_cfg['vm_noise_scale_ado']
        self.lf_noise_scale = model_cfg['lf_noise_scale']
        self.lr_noise_scale = model_cfg['lr_noise_scale']
        self.steering_offset_noise_scale = model_cfg['steering_offset_noise_scale']
        self.gas_noise_scale = model_cfg['gas_noise_scale']

        self.time_since_col = 100.*torch.ones((self.num_envs, self.num_agents), dtype=torch.float, device= device)
        self.col_decay_time = model_cfg['col_decay_time']

        self.vn = vn
        #BLR ModelParameters for TRI KART
        self.alphaf_model_mean = torch.tensor(model_cfg['alpha_f_mean'], device=self.device)
        self.alphaf_model_sigma = torch.tensor(model_cfg['alpha_f_sigma'], device=self.device)
        
        self.init_noise_vec(self.num_envs)
        if sim_cfg['collide']:
            self.init_col_switch(self.num_envs, model_cfg['col_decay_time'], self.device)
        
        self.randomized_params = ['max_vel'] if self.test else ['max_vel', 'lf', 'lb', 'steering_offset', 'throttlecmd', 'w_blr']
  
    def set_parameters(self, par, vn):
        self.par = par
        self.vn = vn

    def get_state_derivative(self, state, actions, col_wrenches):
        # https://onlinelibrary.wiley.com/doi/pdf/10.1002/oca.2123
        # 
        # [x, y, theta, xdot, ydot, thetadot]
        #fitted steering model
        steer = 1.5739 *torch.clip(actions[:, :, self.vn['A_STEER']], -0.35, 0.35) - 0.044971
        steer = torch.clip(steer-self.last_steer, -self.max_d_steer, self.max_d_steer) + steer

        fast_fwd = state[:, :, self.vn['S_DX']] > self.max_vel_vec.squeeze()
        fast_bwd = -state[:, :, self.vn['S_DX']] > 0.1  # 0.01
        gas = actions[:, :, self.vn['A_GAS']]
        gas_clip = 1.0*fast_fwd*torch.clip(gas, -1,0) + 1.0*fast_bwd*torch.clip(gas, 0,1) + ~(fast_fwd|fast_bwd) *  torch.clip(gas, -1.0, 0.3)
        gas_features = torch.cat((state[:, :, self.vn['S_DX']].unsqueeze(2), 
                                  actions[:, :, self.vn['A_STEER']].unsqueeze(2), 
                                  gas_clip.unsqueeze(2)), dim = -1)
        
        gas_features += self.gas_noise
        
        ddx = self.throttle_encoder(gas_features).detach()

        v_bax = state[:, :, self.vn['S_DX']]
        v = v_bax.view(-1,self.num_agents,1)
        d = steer.view(-1,self.num_agents,1)
        X = torch.cat((d,
                        d*v, 
                        d**3), dim = -1)

        slip_angle = -torch.einsum('eac, epc -> ea', X, self.w_sample)
        beta = torch.arctan((self.lr+self.lr_noise)*torch.tan(steer + self.steering_offset_noise+slip_angle)/(self.L+self.lf_noise + self.lr_noise))
        ddy = 0*ddx
        dx = v_bax*torch.cos(state[:, :, self.vn['S_THETA']] + beta)
        dy = v_bax*torch.sin(state[:, :, self.vn['S_THETA']] + beta)
        
        #evaluate BLR model
        angle = steer + self.steering_offset_noise + slip_angle 
        dtheta = v_bax*(torch.tan(angle)/torch.sqrt((self.L+self.lf_noise + self.lr_noise)**2 + ((self.lr+self.lr_noise)*torch.tan(angle))**2))
        
        ddtheta = 0 * ddx
        
        ddelta = 0 * ddtheta

        dstate = torch.cat((dx.view(-1,self.num_agents, 1),
                            dy.view(-1,self.num_agents, 1),
                            dtheta.view(-1,self.num_agents, 1),
                            ddx.view(-1,self.num_agents, 1),
                            ddy.view(-1,self.num_agents, 1),
                            ddtheta.view(-1,self.num_agents, 1),
                            ddelta.view(-1,self.num_agents, 1)
                            ), dim=2)

        #dynamic part
        if self.use_collisions:
            lf_perturbed = self.lf + self.lf_noise
            lr_perturbed = self.lr + self.lr_noise
            steer = steer.view(-1,self.num_agents)
            alphaf = -torch.atan2(state[:, :, self.vn['S_DTHETA']]*lf_perturbed + state[:, :, self.vn['S_DY']], state[:, :, self.vn['S_DX']] ) + steer
            alphar = torch.atan2(state[:, :, self.vn['S_DTHETA']]*lr_perturbed - state[:, :, self.vn['S_DY']], state[:, :, self.vn['S_DX']])
            Ffy = self.Df * torch.sin(self.Cf*torch.atan(self.Bf*alphaf)) 
            Fry = self.Dr * torch.sin(self.Cr*torch.atan(self.Br*alphar))  
            Frx = ddx.squeeze()*self.M*0.5
            Ffx = ddx.squeeze()*self.M*0.5
            dx_dyn = state[:, :, self.vn['S_DX']] * torch.cos(state[:, :, self.vn['S_THETA']]) - \
            state[:, :, self.vn['S_DY']] * torch.sin(state[:, :, self.vn['S_THETA']])
            dy_dyn = state[:, :, self.vn['S_DX']] * torch.sin(state[:, :, self.vn['S_THETA']]) + \
                state[:, :, self.vn['S_DY']] * torch.cos(state[:, :, self.vn['S_THETA']])
            dtheta_dyn = state[:, :, self.vn['S_DTHETA']]

            ddx_dyn = 1 / self.M * ( Frx + Ffx*torch.cos(steer) - Ffy*torch.sin(steer)  + self.M * (state[:, :, self.vn['S_DY']]*state[:, :, self.vn['S_DTHETA']])  \
                            +   (torch.cos(state[:, :, self.vn['S_THETA']]) * col_wrenches[:, :, 0] + torch.sin(state[:, :, self.vn['S_THETA']])* col_wrenches[:, :, 1]  \
                            - 0.1  * (state[:, :, self.vn['S_DX']]>0) +  0.1  * (state[:, :, self.vn['S_DX']]<0)) )

            ddy_dyn = 1 / self.M * (Fry + Ffx*torch.sin(steer) + Ffy * torch.cos(steer) - self.M * (state[:, :, self.vn['S_DX']] * state[:, :, self.vn['S_DTHETA']])
                                - (torch.sin(state[:, :, self.vn['S_THETA']]) * col_wrenches[:, :, 0] + torch.cos(state[:, :, self.vn['S_THETA']]) * col_wrenches[:, :, 1]))

            ddtheta_dyn = 1 / self.Iz * \
                (Ffy * lf_perturbed* torch.cos(steer) - Fry * lr_perturbed + 5.0*col_wrenches[:, :, 2])
            ddelta_dyn = 0 * ddtheta

            dstate_dyn = torch.cat((dx_dyn.view(-1,self.num_agents, 1),
                                dy_dyn.view(-1,self.num_agents, 1),
                                dtheta_dyn.view(-1,self.num_agents, 1),
                                ddx_dyn.view(-1,self.num_agents, 1),
                                ddy_dyn.view(-1,self.num_agents, 1),
                                ddtheta_dyn.view(-1,self.num_agents, 1),
                                ddelta_dyn.view(-1,self.num_agents, 1)
                                ), dim=2)

            is_col = torch.any(col_wrenches, dim =-1)
            
            self.time_since_col *= (~is_col).view(-1, self.num_agents)  
            model_switch = ((self.time_since_col < self.col_decay_time)*(torch.abs(alphar).view(-1, self.num_agents)>0.1)).view(-1,self.num_agents,1)

            dstate = model_switch*dstate_dyn + ~model_switch*dstate
            self.time_since_col += self.dt

        self.last_steer = steer.view(-1, self.num_agents)

        return dstate

    def forward(self, state, actions, col_wrenches, shove):
        d_state = self.get_state_derivative(state, actions, col_wrenches)
        next_state = state + self.dt * d_state
        next_state[..., self.vn['S_X']:self.vn['S_Y'] + 1] += shove[...,:2]  #  * self.dt/0.02 (maybe?)
        next_state[:, :, self.vn['S_THETA']] += 0.8*shove[:,:,2]  #  * self.dt/0.02 (maybe?)
        return next_state
    
    def update_noise_vec(self, envs, noise_level, team_size):
        if not self.test:
            self.lf_noise[envs] = torch_unif_rand((len(envs), 1), -self.lf_noise_scale*noise_level, self.lf_noise_scale*noise_level, device=self.device)
            self.lr_noise[envs] = torch_unif_rand((len(envs), 1), -self.lr_noise_scale*noise_level, self.lr_noise_scale*noise_level, device=self.device)
            self.steering_offset_noise[envs] = torch_unif_rand((len(envs), 1 ), -self.steering_offset_noise_scale*noise_level, self.steering_offset_noise_scale*noise_level, device=self.device)
            self.gas_noise[envs] = torch_unif_rand((len(envs), self.num_agents, 1), -self.gas_noise_scale[0]*noise_level, self.gas_noise_scale[1]*noise_level, device=self.device)
        # self.max_vel_vec[envs] = self.max_vel*(1 - self.vm_noise_scale*noise_level) + torch_unif_rand((len(envs), self.num_agents, 1), 0, self.vm_noise_scale*self.max_vel*noise_level, device=self.device)
        self.max_vel_vec[envs, :team_size] = self.max_vel*(1 - self.vm_noise_scale_ego*noise_level) + torch_unif_rand((len(envs), team_size, 1), 0, self.vm_noise_scale_ego*self.max_vel*noise_level, device=self.device)
        self.max_vel_vec[envs, team_size:] = self.max_vel*(1 - self.vm_noise_scale_ado*noise_level) + torch_unif_rand((len(envs), self.num_agents-team_size, 1), 0, self.vm_noise_scale_ado*self.max_vel*noise_level, device=self.device)

    def init_noise_vec(self, num_envs):
        self.lf_noise = torch.zeros((num_envs,  self.num_agents, ), dtype=torch.float, device= self.device)
        self.lr_noise = torch.zeros((num_envs,  self.num_agents, ), dtype=torch.float, device= self.device)
        self.gas_noise = torch.zeros((num_envs, self.num_agents, 1), dtype=torch.float, device= self.device)
        
        self.steering_offset_noise = torch.zeros((num_envs, self.num_agents, ), dtype=torch.float, device= self.device)
        self.max_vel_vec = self.max_vel*torch.ones((num_envs, self.num_agents, 1), dtype=torch.float, device= self.device)
        self.alphaf_model_mean = torch.tile(self.alphaf_model_mean, (num_envs, 1)).view(-1,1, len(self.alphaf_model_mean))
        self.w_sample = self.alphaf_model_mean.clone()
        self.last_steer = torch.zeros((num_envs, self.num_agents), dtype=torch.float, device= self.device)

    def init_col_switch(self, num_envs, col_decay_time, device):
        self.use_collisions = True
        self.device = device
        self.time_since_col = 100.*torch.ones((num_envs, self.num_agents), dtype=torch.float, device= device)
        self.col_decay_time = col_decay_time

    def set_test_mode(self, ):
        self.test = True
    
    def reset(self, envs):
        if not self.test:
            self.w_sample[envs] = torch.distributions.MultivariateNormal(self.alphaf_model_mean[envs, ...], self.alphaf_model_sigma).sample().view(-1, 1, self.alphaf_model_mean.shape[2])
        if self.use_collisions:
            self.time_since_col[envs, ...] = 100.0
        self.last_steer[envs] = 0