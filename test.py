import torch
import numpy as np

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

def build_col_poly_eqns(w, l, device):
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
    P_tot[8, 0] = -1
    P_tot[9, 0] = 1
    P_tot[10, 0] = 1/np.sqrt(2)
    P_tot[10, 1] = 1/np.sqrt(2)
    P_tot[11, 0] = 1/np.sqrt(2)
    P_tot[11, 1] = -1/np.sqrt(2)

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
    D_tot[10] = -l/(2*np.sqrt(2)) + w/(2*np.sqrt(2))
    D_tot[11] = l/(2*np.sqrt(2)) + w/(2*np.sqrt(2))
    #D4
    D_tot[12:16, 0] = D_tot[8:12, 0]

    S_mat = torch.eye(4, device=device, dtype=torch.float, requires_grad = False)
    tmp = torch.ones((1,4), device=device, dtype=torch.float, requires_grad = False)
    S_mat = torch.kron(S_mat, tmp)
    return P_tot, D_tot, S_mat

def transform_col_verts(rel_trans_global : torch.Tensor, 
                        theta_A : torch.Tensor, 
                        theta_B : torch.Tensor, 
                        verts: torch.Tensor) ->torch.Tensor:
    '''
    (B collider vertex rep, A collidee poly rep)
    input: 
    rel_trans_global (num_envs, 2) :relative translation pB-pA in global frame 
    theta_A (num envs) : relative rotation of A to world
    theta_B (num envs) : relative rotation of B to world
    verts : vertex tensor (numenvs, 4 car vertices, 2 cords) 
    '''
    theta_rel = theta_B - theta_A
    R = torch.zeros((len(theta_rel), 2, 2), device = theta_rel.device)
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
    return verts_rot

def get_contact_forces(P_tot, D_tot, verts_rot):
    cf = torch.zeros((num_envs, 4, 2), device=device, dtype=torch.float, requires_grad=False)
    


device = 'cuda:0'
num_envs = 2000
w = 0.5
l = 1
verts = get_car_vert_mat(w, l, num_envs, device)
theta_a = 0*torch.ones((num_envs), device=device, dtype=torch.float, requires_grad=False)
theta_b = np.pi/2 *torch.ones((num_envs), device=device, dtype=torch.float, requires_grad=False)
trans = torch.ones((num_envs,2), device=device, dtype=torch.float, requires_grad=False)
trans[:, 0] = -0.75
trans[:, 1] = 0.75


verts_tf = transform_col_verts(trans, theta_A = theta_a, theta_B=theta_b, verts = verts)
P_tot, D_tot, S_mat = build_col_poly_eqns(w, l, device)
testpt = torch.ones((2, 1), device=device, dtype=torch.float, requires_grad=False)
testpt[0,0] =-0.5
testpt[1,0] = 1.25
print(P_tot@testpt +  D_tot)
#"p(polyg, 2) verts (numenv, 4, 2) "
vert_poly_dists = torch.einsum('ij, lkj->lki', P_tot, verts_tf) #+ torch.tile(D_tot.squeeze(), (num_envs, 4, 1)) 


in_poly = torch.einsum('ij, lkj -> lki', S_mat, 1.0*(vert_poly_dists>=0)) == 4

print(in_poly[0,:,:])


print('done')

