#track generator copied from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
from typing import overload

import numpy as np
import math
import cv2 as cv
from scipy.sparse import coo_matrix
import torch
 
def get_track(cfg, device, ccw = True):
    SCALE = cfg['track']['SCALE'] # Track scale
    TRACK_RAD = cfg['track']['TRACK_RAD'] / SCALE  # Track is heavily morphed circle with this radius
    CHECKPOINTS = cfg['track']['CHECKPOINTS']
    TRACK_DETAIL_STEP = cfg['track']['TRACK_DETAIL_STEP'] / SCALE
    TRACK_TURN_RATE = cfg['track']['TRACK_TURN_RATE']
    TRACK_WIDTH = cfg['track']['TRACK_WIDTH'] / SCALE
    BORDER = cfg['track']['BORDER'] / SCALE
    BORDER_MIN_COUNT = cfg['track']['BORDER_MIN_COUNT'] 
    verbose = cfg['track']['verbose']
    seed = cfg['track']['seed']
    width = cfg['viewer']['width']
    height = cfg['viewer']['height']

    np.random.seed(seed)
    # Create checkpoints
    checkpoints = []
    for c in range(CHECKPOINTS):
        noise = np.random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
        alpha = 2 * math.pi * c / CHECKPOINTS + noise
        rad = np.random.uniform(TRACK_RAD / 3, TRACK_RAD)

        if c == 0:
            alpha = 0
            rad = 1.5 * TRACK_RAD
        if c == CHECKPOINTS - 1:
            alpha = 2 * math.pi * c / CHECKPOINTS
            start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
            rad = 1.5 * TRACK_RAD

        checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
    road = []

    # Go from one checkpoint to another to create track
    x, y, beta = 1.5 * TRACK_RAD, 0, 0
    dest_i = 0
    laps = 0
    track = []
    no_freeze = 2500
    visited_other_side = False
    while True:
        alpha = math.atan2(y, x)
        if visited_other_side and alpha > 0:
            laps += 1
            visited_other_side = False
        if alpha < 0:
            visited_other_side = True
            alpha += 2 * math.pi

        while True:  # Find destination from checkpoints
            failed = True

            while True:
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                if alpha <= dest_alpha:
                    failed = False
                    break
                dest_i += 1
                if dest_i % len(checkpoints) == 0:
                    break

            if not failed:
                break

            alpha -= 2 * math.pi
            continue

        r1x = math.cos(beta)
        r1y = math.sin(beta)
        p1x = -r1y
        p1y = r1x
        dest_dx = dest_x - x  # vector towards destination
        dest_dy = dest_y - y
        # destination vector projected on rad:
        proj = r1x * dest_dx + r1y * dest_dy
        while beta - alpha > 1.5 * math.pi:
            beta -= 2 * math.pi
        while beta - alpha < -1.5 * math.pi:
            beta += 2 * math.pi
        prev_beta = beta
        proj *= SCALE
        if proj > 0.3:
            beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
        if proj < -0.3:
            beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
        x += p1x * TRACK_DETAIL_STEP
        y += p1y * TRACK_DETAIL_STEP
        track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
        if laps > 4:
            break
        no_freeze -= 1
        if no_freeze == 0:
            break

    # Find closed loop range i1..i2, first loop should be ignored, second is OK
    i1, i2 = -1, -1
    i = len(track)
    while True:
        i -= 1
        #if i == 0:
        #    return False  # Failed
        pass_through_start = (
            track[i][0] > start_alpha and track[i - 1][0] <= start_alpha
        )
        if pass_through_start and i2 == -1:
            i2 = i
        elif pass_through_start and i1 == -1:
            i1 = i
            break
    if verbose == 1:
        print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
    assert i1 != -1
    assert i2 != -1

    track = track[i1 : i2 - 1]

    first_beta = track[0][1]
    first_perp_x = math.cos(first_beta)
    first_perp_y = math.sin(first_beta)
    # Length of perpendicular jump to put together head and tail
    well_glued_together = np.sqrt(
        np.square(first_perp_x * (track[0][2] - track[-1][2]))
        + np.square(first_perp_y * (track[0][3] - track[-1][3]))
    )
    #if well_glued_together > TRACK_DETAIL_STEP:
    #    return False

    # Red-white border on hard turns
    border = [False] * len(track)
    centerline = np.zeros((len(track), 2))
    alphas = []

    track_poly_verts = []
    border_poly_verts = []
    border_poly_col = []

    for i in range(len(track)):
        good = True
        oneside = 0
        for neg in range(BORDER_MIN_COUNT):
            beta1 = track[i - neg - 0][1]
            beta2 = track[i - neg - 1][1]
            good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
            oneside += np.sign(beta1 - beta2)
        good &= abs(oneside) == BORDER_MIN_COUNT
        border[i] = good
    for i in range(len(track)):
        for neg in range(BORDER_MIN_COUNT):
            border[i - neg] |= border[i]

    # Create tiles
    for i in range(len(track)):
        alpha1, beta1, x1, y1 = track[i]
        alpha2, beta2, x2, y2 = track[i - 1]
        road1_l = (
            x1 - TRACK_WIDTH * math.cos(beta1),
            y1 - TRACK_WIDTH * math.sin(beta1),
        )
        road1_r = (
            x1 + TRACK_WIDTH * math.cos(beta1),
            y1 + TRACK_WIDTH * math.sin(beta1),
        )
        road2_l = (
            x2 - TRACK_WIDTH * math.cos(beta2),
            y2 - TRACK_WIDTH * math.sin(beta2),
        )
        road2_r = (
            x2 + TRACK_WIDTH * math.cos(beta2),
            y2 + TRACK_WIDTH * math.sin(beta2),
        )
        vert = np.zeros((4,2))
        vert[0,:] = road1_l
        vert[1,:] = road1_r
        vert[2,:] = road2_r
        vert[3,:] = road2_l
        track_poly_verts.append(vert)
        centerline[i,0] = track[i][2]
        centerline[i,1] = track[i][3]            
        alphas.append(track[i][1])
        
        
        if border[i]:
            side = np.sign(beta2 - beta1)
            b1_l = (
                x1 + side * TRACK_WIDTH * math.cos(beta1),
                y1 + side * TRACK_WIDTH * math.sin(beta1),
            )
            b1_r = (
                x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
            )
            b2_l = (
                x2 + side * TRACK_WIDTH * math.cos(beta2),
                y2 + side * TRACK_WIDTH * math.sin(beta2),
            )
            b2_r = (
                x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
            )

            vert = np.zeros((4,2))
            vert[0,:] = b1_l
            vert[1,:] = b1_r
            vert[2,:] = b2_r
            vert[3,:] = b2_l
            border_poly_verts.append(vert)
            border_poly_col.append((255, 255, 255) if i % 2 == 0 else (0, 0, 255))
            #road_poly.append(
            #    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
            #)

    if ccw:    
        track_poly_verts = np.array(track_poly_verts)
        border_poly_verts = np.array(border_poly_verts)
        alphas = np.array(alphas)
    else:
        track_poly_verts = np.array(track_poly_verts)[::-1]
        border_poly_verts = np.array(border_poly_verts)[::-1]
        alphas = np.array(alphas[::-1]) + np.pi 
        centerline = centerline[::-1]

    A, b, S_mat = construct_poly_track_eqns(track_poly_verts, device)
    return [centerline, track_poly_verts, alphas, A, b, S_mat, border_poly_verts, border_poly_col], TRACK_DETAIL_STEP, len(track_poly_verts)

def draw_track(img, track, cords2px, cl = False):
    centerline = track[0].copy()
    track_poly_verts = track[1].copy()
    border_poly_verts = track[6].copy()
    border_poly_col = track[7]
    
    img[:, :, 0] = 130
    img[:, :, 1] = 255
    img[:, :, 2] = 130
    
    overlay = img.copy()
    for idx in range(len(track_poly_verts)):
        verts = track_poly_verts[idx, :, :]
        vert_px = cords2px(verts)
        cv.fillPoly(img, [vert_px], color = (178,178,178) if idx%2 ==0 else (168,168,168))
        cv.polylines(overlay, [vert_px], isClosed = True,  color = (0,0,0), thickness = 1)
    
    
    verts = track[1][0, :, :].copy()
    vert_px = cords2px(verts)
    cv.polylines(overlay, [vert_px], isClosed = True,  color = (0,0 ,255), thickness = 3)
    
    #img = cv.addWeighted(underlay, 0.3, img, 0.9, 0)
    if cl:
        cl_px = cords2px(centerline)
        cv.polylines(img, [cl_px], isClosed = True, color = (0,0,0), thickness = 1)
        num_cl = len(cl_px)
        for idx in range(num_cl):
            cv.circle(img, (cl_px[idx, 0], cl_px[idx, 1]), 1, (0,0,int(idx/num_cl *255)))

    for idx in range(len(border_poly_verts)):
        verts = border_poly_verts[idx, :, :]
        vert_px = cords2px(verts)
        cv.fillPoly(img, [vert_px], color = border_poly_col[idx])
    
    img = cv.addWeighted(overlay, 0.01, img, 0.99, 0)
      
    draw_cord_axs(img, cords2px)
    
def draw_cord_axs(img, cords2px):
    xverts = np.array([[-10,0],[10,0]])
    yverts = np.array([[0,10],[0,-10]])
    xverts = cords2px(xverts)
    yverts = cords2px(yverts)

    cv.polylines(img, [xverts], isClosed = True,  color = (0,0,0), thickness = 2)
    cv.polylines(img, [yverts], isClosed = True,  color = (0,0,0), thickness = 2)


def construct_poly_track_eqns(track_poly_verts, device):
    A = torch.zeros((4*len(track_poly_verts), 2), device = device, dtype = torch.float, requires_grad = False)
    b = torch.zeros((len(track_poly_verts)*4,), device= device, dtype = torch.float, requires_grad = False)
    order  = [[0,1], [1,2], [2,3], [3,0]]
    for idx in range(len(track_poly_verts)):
        verts = track_poly_verts[idx,:,:]
        pair_idx = 0
        for pair in order:
            vert_pair = verts[pair,:]
            diff = vert_pair[1,:] - vert_pair[0,:]
            diff_rot = diff.copy()
            diff_rot[0] = diff_rot[1]
            diff_rot[1] = -diff[0]  
            A[4*idx + pair_idx, 0] = diff_rot[0]
            A[4*idx + pair_idx, 1] = diff_rot[1]
            b[4*idx + pair_idx] = diff_rot@vert_pair[0,:]
            pair_idx+=1
    S_mat = torch.eye(len(track_poly_verts), device = device, dtype = torch.float, requires_grad = False)
    tmp = torch.ones((1,4), device = device, dtype = torch.float, requires_grad = False)
    S_mat = torch.kron(S_mat, tmp)
    return A, b, S_mat

def get_track_ensemble(Ntracks, cfg, device):
    tracks = []
    for _ in range(Ntracks):
        tracks.append()