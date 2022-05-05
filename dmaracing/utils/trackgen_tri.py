# track generator copied from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
# Track generator for TRIKart tracks to work with RLAgent code.

from typing import overload

import csv
import numpy as np
import math
# import cv2 as cv
from scipy.sparse import coo_matrix
import torch
from torch._C import dtype


def decimate_points(pts, pt_separation):
    theta = np.deg2rad(90)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    decimated_pts = []
    normal_vecs = []
    alphas = []
    last_pt = pts[0]
    i = 0
    decimated_pts.append(pts[0])
    pt_dist = 0.0
    for pt in pts:
        pt_dist = math.sqrt((last_pt[0] - pt[0])**2 + (last_pt[1] - pt[1])**2)
        if pt_dist >= pt_separation:
            decimated_pts.append(pt)
            pt0 = np.array(last_pt)
            pt1 = np.array(pt)
            dir_vec = pt1 - pt0
            alpha = math.atan2(dir_vec[1], dir_vec[0])
            alphas.append(alpha)
            perpendicular_unit_vec = np.dot(dir_vec, rot) / np.linalg.norm(dir_vec)
            normal_vecs.append(perpendicular_unit_vec)
            last_pt = pt
    # Add in the lastpt->firstpt polygon
    print(pt_dist)
    pt0 = np.array(last_pt)
    pt1 = np.array(pts[0])
    dir_vec = pt1 - pt0
    alpha = math.atan2(dir_vec[1], dir_vec[0])
    alphas.append(alpha)
    perpendicular_unit_vec = np.dot(dir_vec, rot) / np.linalg.norm(dir_vec)
    normal_vecs.append(perpendicular_unit_vec)
    return np.array(normal_vecs), np.array(decimated_pts), np.array(alphas)


def generate_polygons(normal_vecs, decimated_pts, half_track_width):
    polygons = []
    for idx, pt in enumerate(decimated_pts):
        next_idx = (idx + 1) % len(normal_vecs)
        polygon = []
        pt0 = pt
        pt1 = decimated_pts[next_idx]
        n0 = normal_vecs[idx]
        n1 = normal_vecs[next_idx]
        # print(n0)
        # print(n1)
        poly1 = pt0 - (n0 * half_track_width)
        poly2 = pt1 - n1 * half_track_width
        poly3 = pt1 + n1 * half_track_width
        poly4 = pt0 + n0 * half_track_width
        polygons.append([poly1, poly2, poly3, poly4])
    return np.array(polygons)


def draw_track(img,
               centerline,
               track_poly_verts,
               border_poly_verts,
               border_poly_col,
               track_tile_count,
               cords2px,
               cl=True):

    img[:, :, 0] = 130
    img[:, :, 1] = 255
    img[:, :, 2] = 130

    overlay = img.copy()
    for idx in range(track_tile_count):
        verts = track_poly_verts[idx, :, :]
        vert_px = cords2px(verts)
        if idx == 0:
            cv.fillPoly(img, [vert_px], color=(178, 178, 178))
            cv.polylines(overlay, [vert_px], isClosed=True,  color=(0, 0, 0), thickness=1)
            cv.polylines(img, [vert_px[0:2, :]], isClosed=False,  color=(0, 0, 0), thickness=3)
        else:
            cv.fillPoly(img, [vert_px], color=(178, 178, 178) if idx % 2 == 0 else (168, 168, 168))
            cv.polylines(overlay, [vert_px], isClosed=True,  color=(0, 0, 0), thickness=1)

    #cv.polylines(overlay, [vert_px], isClosed = True,  color = (0,0,0), thickness = 1)

    if cl:
        cl_px = cords2px(centerline[:track_tile_count, ...])
        cv.polylines(img, [cl_px], isClosed=True, color=(0, 0, 0), thickness=1)
        num_cl = len(cl_px)
        for idx in range(num_cl):
            cv.circle(img, (cl_px[idx, 0], cl_px[idx, 1]), 1, (0, 0, int(idx / num_cl * 255)))

    for idx in range(len(border_poly_verts)):
        verts = border_poly_verts[idx, :, :]
        vert_px = cords2px(verts)
        cv.fillPoly(img, [vert_px], color=(0, 0, 255) if idx % 2 == 0 else (255, 255, 255))

    img = cv.addWeighted(overlay, 0.1, img, 0.9, 0)

    #draw_cord_axs(img, cords2px)
    return img


def draw_cord_axs(img, cords2px):
    xverts = np.array([[-10, 0], [10, 0]])
    yverts = np.array([[0, 10], [0, -10]])
    xverts = cords2px(xverts)
    yverts = cords2px(yverts)

    cv.polylines(img, [xverts], isClosed=True,  color=(0, 0, 0), thickness=2)
    cv.polylines(img, [yverts], isClosed=True,  color=(0, 0, 0), thickness=2)


def construct_poly_track_eqns(track_poly_verts, device):
    A = torch.zeros((4 * len(track_poly_verts), 2), device=device, dtype=torch.float, requires_grad=False)
    b = torch.zeros((len(track_poly_verts) * 4,), device=device, dtype=torch.float, requires_grad=False)
    order = [[0, 1], [1, 2], [2, 3], [3, 0]]
    for idx in range(len(track_poly_verts)):
        verts = track_poly_verts[idx, :, :]
        pair_idx = 0
        for pair in order:
            vert_pair = verts[pair, :]
            diff = vert_pair[1, :] - vert_pair[0, :]
            diff_rot = diff.copy()
            diff_rot[0] = diff_rot[1]
            diff_rot[1] = -diff[0]
            A[4 * idx + pair_idx, 0] = diff_rot[0]
            A[4 * idx + pair_idx, 1] = diff_rot[1]
            b[4 * idx + pair_idx] = diff_rot@vert_pair[0, :]
            pair_idx += 1
    S_mat = torch.eye(len(track_poly_verts), device=device, dtype=torch.float, requires_grad=False)
    tmp = torch.ones((1, 4), device=device, dtype=torch.float, requires_grad=False)
    S_mat = torch.kron(S_mat, tmp)
    return A, b, S_mat


def get_tri_track_ensemble(Ntracks, cfg, device):
    # Load a centerline track, generate polygons and matricies, and repeat Ntracks times
    TRACK_POLYGON_SPACING = 0.5
    TRACK_HALF_WIDTH = 0.5
    file_path = "maps/large_oval.csv"
    path = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for waypoint in csv_reader:
            path.append((waypoint[0], waypoint[1]))
    normal_vecs, track_pts, alphas = decimate_points(path, TRACK_POLYGON_SPACING)
    # print(track_pts)
    # print(normal_vecs)
    polygons = generate_polygons(normal_vecs, track_pts, TRACK_HALF_WIDTH)
    # polygons = torch.tensor(polygons)  # .unsqueeze(0)
    As, b, S_mat = construct_poly_track_eqns(polygons, device)
    val = [track_pts, polygons, alphas, As, b, S_mat, None,
           None], 0.5, len(polygons)

    alpha = torch.tensor(alphas, device=device, requires_grad=False, dtype=torch.float).unsqueeze(0).expand(Ntracks, -1)
    # A = torch.tensor(As, device=device, requires_grad=False, dtype=torch.float).unsqueeze(0).expand(Ntracks, -1, -1)
    # b = torch.tensor(b, device=device, requires_grad=False, dtype=torch.float).unsqueeze(0).expand(Ntracks, -1)
    # S_mat = torch.tensor(S_mat, device=device, requires_grad=False, dtype=torch.float).unsqueeze(0).expand(Ntracks, -1, -1)
    A = As.unsqueeze(0).expand(Ntracks, -1, -1)
    b = b.unsqueeze(0).expand(Ntracks, -1)
    S_mat = S_mat.unsqueeze(0).expand(Ntracks, -1, -1)
    centerline = torch.tensor(track_pts, device=device, requires_grad=False, dtype=torch.float).unsqueeze(0).expand(Ntracks, -1, -1)
    track_poly_verts = np.tile(polygons, [Ntracks, 1, 1, 1])
    track_tile_counts = np.array([len(polygons)])
    track_tile_counts = np.tile(track_tile_counts, [Ntracks])

    return [centerline, track_poly_verts, alpha, A, b, S_mat, np.ones([Ntracks,0]), np.ones([Ntracks,0])], 0.5, track_tile_counts


if __name__ == "__main__":
    file_path = "maps/large_oval.csv"
    path = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for waypoint in csv_reader:
            path.append((waypoint[0], waypoint[1]))
    normal_vecs, track_pts, alphas = decimate_points(path, 0.5)
    # print(track_pts)
    # print(normal_vecs)
    polygons = generate_polygons(normal_vecs, track_pts, 0.5)
    # polygons = torch.tensor(polygons)  # .unsqueeze(0)
    print(polygons.shape)
    As, b, S_mat = construct_poly_track_eqns(polygons, "cpu")
    val = [track_pts, polygons, alphas, As, b, S_mat, None,
           None], 0.5, len(polygons)

    ret = get_tri_track_ensemble(4, None, "cpu")
    # val = [centerline, track_poly_verts, alphas, As, b, S_mat, border_poly_verts,
    #        border_poly_col], TRACK_DETAIL_STEP, len(track_poly_verts)
    # print(polygons)
    import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for polygon in polygons:
        plt.plot(polygon[:, 0], polygon[:, 1])
    plt.plot(np.array(path)[:, 0], np.array(path)[:, 1])
    plt.savefig("polygons.png")
    plt.show()

    # Get di
