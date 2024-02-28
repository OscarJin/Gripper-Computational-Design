from GeometryUtils import GraspingObj, ContactPoints
import numpy as np
from typing import List


def should_pop(a, c, obj: GraspingObj) -> bool:
    ac = (c - a) / np.linalg.norm(c - a)
    a += ac * 1e-6
    c -= ac * 1e-6
    return not obj.intersect_segment(a, c - a)


def should_pop_surface(a, c, vid_a, vid_c, obj: GraspingObj) -> bool:
    ac = (c - a) / np.linalg.norm(c - a)
    n_a = obj.compute_vertex_normal(vid_a) if vid_a != -1 else None
    n_c = obj.compute_vertex_normal(vid_c) if vid_c != -1 else None
    if n_a is not None and n_c is not None and np.dot(n_a, ac) < 0 < np.dot(n_c, ac):
        return False
    a += ac * 1e-6
    c -= ac * 1e-6
    return not obj.intersect_segment(a, c - a)


def point_to_line_dist(a, b, p):
    ab = b - a
    ap = p - a
    proj = (np.dot(ap, ab) / np.linalg.norm(ab)) * (ab / np.linalg.norm(ab))
    return np.linalg.norm(ap - proj)


def initialize_finger_skeleton(fid: int, obj: GraspingObj, effector_pos, n_finger_joints: int, expand_dist=.03):
    vid: int = obj.compute_closest_point(fid)
    finger = np.empty([1, 3], dtype=float)
    finger[0] = np.average(obj.faces[fid], axis=0)
    fingerVid: List[int] = [-1]

    while vid != -1:
        toPush = obj.vertices[vid]
        while len(finger) > 1 and should_pop_surface(finger[-2], toPush, fingerVid[-2], vid, obj):
            finger = finger[:-1]
            fingerVid = fingerVid[:-1]
        finger = np.concatenate((finger, toPush.reshape((1, 3))), axis=0)
        fingerVid.append(vid)
        vid = obj.parent[vid]
    finger = np.concatenate((finger, effector_pos.reshape((1, 3))), axis=0)
    fingerVid.append(-1)

    # expand segment by expand_dist
    for j in range(1, len(finger) - 1):
        VN = obj.compute_vertex_normal(fingerVid[j])
        if VN[-1] < 0:
            offset_dist = min(expand_dist, (finger[j][-1] - obj.minHeight) / (1 - VN[-1]))
        else:
            offset_dist = min(expand_dist, (effector_pos[-1] - finger[j][-1]) / (1 + VN[-1]))
        finger[j] += VN * offset_dist

    # offset effector (ind: -1)
    n_finger = np.cross(finger[-1] - finger[-2], finger[-3] - finger[-2])
    oa = finger[0][:-1] - finger[-1][:-1]
    oa /= np.linalg.norm(oa)
    offset_dist = 0.03
    finger[-1][:-1] += oa * offset_dist
    finger = np.concatenate((finger, effector_pos.reshape((1, 3))), axis=0)
    # if len(finger) > 3 and should_pop(finger[-4], finger[-2], obj):
    if len(finger) > 3 and np.dot(n_finger, np.cross(finger[-2] - finger[-3], finger[-4] - finger[-3])) < 0:
        finger = np.delete(finger, -3, axis=0)

    # offset contact point
    n_cp = obj.normals[fid] / np.linalg.norm(obj.normals[fid])
    offset_dist = min(.02, (finger[0][-1] - obj.minHeight) / (1 + max(-n_cp[-1], 1e-6)))
    finger = np.insert(finger, 1, finger[0] + offset_dist * n_cp, axis=0)
    # if len(finger) > 4 and should_pop(finger[1], finger[3], obj):
    if len(finger) > 4 and np.dot(n_finger, np.cross(finger[3] - finger[2], finger[1] - finger[2])) < 0:
        finger = np.delete(finger, 2, axis=0)

    # fix number of segment
    exist_minus_angle = False
    while len(finger) > n_finger_joints + 1 or exist_minus_angle:
        best_id = -1
        best_fall_back_id = -1
        best = np.inf
        best_fall_back = np.inf
        exist_minus_angle = False
        for j in range(2, len(finger) - 2):
            cost = point_to_line_dist(finger[j - 1], finger[j + 1], finger[j])
            n_cur = np.cross(finger[j + 1] - finger[j], finger[j - 1] - finger[j])
            if np.dot(n_finger, n_cur) < 0.:
                exist_minus_angle = True
            # cost = min(np.linalg.norm(finger[j] - finger[j - 1]), np.linalg.norm(finger[j] - finger[j + 1]))
            if should_pop(finger[j - 1], finger[j + 1], obj):
                if cost < best:
                    best = cost
                    best_id = j
            if cost < best_fall_back:
                best_fall_back = cost
                best_fall_back_id = j
        if best_id == -1:
            best_id = best_fall_back_id
        finger = np.delete(finger, best_id, axis=0)

    while len(finger) < n_finger_joints + 1:
        # best_id = -1
        # best = -1
        # for j in range(1, len(finger)):
        #     cur = np.linalg.norm(finger[j] - finger[j - 1])
        #     if cur > best:
        #         best = cur
        #         best_id = j
        # finger = np.insert(finger, best_id, (finger[best_id] + finger[best_id - 1]) / 2., axis=0)
        finger = np.concatenate((finger, np.full([1, 3], np.nan)), axis=0)

    # co-plane
    for i in range(1, n_finger_joints):
        op = finger[i][:-1] - finger[0][:-1]
        proj = np.dot(op, oa) * oa
        finger[i][:-1] = finger[0][:-1] + proj

    return finger


def initialize_fingers(cps: ContactPoints, effector_pos, n_finger_joints: int, expand_dist=.03):
    if cps.obj.effector_pos is None or not np.all(np.isclose(cps.obj.effector_pos, effector_pos)):
        cps.obj.compute_connectivity_from(effector_pos)

    res = np.empty([cps.nContact, n_finger_joints + 1, 3])
    for i in range(cps.nContact):
        res[i] = initialize_finger_skeleton(cps.fid[i], cps.obj, effector_pos, n_finger_joints, expand_dist)
    return res


def compute_skeleton(finger_skeletons, cps:ContactPoints, effector_pos, n_finger_joints: int):
    L = np.full([cps.nContact, n_finger_joints], np.nan, dtype=float)
    angle = np.full([cps.nContact, n_finger_joints], np.nan, dtype=float)
    ori = np.empty(cps.nContact, dtype=float)
    for i, f in enumerate(finger_skeletons):
        n_joints = np.sum(~np.isnan(f).any(axis=1))
        for j in range(1, n_joints)[::-1]:
            L[i][-j] = np.linalg.norm(f[j - 1] - f[j]) * 1000      # mm
        for j in range(2, n_joints)[::-1]:
            a = f[j] - f[j - 1]
            b = f[j - 2] - f[j - 1]
            angle[i][-j] = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        # gc_n = cps.obj.normals[cps.fid[i]] / np.linalg.norm(cps.obj.normals[cps.fid[i]])   # outer
        # angle[i][-1] = np.arcsin(np.dot(f[1] - f[0], gc_n) / np.linalg.norm(f[1] - f[0]))
        # angle[i][-1] = angle[i][-1] if angle[i][-1] > np.deg2rad(5) else np.pi / 2
        angle[i][-1] = np.pi / 2

        v_ori = f[0][:-1] - effector_pos[:-1]
        ori[i] = np.arctan2(v_ori[1], v_ori[0])
    return L, angle, ori


import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle


if __name__ == "__main__":
    # stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.stl")
    # test_obj = GraspingObj(friction=0.5)
    # test_obj.read_from_stl(stl_file)
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)
    cps = ContactPoints(test_obj, [364, 1839, 2268, 3259])
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .02])
    # test_obj.compute_connectivity_from(end_effector_pos)
    skeletons = initialize_fingers(cps, end_effector_pos, 8)
    Ls, angles, oris = compute_skeleton(skeletons, cps, end_effector_pos, 8)
    print(Ls, angles, oris)

    # visualization
    figure = plt.figure(dpi=300)
    ax = figure.add_axes(mplot3d.Axes3D(figure))
    ax.add_collection3d(Poly3DCollection(test_obj.faces, alpha=.3, facecolors="lightgrey"))
    scale = test_obj._mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    for i in range(cps.nContact):
        for j in range(8):
            x = [skeletons[i][j][0], skeletons[i][j + 1][0]]
            y = [skeletons[i][j][1], skeletons[i][j + 1][1]]
            z = [skeletons[i][j][2], skeletons[i][j + 1][2]]
            ax.plot(x, y, z, c='r', linewidth=1., marker='.')
    plt.show()
