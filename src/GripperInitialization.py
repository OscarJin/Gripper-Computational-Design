from GeometryUtils import GraspingObj, ContactPoints
import numpy as np
from typing import List
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def should_pop(a, c, obj: GraspingObj) -> bool:
    ac = (c - a) / np.linalg.norm(c - a)
    a += ac * 1e-6
    c -= ac * 1e-6
    return not obj.intersect_segment(a, c - a)


def point_to_line_dist(a, b, p):
    ab = b - a
    ap = p - a
    proj = (np.dot(ap, ab) / np.linalg.norm(ab)) * (ab / np.linalg.norm(ab))
    return np.linalg.norm(ap - proj)


def initialize_finger_skeleton(fid: int, obj: GraspingObj, effector_pos, n_finger_joints: int, expand_dist=.02):
    vid: int = obj.compute_closest_point(fid)
    finger = np.empty([1, 3], dtype=float)
    finger[0] = np.average(obj.faces[fid], axis=0)
    fingerVid: List[int] = [-1]

    while vid != -1:
        toPush = obj.vertices[vid]
        while len(finger) > 1 and should_pop(finger[-2], toPush, obj):
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
        finger[j] += VN * expand_dist

    # offset effector (ind: -1)
    oa = finger[0][:-1] - finger[-1][:-1]
    oa /= np.linalg.norm(oa)
    offset_dist = 0.03
    finger[-1][:-1] += oa * offset_dist
    finger = np.concatenate((finger, effector_pos.reshape((1, 3))), axis=0)

    # fix number of segment
    while len(finger) > n_finger_joints + 1:
        best_id = -1
        best_fall_back_id = -1
        best = np.inf
        best_fall_back = np.inf
        for j in range(1, len(finger) - 1):
            cost = point_to_line_dist(finger[j - 1], finger[j + 1], finger[j])
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


def initialize_fingers(cps: ContactPoints, effector_pos, n_finger_joints: int):
    cps.obj.compute_connectivity_from(effector_pos)

    res = np.empty([cps.nContact, n_finger_joints + 1, 3])
    for i in range(cps.nContact):
        res[i] = initialize_finger_skeleton(cps.fid[i], cps.obj, effector_pos, n_finger_joints)
    return res


def initialize_gripper(cps: ContactPoints, effector_pos, n_finger_joints: int):
    fingers = initialize_fingers(cps, effector_pos, n_finger_joints)
    L = np.full([cps.nContact, n_finger_joints], np.nan, dtype=float)
    angle = np.full([cps.nContact, n_finger_joints - 1], np.nan, dtype=float)
    ori = np.empty(cps.nContact, dtype=float)

    for i, f in enumerate(fingers):
        n_joints = np.sum(~np.isnan(f).any(axis=1))
        for j in range(1, n_joints)[::-1]:
            L[i][-j] = np.linalg.norm(f[j - 1] - f[j])
        for j in range(1, n_joints - 1)[::-1]:
            a = f[j + 1] - f[j]
            b = f[j - 1] - f[j]
            angle[i][-j] = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        v_ori = f[0][:-1] - effector_pos[:-1]
        ori[i] = np.arctan2(v_ori[1], v_ori[0])

    return fingers, L, angle, ori


if __name__ == "__main__":
    stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.stl")
    test_obj = GraspingObj(friction=0.4)
    test_obj.read_from_stl(stl_file)
    cps = ContactPoints(test_obj, [176, 306, 959, 2036])
    end_effector_pos = np.asarray([test_obj.center_of_mass[0], test_obj.center_of_mass[1], test_obj.maxHeight + .02])
    # fingers = initialize_fingers(cps, end_effector_pos, 4)
    fingers, Ls, angles, oris = initialize_gripper(cps, end_effector_pos, 4)
    print(Ls, angles, oris)

    # visualization
    figure = plt.figure(dpi=300)
    ax = figure.add_axes(mplot3d.Axes3D(figure))
    ax.add_collection3d(Poly3DCollection(test_obj.faces, alpha=.3, facecolors="lightgrey"))
    scale = test_obj._mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    for i in range(cps.nContact):
        for j in range(4):
            x = [fingers[i][j][0], fingers[i][j + 1][0]]
            y = [fingers[i][j][1], fingers[i][j + 1][1]]
            z = [fingers[i][j][2], fingers[i][j + 1][2]]
            ax.plot(x, y, z, c='r', linewidth=1., marker='.')
    plt.show()
