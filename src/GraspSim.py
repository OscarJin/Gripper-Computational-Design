import pybullet as p
import os
import time
import pybullet_data
import numpy as np
from GeometryUtils import GraspingObj
from GripperModel import FOAMGripper, Finger
from typing import List
from scipy.optimize import fsolve
import warnings


def calc_area_0(finger: Finger):
    areas_0 = np.empty(finger.n_unit - 1, dtype=float)
    for i in range(finger.n_unit - 1):
        u_left = finger.units[i]
        u_right = finger.units[i + 1]
        a = np.sqrt(u_left.height ** 2 + (u_left.height / np.tan(u_left.theta2) + u_right.gap / 2) ** 2)
        b = np.sqrt(u_right.height ** 2 + (u_right.height / np.tan(u_right.theta1) + u_right.gap / 2) ** 2)
        phi_0 = np.pi - np.arcsin(u_left.height / a) - np.arcsin(u_right.height / b)
        a /= 1000
        b /= 1000
        areas_0[i] = a * b * np.sin(phi_0) / 2
    return areas_0


def compute_joint_position(finger: Finger, joint: int, cur_area: float, dt: float, delta_area: float) -> float:
    def calc_cos(_a, _b, _c):
        return np.arccos(np.clip((_a ** 2 + _b ** 2 - _c ** 2) / (2 * _a * _b), -1, 1))

    u_left = finger.units[joint]
    u_right = finger.units[joint + 1]
    a = np.sqrt(u_left.height ** 2 + (u_left.height / np.tan(u_left.theta2) + u_right.gap / 2) ** 2)
    b = np.sqrt(u_right.height ** 2 + (u_right.height / np.tan(u_right.theta1) + u_right.gap / 2) ** 2)
    phi_0 = np.pi - np.arcsin(u_left.height / a) - np.arcsin(u_right.height / b)
    if a > b:
        a, b = b, a
    a /= 1000
    b /= 1000
    l_skin = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(phi_0)) / 2

    phi_trans = calc_cos(a, b-l_skin, l_skin)
    area_trans = a * (b - l_skin) * np.sin(phi_trans) / 2

    x_max = (b + 2 * l_skin - a) / 2
    phi_min = calc_cos(a, b - x_max, 2 * l_skin - x_max)
    area_min = a * (b - x_max) * np.sin(phi_min) / 2
    area = max(cur_area - delta_area * dt, area_min)

    def func1(_x):
        phi = _x[0]
        if not 0 <= phi <= phi_0:
            return 1
        c = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(phi)) / 2
        h = np.sqrt(l_skin ** 2 - c ** 2)
        return a * b * np.sin(phi) / 2 - c * h - area

    def func2(_x):
        if _x > x_max:
            return 1
        phi = calc_cos(a, b - _x, 2 * l_skin - _x)
        return a * (b - _x) * np.sin(phi) / 2 - area

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if area > area_trans:
            pos = fsolve(func1, np.asarray([phi_0 / 2]))[0]
        else:
            x = fsolve(func2, np.asarray([l_skin]))[0]
            pos = calc_cos(a, b - x, 2 * l_skin - x)

    return phi_0 - pos


def compute_finger_positions(finger: Finger, joint: int, area_0: float, frames: int = 500, initial_delta_area: float = 10e-4):
    joint_pos = np.empty(frames, dtype=float)
    cur_area = area_0
    for i in range(frames):
        joint_pos[i] = compute_joint_position(finger, joint, cur_area, 1 / 240, initial_delta_area)
        cur_area -= initial_delta_area / 240

    return joint_pos.reshape((1, frames))


def gripper_sim(obj: GraspingObj, obj_urdf: str, gripper: FOAMGripper, mode: int = p.DIRECT):
    # begin pybullet test
    physicsClient = p.connect(mode)
    p.setGravity(0, 0, -9.8, physicsClientId=physicsClient)

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physicsClient)
    planeId = p.loadURDF("plane.urdf", physicsClientId=physicsClient)

    # load grasping object
    startPos = [0., 0., obj.cog[-1]]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF(obj_urdf, startPos, startOrientation, physicsClientId=physicsClient,
                        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    p.changeDynamics(box_id, -1, mass=50e-3, lateralFriction=.5, physicsClientId=physicsClient)

    # load gripper
    f_debug = open(os.path.join(os.path.abspath('..'), 'debug.txt'), 'w')
    joint_positions = {}
    finger_id = []
    for f in gripper.fingers:
        startPos = [0, 0, obj.height + .04]
        # startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
        f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1, physicsClientId=physicsClient,
                          flags=p.URDF_USE_SELF_COLLISION)
        finger_id.append(f_id)

        areas_0 = calc_area_0(f)
        pos = np.empty([f.n_unit - 1, 500], dtype=float)
        for i in range(f.n_unit - 1):
            pos[i] = compute_finger_positions(f, i, float(areas_0[i]),
                                              initial_delta_area=5e-4)

        joint_positions[f_id] = pos

    p.setRealTimeSimulation(0, physicsClientId=physicsClient)

    for i in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=min(joint_positions[f][j][i], limit), physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
        for j in range(p.getNumJoints(finger_id[0], physicsClient)):
            pos = p.getJointState(finger_id[0], j, physicsClient)[0]
            f_debug.write(str(np.rad2deg(pos))+'\t')
        f_debug.write('\n')
        if mode == p.GUI:
            time.sleep(1. / 240.)

    for _ in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=limit, physicsClientId=physicsClient)
            p.resetBaseVelocity(f, [0, 0, .05], physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
        if mode == p.GUI:
            time.sleep(1. / 240.)

    objPos, _ = p.getBasePositionAndOrientation(box_id, physicsClient)

    p.disconnect(physicsClientId=physicsClient)
    # p.resetSimulation()
    # end pybullet test

    return objPos


def multiple_gripper_sim(obj: GraspingObj, obj_urdf: str, grippers: List[FOAMGripper], mode: int = p.DIRECT):
    # begin pybullet test
    physicsClient = p.connect(mode)
    p.setGravity(0, 0, -9.8, physicsClientId=physicsClient)

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physicsClient)
    planeId = p.loadURDF("plane.urdf", physicsClientId=physicsClient)

    n_obj = len(grippers)
    positions = np.empty((n_obj, 2), dtype=int)
    for i in range(n_obj):
        positions[i][0] = i % round(np.sqrt(n_obj))
        positions[i][1] = i / round(np.sqrt(n_obj))

    # load grasping objects
    obj_id = [None for _ in range(n_obj)]
    obj_gap = 5 * obj.height
    for i in range(n_obj):
        startPos = [positions[i][0] * obj_gap, positions[i][1] * obj_gap, obj.cog[-1]]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        box_id = p.loadURDF(obj_urdf, startPos, startOrientation, physicsClientId=physicsClient,
                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        p.changeDynamics(box_id, -1, mass=50e-3, lateralFriction=.5, physicsClientId=physicsClient)
        obj_id[i] = box_id

    # load grippers
    finger_id = []
    joint_positions = {}
    for i, g in enumerate(grippers):
        for f in g.fingers:
            startPos = [positions[i][0] * obj_gap, positions[i][1] * obj_gap, obj.height + .04]
            startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
            f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1, physicsClientId=physicsClient,
                              flags=p.URDF_USE_SELF_COLLISION)
            finger_id.append(f_id)

            areas_0 = calc_area_0(f)
            pos = np.empty([f.n_unit - 1, 500], dtype=float)
            for k in range(f.n_unit - 1):
                pos[k] = compute_finger_positions(f, k, float(areas_0[k]), initial_delta_area=3e-4)
            joint_positions[f_id] = pos

    p.setRealTimeSimulation(0, physicsClientId=physicsClient)

    for i in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=min(0.01 * i, .95 * limit), physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
        if mode == p.GUI:
            time.sleep(1. / 240.)

    for _ in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=.95 * limit, physicsClientId=physicsClient)
            p.resetBaseVelocity(f, [0, 0, .05], physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
        if mode == p.GUI:
            time.sleep(1. / 240.)

    final_pos = [None for _ in range(n_obj)]
    for i, id in enumerate(obj_id):
        objPos, _ = p.getBasePositionAndOrientation(id, physicsClient)
        final_pos[i] = objPos

    p.disconnect(physicsClientId=physicsClient)
    # end pybullet test

    return final_pos


import pickle
from GeometryUtils import ContactPoints
from GripperModel import initialize_gripper
from GripperInitialization import initialize_fingers
import time

if __name__ == "__main__":
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)

    test_obj_urdf = os.path.join(os.path.abspath('..'), "assets/ycb/013_apple.urdf")
    cps = ContactPoints(test_obj, [206, 712, 2067, 3062])
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .04])

    widths = np.linspace(15., 25., 5)
    height_ratio = np.linspace(1., 2., 5)
    ww, rr = np.meshgrid(widths, height_ratio)

    # for i, w in np.ndenumerate(ww):
    #     _, fingers = initialize_gripper(cps, end_effector_pos, 8, expand_dist=20., height_ratio=rr[i], width=w)
    #     gripper = FOAMGripper(fingers)
    #     # final_pos = gripper_sim(test_obj, test_obj_urdf, gripper, mode=p.DIRECT)
    #     final_pos = multiple_gripper_sim(test_obj, test_obj_urdf, [gripper] * 50, p.DIRECT)
    #     success_cnt = 0
    #     for pos in final_pos:
    #         if pos[-1] > test_obj.cog[-1] + .5 * (.05 * 500 / 240):
    #             success_cnt += 1
    #     print(f'Width: {w}, Ratio: {rr[i]}', success_cnt)
    #     gripper.clean()

    t1 = time.time()
    skeleton = initialize_fingers(cps, end_effector_pos, 8, root_length=.05)
    _, fingers = initialize_gripper(cps, end_effector_pos, 8, height_ratio=2, width=25, finger_skeletons=skeleton)
    gripper = FOAMGripper(fingers)
    t2 = time.time()
    print(t2 - t1)
    # gripper_sim(test_obj, test_obj_urdf, gripper, p.GUI)
    final_pos = multiple_gripper_sim(test_obj, test_obj_urdf, [gripper] * 2, p.GUI)
    for i, pos in enumerate(final_pos):
        if pos[-1] < test_obj.cog[-1] + .5 * (.05 * 500 / 240):
            print(i)

    # # test positions
    # with open(os.path.join(os.path.abspath('..'), 'debug.txt'), 'w') as f:
    #     positions = np.empty([gripper.fingers[0].n_unit - 1, 500], dtype=float)
    #     for i in range(gripper.fingers[0].n_unit - 1):
    #         for j in range(500):
    #             positions[i][j] = np.rad2deg(compute_joint_position(gripper.fingers[0], i, j / 240, 2e-4))
    #             f.write(str(positions[i][j])+'\t')
    #         f.write('\n')

    # gripper.assemble(bottom_thick=1.2)
    gripper.fingers[2].assemble(bottom_thick=2.)
    gripper.clean()
