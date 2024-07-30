import pybullet as p
import os
import pybullet_data
import numpy as np
import random
from GeometryUtils import GraspingObj
from GripperModel import FOAMGripper, Finger
from typing import List
from scipy.optimize import fsolve
import warnings
from time import sleep


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


def compute_finger_joints_maxforce(finger: Finger, joint: int, dp=90e3):
    u_left = finger.units[joint]
    u_right = finger.units[joint + 1]
    a = np.sqrt(u_left.height ** 2 + (u_left.height / np.tan(u_left.theta2) + u_right.gap / 2) ** 2)
    b = np.sqrt(u_right.height ** 2 + (u_right.height / np.tan(u_right.theta1) + u_right.gap / 2) ** 2)
    phi_0 = np.pi - np.arcsin(u_left.height / a) - np.arcsin(u_right.height / b)
    a /= 1000
    b /= 1000
    l_skin = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(phi_0))
    x = (a + b - l_skin) / 2
    max_force = dp * (finger.max_unit_width / 1000) * .5 * (x ** 2)
    return max_force


def _kinematic_verify(gripper: FOAMGripper, speed: float, frames: int, test_ind: int = 0, mode: int = p.DIRECT):
    # begin pybullet test
    physicsClient = p.connect(mode)
    p.setGravity(0, 0, -9.8, physicsClientId=physicsClient)

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physicsClient)
    planeId = p.loadURDF("plane.urdf", physicsClientId=physicsClient)

    # load gripper
    f_debug = open(os.path.join(os.path.abspath('..'), 'results', '-0', 'debug.txt'), 'w')

    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, gripper.fingers[test_ind].orientation])
    finger_id = p.loadURDF(gripper.fingers[test_ind].filename, startPos, startOrientation, useFixedBase=1, physicsClientId=physicsClient,
                           flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)
    areas_0 = calc_area_0(gripper.fingers[test_ind])
    pos = np.empty([gripper.fingers[test_ind].n_unit - 1, frames], dtype=float)
    for i in range(gripper.fingers[test_ind].n_unit - 1):
        pos[i] = compute_finger_positions(gripper.fingers[test_ind], i, float(areas_0[i]), frames=frames, initial_delta_area=speed)
        # f_debug.write(str(np.rad2deg(pos[i])) + '\n')

    p.setRealTimeSimulation(0, physicsClientId=physicsClient)

    for _i in range(frames):
        for j in range(p.getNumJoints(finger_id, physicsClient)):
            limit = p.getJointInfo(finger_id, j, physicsClient)[9]
            p.setJointMotorControl2(finger_id, j, p.POSITION_CONTROL,
                                    targetPosition=min(pos[j][_i], .95 * limit),
                                    physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
        if _i % 8 == 0:
            for j in range(p.getNumJoints(finger_id, physicsClient)):
                cur_pos = p.getJointState(finger_id, j, physicsClient)[0]
                # f_debug.write(str(np.rad2deg(cur_pos))+'\t')
                f_debug.write(str(np.rad2deg(pos[j][_i])) + '\t')
            f_debug.write('\n')
        if mode == p.GUI:
            sleep(1. / 240.)

    p.disconnect(physicsClientId=physicsClient)
    # end pybullet test


def multiple_gripper_sim(obj: GraspingObj, obj_urdf: str, grippers: List[FOAMGripper],
                         height: float, max_deviation=.1, obj_mass: float = 50e-3, increasing_mass=None,
                         mode: int = p.DIRECT):
    """
    Pybullet simulation of grasping with random x and y deviation
    :param increasing_mass: increasing object mass (default None, all the same mass)
    :param height: distance from end effector to object top
    :param max_deviation: max percentage of palm radius (x_span + y_span) / 4, default 10%
    :return: final positions of objects and a bool of collision between grippers and ground
    """
    # begin pybullet test
    physicsClient = p.connect(mode)
    if mode == p.GUI:
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=-40, cameraPitch=-40, cameraTargetPosition=[0, 0, 0],
                                     physicsClientId=physicsClient)
    p.setGravity(0, 0, -9.8, physicsClientId=physicsClient)

    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physicsClient)
    planeId = p.loadURDF("plane.urdf", physicsClientId=physicsClient)

    n_obj = len(grippers)
    positions = np.empty((n_obj, 2), dtype=int)
    for _i in range(n_obj):
        positions[_i][0] = _i % round(np.sqrt(n_obj))
        positions[_i][1] = _i / round(np.sqrt(n_obj))

    # load grasping objects
    obj_id = [None for _ in range(n_obj)]
    obj_gap: int = 1
    for _i in range(n_obj):
        # startPos = [positions[_i][0] * obj_gap, positions[_i][1] * obj_gap, obj.cog[-1]]
        startPos = [positions[_i][0] * obj_gap, positions[_i][1] * obj_gap, 0.0075]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        box_id = p.loadURDF(obj_urdf, startPos, startOrientation, physicsClientId=physicsClient,
                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        cur_obj_mass = obj_mass if increasing_mass is None else obj_mass * np.power(1 + increasing_mass, _i)
        p.changeDynamics(box_id, -1, mass=cur_obj_mass / 3, lateralFriction=.5, physicsClientId=physicsClient)
        p.changeDynamics(box_id, 0, mass=cur_obj_mass / 3, lateralFriction=.5, physicsClientId=physicsClient)
        p.changeDynamics(box_id, 1, mass=cur_obj_mass / 3, lateralFriction=.5, physicsClientId=physicsClient)
        obj_id[_i] = box_id
        # add text
        p.addUserDebugText(str(int(cur_obj_mass * 1000)),
                           [positions[_i][0] * obj_gap, positions[_i][1] * obj_gap - 0.1, 0],
                           [0, 0, 0], 1)

    # load grippers
    finger_id = []
    joint_positions = {}
    joint_positions_memory = {}
    joint_maxforce = {}
    joint_maxforce_memory = {}
    palm_r = (obj.x_span + obj.y_span) / 4
    for _i, g in enumerate(grippers):
        for f in g.fingers:
            deviation_r = random.uniform(0, max_deviation) * palm_r
            deviation_angle = random.uniform(0, 2 * np.pi)
            deviation_x = deviation_r * np.cos(deviation_angle)
            deviation_y = deviation_r * np.sin(deviation_angle)
            startPos = [positions[_i][0] * obj_gap, positions[_i][1] * obj_gap, obj.height + height]
            if _i != 0:
                startPos[0] += deviation_x
                startPos[1] += deviation_y
            startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
            f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1, physicsClientId=physicsClient,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_MAINTAIN_LINK_ORDER)
            finger_id.append(f_id)

            areas_0 = calc_area_0(f)
            if f.id in joint_positions_memory.keys():
                pos = joint_positions_memory[f.id]
            else:
                pos = np.empty([f.n_unit - 1, 500], dtype=float)
                for k in range(f.n_unit - 1):
                    pos[k] = compute_finger_positions(f, k, float(areas_0[k]), initial_delta_area=4e-4)
                joint_positions_memory[f.id] = pos
            joint_positions[f_id] = pos

            if f.id in joint_maxforce_memory.keys():
                max_force = joint_maxforce_memory[f.id]
            else:
                max_force = np.empty([f.n_unit - 1], dtype=float)
                for k in range(f.n_unit - 1):
                    max_force[k] = compute_finger_joints_maxforce(f, k)
                joint_maxforce_memory[f.id] = max_force
            joint_maxforce[f_id] = max_force

    p.setRealTimeSimulation(0, physicsClientId=physicsClient)

    gripper_ground_collision = False
    collision_start_frame = 0

    for _i in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=min(joint_positions[f][j][_i], .95 * limit),
                                        force=joint_maxforce[f][j],
                                        physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
        if not gripper_ground_collision:
            p.performCollisionDetection(physicsClientId=physicsClient)
            for f in finger_id:
                f_plane_cps = p.getContactPoints(bodyA=f, bodyB=planeId)
                if len(f_plane_cps) != 0:
                    gripper_ground_collision = True
                    collision_start_frame = _i
                    break

        if mode == p.GUI:
            sleep(1. / 240.)

    for _ in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=min(joint_positions[f][j][-1], .95 * limit),
                                        force=joint_maxforce[f][j],
                                        physicsClientId=physicsClient)
            p.resetBaseVelocity(f, [0, 0, .05], physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
        if mode == p.GUI:
            sleep(1. / 240.)

    _final_pos = [None for _ in range(n_obj)]
    for _i, id in enumerate(obj_id):
        objPos, _ = p.getBasePositionAndOrientation(id, physicsClient)
        _final_pos[_i] = objPos

    p.disconnect(physicsClientId=physicsClient)
    # end pybullet test

    return _final_pos, (gripper_ground_collision, collision_start_frame)


import pickle
from GeometryUtils import ContactPoints
from GripperModel import initialize_gripper
from GripperInitialization import initialize_fingers
from time import perf_counter

if __name__ == "__main__":
    if True:
        """single object grasping"""
        ycb_model = '000_stage'
        with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}/{ycb_model}.pickle"),
                  'rb') as f_test_obj:
            test_obj: GraspingObj = pickle.load(f_test_obj)

        test_obj_urdf = os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}.urdf")
        cps = ContactPoints(test_obj, np.take(test_obj.faces_mapping_clamp_height_and_radius, [0, 28, 640, 732]).tolist())
        end_effector_pos = test_obj.effector_pos[0]
        # widths = np.linspace(12.5, 37.5, 11)
        widths = np.linspace(12.5, 25., 6)
        _min_height_ratio = int(25e-2 / (test_obj.effector_pos[-1][-1] - test_obj.maxHeight)) / 10
        height_ratio = np.arange(_min_height_ratio, .7, .1)
        # height_ratio = np.linspace(0.4, 0.7, 4)

        t1 = perf_counter()
        end_height = end_effector_pos[-1] - test_obj.maxHeight
        skeleton = initialize_fingers(cps, end_effector_pos, 8, root_length=.04, expand_dist=end_height,
                                      grasp_force=1e-3)
        _, fingers = initialize_gripper(cps, end_effector_pos, 8, expand_dist=end_height * 1000,
                                        height_ratio=height_ratio[3], width=widths[4], gap=2, finger_skeletons=skeleton)
        gripper = FOAMGripper(fingers)
        t2 = perf_counter()
        print(t2 - t1)
        if False:
            """bullet sim"""
            final_pos, collision = multiple_gripper_sim(test_obj, test_obj_urdf, [gripper] * 20, end_height,
                                                        max_deviation=.1, mode=p.GUI, obj_mass=50e-3)
            t3 = perf_counter()
            print(t3 - t2)
            print(f"Collision: {collision}")
            for i, pos in enumerate(final_pos):
                if pos[-1] < test_obj.cog[-1] + .5 * (.05 * 500 / 240):
                    print(i)

        if True:
            """heavy bullet sim"""
            final_pos, collision = multiple_gripper_sim(test_obj, test_obj_urdf, [gripper] * 20, end_height,
                                                        max_deviation=0, mode=p.GUI,
                                                        obj_mass=200e-3, increasing_mass=0.15)
            t3 = perf_counter()
            print(t3 - t2)
            print(f"Collision: {collision}")
            for i, pos in enumerate(final_pos):
                if pos[-1] < test_obj.cog[-1] + .5 * (.05 * 500 / 240):
                    print(i)

        if False:
            _kinematic_verify(gripper, speed=4.63e-5, frames=240 * 22, test_ind=0, mode=p.DIRECT)

        # gripper.assemble(bottom_thick=1.5)
        # print((test_obj.height + (end_effector_pos[-1] - test_obj.maxHeight)) * 1000 + 30)
        # gripper.fingers[1].assemble(bottom_thick=1.5)
        gripper.clean()

    if False:
        """multi objects grasping"""
        ycb_models = ['011_banana', '013_apple']
        ref_ind = 0
        max_height_ind = 1
        grasp_ind = 0
        results_dir = os.path.join(os.path.abspath('..'), 'results')
        save_dir = os.path.join(results_dir, '-'.join(ycb_models) + "-17")
        with open(os.path.join(save_dir, 'reference_object.pickle'), 'rb') as f_test_obj:
            ref_obj: GraspingObj = pickle.load(f_test_obj)
        with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_models[max_height_ind]}/{ycb_models[max_height_ind]}.pickle"),
                  'rb') as f_test_obj:
            max_height_obj: GraspingObj = pickle.load(f_test_obj)
        # with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_models[grasp_ind]}/{ycb_models[grasp_ind]}.pickle"),
        #           'rb') as f_test_obj:
        #     test_obj: GraspingObj = pickle.load(f_test_obj)
        ycb_model = '013_apple'
        with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}/{ycb_model}.pickle"),
                  'rb') as f_test_obj:
            test_obj: GraspingObj = pickle.load(f_test_obj)

        test_obj_urdf = os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}.urdf")
        cps = ContactPoints(ref_obj,
                            np.take(ref_obj.faces_mapping_clamp_height_and_radius, [0, 660, 1415, 1581]).tolist())
        end_effector_pos = ref_obj.effector_pos[0]
        widths = np.linspace(12.5, 25., 6)
        _min_height_ratio = int(25e-2 / (max_height_obj.effector_pos[-1][-1] - max_height_obj.maxHeight)) / 10
        height_ratio = np.arange(_min_height_ratio, .7, .1)

        t1 = perf_counter()
        end_height = end_effector_pos[-1] - max_height_obj.maxHeight
        skeleton = initialize_fingers(cps, end_effector_pos, 8, root_length=.04, expand_dist=end_height)
        _, fingers = initialize_gripper(cps, end_effector_pos, 8, expand_dist=end_height * 1000,
                                        height_ratio=height_ratio[3], width=widths[4], gap=2, finger_skeletons=skeleton)
        gripper = FOAMGripper(fingers)
        t2 = perf_counter()
        print(t2 - t1)

        final_pos, collision = multiple_gripper_sim(test_obj, test_obj_urdf, [gripper] * 16,
                                                    end_effector_pos[-1] - test_obj.maxHeight,
                                                    max_deviation=.1, mode=p.GUI)
        print(f"Collision: {collision}")
        for i, pos in enumerate(final_pos):
            if pos[-1] < test_obj.cog[-1] + .5 * (.05 * 500 / 240):
                print(i)

        gripper.assemble(bottom_thick=1.5)
        print((test_obj.height + (end_effector_pos[-1] - test_obj.maxHeight)) * 1000 + 30)
        # gripper.fingers[1].assemble(bottom_thick=1.5)
        gripper.clean()
