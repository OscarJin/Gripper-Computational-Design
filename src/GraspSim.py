import pybullet as p
import os
import time
import pybullet_data
import numpy as np
from GeometryUtils import GraspingObj
from GripperModel import FOAMGripper
from typing import List


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
    finger_id = []
    for f in gripper.fingers:
        startPos = [0, 0, obj.height + .02]
        startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
        f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1, physicsClientId=physicsClient,
                          flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        finger_id.append(f_id)

    p.setRealTimeSimulation(0, physicsClientId=physicsClient)

    for i in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=min(0.01 * i, limit), physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
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
    for i, g in enumerate(grippers):
        for f in g.fingers:
            startPos = [positions[i][0] * obj_gap, positions[i][1] * obj_gap, obj.height + .02]
            startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
            f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1, physicsClientId=physicsClient,
                              flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
            finger_id.append(f_id)

    p.setRealTimeSimulation(0, physicsClientId=physicsClient)

    for i in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f, physicsClient)):
                limit = p.getJointInfo(f, j, physicsClient)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL,
                                        targetPosition=min(0.01 * i, limit), physicsClientId=physicsClient)
        p.stepSimulation(physicsClientId=physicsClient)
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

if __name__ == "__main__":
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)

    test_obj_urdf = os.path.join(os.path.abspath('..'), "assets/ycb/013_apple.urdf")
    cps = ContactPoints(test_obj, [0, 708, 1432, 2856])
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .02])

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

    _, fingers = initialize_gripper(cps, end_effector_pos, 8, expand_dist=20., height_ratio=1.75, width=25.)
    gripper = FOAMGripper(fingers)
    final_pos = multiple_gripper_sim(test_obj, test_obj_urdf, [gripper] * 50, p.DIRECT)
    for i, pos in enumerate(final_pos):
        if pos[-1] < test_obj.cog[-1] + .5 * (.05 * 500 / 240):
            print(i)

    # gripper.assemble(bottom_thick=1.2)
    # gripper.seal_mask()
    gripper.clean()
