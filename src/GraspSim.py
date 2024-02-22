import pybullet as p
import os
import time
import pybullet_data
import numpy as np
from GeometryUtils import GraspingObj
from GripperModel import FOAMGripper


def gripper_sim(obj: GraspingObj, obj_urdf: str, gripper: FOAMGripper):
    # begin pybullet test
    # physicsClient = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    # load grasping object
    startPos = [0., 0., obj.cog[-1]]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF(obj_urdf, startPos, startOrientation,
                        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    p.changeDynamics(box_id, -1, mass=50e-3, lateralFriction=.5)

    # load gripper
    finger_id = []
    for f in gripper.fingers:
        startPos = [0, 0, obj.height + .02]
        startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
        f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1,
                          flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        finger_id.append(f_id)

    p.setRealTimeSimulation(0)

    for i in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f)):
                limit = p.getJointInfo(f, j)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL, targetPosition=min(0.01 * i, .95 * limit))
        p.stepSimulation()
        # time.sleep(1. / 240.)

    for _ in range(500):
        for f in finger_id:
            for j in range(p.getNumJoints(f)):
                limit = p.getJointInfo(f, j)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL, targetPosition=.95 * limit)
            p.resetBaseVelocity(f, [0, 0, .05])
        p.stepSimulation()
        # time.sleep(1. / 240.)

    objPos, _ = p.getBasePositionAndOrientation(box_id)

    # p.disconnect()
    p.resetSimulation()
    # end pybullet test

    return objPos


import pickle
from GeometryUtils import ContactPoints
from GripperModel import initialize_gripper

if __name__ == "__main__":
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)

    test_obj_urdf = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle.urdf")
    cps = ContactPoints(test_obj, [287, 1286, 1821, 2036])
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .02])
    _, fingers = initialize_gripper(cps, end_effector_pos, 4, width=20.)
    gripper = FOAMGripper(fingers)

    physicsClient = p.connect(p.DIRECT)
    final_pos = gripper_sim(test_obj, test_obj_urdf, gripper)
    print(final_pos)
    p.disconnect()
