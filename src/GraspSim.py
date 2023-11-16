import pybullet as p
import time
import pybullet_data
import numpy as np

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0.]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    boxId = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/foam-2.urdf", startPos,
                       startOrientation, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    print(boxId)
    p.setRealTimeSimulation(0)
    p.createConstraint(planeId, -1, boxId, -1, p.JOINT_FIXED, [1, 1, 1], [0, 0, 0], [0, 0, 0])
    p.enableJointForceTorqueSensor(boxId, 0, 1)
    p.performCollisionDetection()

    for i in range(500):
        p.stepSimulation()
        p.applyExternalForce(boxId, 0, [0, 0, 300], [1, 1, 0], p.LINK_FRAME)
        time.sleep(1. / 240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(cubePos, cubeOrn)
    p.disconnect()
