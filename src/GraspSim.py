import pybullet as p
import time
import pybullet_data
import numpy as np

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0., 4.5]
    startOrientation = p.getQuaternionFromEuler([np.pi, 0, 0])
    boxId = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/foam-2.urdf", startPos,
                       startOrientation, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    p.setRealTimeSimulation(0)
    p.createConstraint(planeId, -1, boxId, -1, p.JOINT_PRISMATIC, [0, 0, 1], startPos, [0, 0, 0], startOrientation)
    p.enableJointForceTorqueSensor(boxId, 0, 1)
    p.performCollisionDetection()
    p.resetDebugVisualizerCamera(cameraDistance=10.0, cameraYaw=180, cameraPitch=-5, cameraTargetPosition=startPos)

    for i in range(500):
        p.resetBaseVelocity(boxId, [0, 0, 1.])
        p.applyExternalForce(boxId, 0, [0, 0, 300], [1, 1, 0], p.LINK_FRAME)
        p.applyExternalForce(boxId, 1, [0, 0, 300], [1, 1, 0], p.LINK_FRAME)
        p.stepSimulation()
        time.sleep(1. / 240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
    print(p.getJointState(boxId, 0)[0], p.getJointState(boxId, 1)[0])
    print(cubePos, cubeOrn)
    p.disconnect()
