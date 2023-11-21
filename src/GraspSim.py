import pybullet as p
import time
import pybullet_data
import numpy as np

if __name__ == "__main__":
    physicsClient = p.connect(p.GUI)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.8)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    startPos = [0, 0., 5]
    startOrientation = p.getQuaternionFromEuler([np.pi, 0, 0])
    finger1_id = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/foam-2.urdf", startPos,
                            startOrientation,
                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    p.createConstraint(planeId, -1, finger1_id, -1, p.JOINT_PRISMATIC, [0, 0, 1], startPos, [0, 0, 0], startOrientation)

    startPos = [-.2, -2, 5]
    startOrientation = p.getQuaternionFromEuler([np.pi, 0, np.pi])
    finger2_id = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/foam-2.urdf", startPos,
                            startOrientation,
                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    # p.createConstraint(planeId, -1, finger2_id, -1, p.JOINT_PRISMATIC, [0, 0, 1], startPos, [0, 0, 0], startOrientation)
    p.createConstraint(finger1_id, -1, finger2_id, -1, p.JOINT_FIXED, [1, 1, 1], [0, 0, 0], [0, 1, 0],
                       p.getQuaternionFromEuler([0, 0, np.pi]))
    startPos = [-1.3, -1.3, 0.]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/cube.urdf", startPos,
                        startOrientation, flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    # p.enableJointForceTorqueSensor(finger1_id, 0, 1)
    p.performCollisionDetection()
    p.resetDebugVisualizerCamera(cameraDistance=10.0, cameraYaw=180, cameraPitch=-40, cameraTargetPosition=startPos)

    for i in range(50):
        p.stepSimulation()
        p.resetBaseVelocity(finger1_id, [0, 0, 0.])
        time.sleep(1. / 240.)

    for i in range(500):
        p.resetBaseVelocity(finger1_id, [0, 0, 1.])
        p.applyExternalForce(finger1_id, 0, [0, 0, 250], [1, 1, 0], p.LINK_FRAME)
        p.applyExternalForce(finger1_id, 1, [0, 0, 250], [1, 1, 0], p.LINK_FRAME)
        p.applyExternalForce(finger2_id, 0, [0, 0, 250], [1, 1, 0], p.LINK_FRAME)
        p.applyExternalForce(finger2_id, 1, [0, 0, 250], [1, 1, 0], p.LINK_FRAME)
        p.stepSimulation()
        time.sleep(1. / 240.)
    cubePos, cubeOrn = p.getBasePositionAndOrientation(box_id)
    print(cubePos, cubeOrn)
    p.disconnect()
