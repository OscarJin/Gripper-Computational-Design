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

    startPos = [0, 0., .05]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    finger1_id = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/foam-2.urdf", startPos,
                            startOrientation, useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    startOrientation = p.getQuaternionFromEuler([0, 0, np.pi])
    finger2_id = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/foam-2.urdf", startPos,
                            startOrientation, useFixedBase=1,
                            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    startPos = [0, 0., .0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF("E:/SGLab/Dissertation/Gripper-Computational-Design/assets/cube.urdf", startPos,
                        startOrientation,
                        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    # p.enableJointForceTorqueSensor(finger1_id, 0, 1)
    # p.enableJointForceTorqueSensor(finger1_id, 1, 1)
    p.performCollisionDetection()
    p.resetDebugVisualizerCamera(cameraDistance=.1, cameraYaw=180, cameraPitch=-30, cameraTargetPosition=startPos)

    for i in range(500):
        p.setJointMotorControl2(finger1_id, 0, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.setJointMotorControl2(finger1_id, 1, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.setJointMotorControl2(finger2_id, 0, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.setJointMotorControl2(finger2_id, 1, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.stepSimulation()
        time.sleep(1. / 240.)

    for i in range(500):
        # p.setJointMotorControl2(finger1_id, 0, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        # p.setJointMotorControl2(finger1_id, 1, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        # p.setJointMotorControl2(finger2_id, 0, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        # p.setJointMotorControl2(finger2_id, 1, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.resetBaseVelocity(finger1_id, [0, 0, .005])
        p.resetBaseVelocity(finger2_id, [0, 0, .005])
        p.stepSimulation()
        time.sleep(1. / 240.)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(box_id)
    # print(cubePos, cubeOrn)
    p.disconnect()
