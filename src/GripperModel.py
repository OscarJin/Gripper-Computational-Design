import hashlib
import os
import time

import numpy
import numpy as np
import stl
from stl import mesh
from typing import List


def _create_id():
    m = hashlib.md5(str(time.perf_counter()).encode("utf-8"))
    return m.hexdigest()


class Unit:
    def __init__(self, length, height, width, theta1, theta2, gap):
        self.length = length
        self.height = height
        self.width = width
        self.theta1 = theta1
        self.theta2 = theta2
        self.gap = gap

        # define 8 vertices
        self.vertices = np.asarray([
            [0, -self.width / 2, 0],
            [self.length, -self.width / 2, 0],
            [self.length, self.width / 2, 0],
            [0, self.width / 2, 0],
            [self.height / np.tan(self.theta1), -self.width / 2, self.height],
            [self.length - self.height / np.tan(self.theta2), -self.width / 2, self.height],
            [self.length - self.height / np.tan(self.theta2), self.width / 2, self.height],
            [self.height / np.tan(self.theta1), self.width / 2, self.height],
        ])
        self.faces = np.asarray([
            [0, 3, 1],
            [1, 3, 2],
            [0, 4, 7],
            [0, 7, 3],
            [4, 5, 6],
            [4, 6, 7],
            [5, 1, 2],
            [5, 2, 6],
            [2, 3, 6],
            [3, 7, 6],
            [0, 1, 5],
            [0, 5, 4]
        ])

        # create mesh
        self.mesh = mesh.Mesh(np.zeros(self.faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(self.faces):
            for j in range(3):
                self.mesh.vectors[i][j] = self.vertices[f[j], :]

        self.id = _create_id()
        self.filename = os.path.join(os.path.abspath('..'), "assets/unit_" + self.id + ".stl")
        self.mesh.save(self.filename, mode=stl.Mode.BINARY)


class Finger:
    def __init__(self, units: List[Unit], orientation):
        self.n_unit = len(units)
        self.units = units
        self.orientation = orientation
        self.id = _create_id()
        self.filename = os.path.join(os.path.abspath('..'), "assets/finger_" + self.id + ".urdf")
        self._create_urdf()

    def _create_urdf(self):
        urdf = open(self.filename, "a")
        urdf.write("<?xml version=\"1.0\"?>\r\n")
        urdf.write(f"<robot name=\"finger_{self.id}\">\r\n")
        urdf.write("\r\n")
        # write links
        for i, u in enumerate(self.units):
            header = f"<link name=\"link_{i}\">\r\n"
            urdf.write(header)
            origin = u.gap if i == 0 else u.gap / 2
            origin /= 1000
            visual = f"""
<visual>
    <origin xyz="{origin} 0 0" rpy="{np.pi} 0 0"/>
    <geometry>
        <mesh filename="unit_{u.id}.stl" scale="0.001 0.001 0.001"/>
    </geometry>
    <material name="blue">
        <color rgba="0 0 .8 1"/>
    </material>
</visual>
"""
            urdf.write(visual)

            if i != 0:
                collision = f"""
<collision>
    <origin xyz="{origin} 0 0" rpy="{np.pi} 0 0"/>
    <geometry>
        <mesh filename="unit_{u.id}.stl" scale="0.001 0.001 0.001"/>
    </geometry>
</collision>
"""
                urdf.write(collision)

            urdf.write("</link>\r\n")
            urdf.write("\r\n")

        # write joints
        for i in range(self.n_unit - 1):
            # effort, velocity will not be read by pybullet
            # upper limit needs to be further determined
            origin = self.units[i].gap if i == 0 else self.units[i].gap / 2
            origin += self.units[i].length + self.units[i + 1].gap / 2
            origin /= 1000
            link = f"""
<joint name="joint_{i}_{i + 1}" type="revolute">
    <origin xyz="{origin} 0 0"/>
    <parent link="link_{i}"/>
    <child link="link_{i + 1}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-.1" upper="1.3" effort="300" velocity="1"/>
</joint>
"""
            urdf.write(link)
            urdf.write("\r\n")

        urdf.write("</robot>")
        pass

    def clean(self):
        os.remove(self.filename)
        for i, u in enumerate(self.units):
            if os.path.exists(u.filename):
                os.remove(u.filename)


class FOAMGripper:
    def __init__(self, fingers: List[Finger]):
        self.n_finger = len(fingers)
        self.fingers = fingers

    def clean(self):
        for i, f in enumerate(self.fingers):
            f.clean()


import pybullet as p
import pybullet_data

if __name__ == "__main__":
    # test
    unit = Unit(20, 5, 10, np.pi / 3, np.pi / 3, 5)
    finger_1 = Finger([unit, unit, unit], 0)
    finger_2 = Finger([unit, unit, unit], np.pi)
    gripper = FOAMGripper([finger_1, finger_2])

    physicsClient = p.connect(p.GUI)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, -9.8)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    finger_id = []
    for i, f in enumerate(gripper.fingers):
        startPos = [0, 0., .05]
        startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
        f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1,
                                  flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        finger_id.append(f_id)

    startPos = [0, 0., .0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF(os.path.join(os.path.abspath('..'), "assets/cube.urdf"), startPos,
                        startOrientation,
                        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    p.performCollisionDetection()
    p.resetDebugVisualizerCamera(cameraDistance=.1, cameraYaw=180, cameraPitch=-30, cameraTargetPosition=[0, 0., .05])
    print(finger_id[0])
    for i in range(500):
        p.setJointMotorControl2(finger_id[0], 0, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.setJointMotorControl2(finger_id[0], 1, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.setJointMotorControl2(finger_id[1], 0, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.setJointMotorControl2(finger_id[1], 1, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        p.stepSimulation()
        time.sleep(1. / 240.)
    p.disconnect()
    gripper.clean()
