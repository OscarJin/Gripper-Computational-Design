import hashlib
import os
import time

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

    def clean(self):
        os.remove(self.filename)
        for i, u in enumerate(self.units):
            if os.path.exists(u.filename):
                os.remove(u.filename)

    def assemble(self, bottom_thick=.4, export=True):
        totLength = self.units[0].gap
        finger_vertices = self.units[0].vertices.copy()
        finger_vertices[:, 0] += totLength
        finger_faces = self.units[0].faces.copy()
        for i in range(1, self.n_unit):
            cur_vertices = self.units[i].vertices.copy()
            totLength += self.units[i - 1].length + self.units[i].gap
            cur_vertices[:, 0] += totLength
            finger_vertices = np.concatenate((finger_vertices, cur_vertices), axis=0)
            cur_faces = self.units[i].faces + self.units[i].vertices.shape[0] * i
            finger_faces = np.concatenate((finger_faces, cur_faces), axis=0)

        totLength += self.units[-1].length
        bottom_width = self.units[0].width
        bottom_v = np.asarray([
            [0, -bottom_width / 2, -bottom_thick],
            [totLength, -bottom_width / 2, -bottom_thick],
            [totLength, bottom_width / 2, -bottom_thick],
            [0, bottom_width / 2, -bottom_thick],
            [0, -bottom_width / 2, 0],
            [totLength, -bottom_width / 2, 0],
            [totLength, bottom_width / 2, 0],
            [0, bottom_width / 2, 0],
        ])
        bottom_f = self.units[0].faces + self.units[0].vertices.shape[0] * self.n_unit
        finger_vertices = np.concatenate((finger_vertices, bottom_v), axis=0)
        finger_faces = np.concatenate((finger_faces, bottom_f), axis=0)

        finger_mesh = mesh.Mesh(np.zeros(finger_faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(finger_faces):
            for j in range(3):
                finger_mesh.vectors[i][j] = finger_vertices[f[j], :]

        if export:
            stl_file = os.path.join(os.path.abspath('..'), "assets/finger_" + self.id + ".stl")
            finger_mesh.save(stl_file, mode=stl.Mode.ASCII)

        return finger_vertices, finger_faces


class FOAMGripper:
    def __init__(self, fingers: List[Finger]):
        self.n_finger = len(fingers)
        self.fingers = fingers

    def assemble(self, export=True):
        gripper_v, gripper_f = self.fingers[0].assemble(export=False)
        gripper_v = self.rotate(gripper_v, self.fingers[0].orientation)
        for i in range(1, self.n_finger):
            cur_v, cur_f = self.fingers[i].assemble(export=False)
            cur_v = self.rotate(cur_v, self.fingers[i].orientation)
            cur_f += gripper_v.shape[0]
            gripper_v = np.concatenate((gripper_v, cur_v), axis=0)
            gripper_f = np.concatenate((gripper_f, cur_f), axis=0)

        cylinder_v, cylinder_f = self.create_cylinder(radius=0.8 * self.min_distance_to_center,
                                                      z1=self.max_unit_height)
        cylinder_f += gripper_v.shape[0]
        gripper_v = np.concatenate((gripper_v, cylinder_v), axis=0)
        gripper_f = np.concatenate((gripper_f, cylinder_f), axis=0)

        gripper_mesh = mesh.Mesh(np.zeros(gripper_f.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(gripper_f):
            for j in range(3):
                gripper_mesh.vectors[i][j] = gripper_v[f[j], :]

        if export:
            stl_file = os.path.join(os.path.abspath('..'), "assets/gripper_" + _create_id() + ".stl")
            gripper_mesh.save(stl_file, mode=stl.Mode.ASCII)

        return gripper_v, gripper_f
        pass

    @staticmethod
    def rotate(v, theta):
        rotate_matrix = np.asarray([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return v @ rotate_matrix.T
        pass

    @staticmethod
    def create_cylinder(radius, z1, z2=-20, resolution=100):
        phi = np.linspace(0, 2 * np.pi, resolution)
        z = np.linspace(z2, z1, resolution)
        z_grid, phi_grid = np.meshgrid(z, phi)

        x = radius * np.cos(phi_grid)
        y = radius * np.sin(phi_grid)
        vertices = np.vstack([x.flatten(), y.flatten(), z_grid.flatten()]).T

        # Generate the triangles
        faces = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v1 = i * resolution + j
                v2 = (i + 1) * resolution + j
                v3 = (i + 1) * resolution + j + 1
                v4 = i * resolution + j + 1

                faces.extend([[v1, v2, v3], [v3, v4, v1]])
        faces = np.asarray(faces)
        return vertices, faces

    @property
    def min_distance_to_center(self):
        d_min = self.fingers[0].units[0].gap
        for i, f in enumerate(self.fingers):
            d_min = min(d_min, f.units[0].gap)
        return d_min

    @property
    def max_unit_height(self):
        h_max = 0
        for i, f in enumerate(self.fingers):
            for j, u in enumerate(f.units):
                h_max = max(h_max, u.height)
        return h_max

    def clean(self):
        for i, f in enumerate(self.fingers):
            f.clean()


import pybullet as p
import pybullet_data

if __name__ == "__main__":
    # test
    unit = Unit(20, 5, 10, np.pi / 3, np.pi / 3, 5)
    unit_root = Unit(20, 5, 10, np.pi / 3, np.pi / 3, 15)
    finger_1 = Finger([unit_root, unit, unit], 0)
    finger_2 = Finger([unit_root, unit, unit], np.pi / 2)
    finger_3 = Finger([unit_root, unit, unit], np.pi)
    finger_4 = Finger([unit_root, unit, unit], -np.pi / 2)
    gripper = FOAMGripper([finger_1, finger_2, finger_3, finger_4])

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

    startPos = [0, 0., 0.]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF(os.path.join(os.path.abspath('..'), "assets/ycb/007_tuna_fish_can.urdf"), startPos,
                        startOrientation,
                        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    p.performCollisionDetection()

    for i in range(500):
        for i, f in enumerate(finger_id):
            for j in range(2):
                p.setJointMotorControl2(f, j, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        objPos, _ = p.getBasePositionAndOrientation(box_id)
        p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=180, cameraPitch=-30,
                                     cameraTargetPosition=[0, 0., .05 + objPos[2]])
        p.stepSimulation()
        time.sleep(1. / 240.)

    for i in range(500):
        for _, f in enumerate(finger_id):
            for j in range(2):
                p.setJointMotorControl2(f, j, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
            p.resetBaseVelocity(f, [0, 0, .05])
        objPos, _ = p.getBasePositionAndOrientation(box_id)
        p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=180, cameraPitch=-30,
                                     cameraTargetPosition=[0, 0., .05 + objPos[2]])
        p.stepSimulation()
        time.sleep(1. / 240.)

    cps = p.getContactPoints(box_id)
    for _, cp in enumerate(cps):
        print(cp[9])    # normal force
    p.disconnect()
    gripper.clean()

    # gripper.assemble()
