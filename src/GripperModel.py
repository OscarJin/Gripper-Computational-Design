import hashlib
import os
import time

import numpy as np
import stl
from stl import mesh
import trimesh
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

    def assemble(self, bottom_thick=1.2, export=True):
        # units
        totLength = self.units[0].gap
        finger_meshes = []
        for i in range(self.n_unit):
            cur_v = self.units[i].vertices.copy()
            if i != 0:
                totLength += self.units[i - 1].length + self.units[i].gap
            cur_v[:, 0] += totLength
            cur_f = self.units[i].faces.copy()
            cur_mesh = trimesh.Trimesh(vertices=cur_v, faces=cur_f)
            finger_meshes.append(cur_mesh)

        totLength += self.units[-1].length
        bottom_width = self.units[0].width
        sub_length = np.linspace(0, totLength, 2 * self.n_unit)
        for i in range(2 * self.n_unit - 1):
            bottom_v = np.asarray([
                [sub_length[i], -bottom_width / 2, -bottom_thick],
                [sub_length[i + 1], -bottom_width / 2, -bottom_thick],
                [sub_length[i + 1], bottom_width / 2, -bottom_thick],
                [sub_length[i], bottom_width / 2, -bottom_thick],
                [sub_length[i], -bottom_width / 2, 0],
                [sub_length[i + 1], -bottom_width / 2, 0],
                [sub_length[i + 1], bottom_width / 2, 0],
                [sub_length[i], bottom_width / 2, 0],
            ])
            bottom_f = self.units[0].faces
            bottom_mesh = trimesh.Trimesh(vertices=bottom_v, faces=bottom_f)
            finger_meshes.append(bottom_mesh)

        finger_mesh = trimesh.boolean.union(finger_meshes)

        # trench
        h1 = self.min_unit_height - 1.
        h2 = self.max_unit_height
        trench_w = 1.
        trench_v = np.asarray([
            [0, -trench_w / 2, h1],
            [totLength, -trench_w / 2, h1],
            [totLength, trench_w / 2, h1],
            [0, trench_w / 2, h1],
            [0, -trench_w / 2, h2],
            [totLength, -trench_w / 2, h2],
            [totLength, trench_w / 2, h2],
            [0, trench_w / 2, h2],
        ])
        trench_f = self.units[0].faces.copy()
        trench_mesh = trimesh.Trimesh(vertices=trench_v, faces=trench_f)

        finger_mesh = trimesh.boolean.difference([finger_mesh, trench_mesh])

        if export:
            stl_file = os.path.join(os.path.abspath('..'), "assets/finger_" + self.id + ".stl")
            finger_mesh.export(stl_file)

        return finger_mesh

    @property
    def max_unit_height(self):
        h_max = 0
        for u in self.units:
            h_max = max(u.height, h_max)
        return h_max

    @property
    def min_unit_height(self):
        h_min = self.units[0].height
        for u in self.units:
            h_min = min(u.height, h_min)
        return h_min

    @property
    def total_length(self):
        tot_l = self.units[0].gap
        for i in range(1, self.n_unit):
            tot_l += self.units[i - 1].length + self.units[i].gap
        tot_l += self.units[-1].length
        return tot_l

    @property
    def max_unit_width(self):
        w_max = 0
        for u in self.units:
            w_max = max(u.width, w_max)
        return w_max

    def mask(self, extend=20.):
        b_w = self.max_unit_width + extend * 2
        b_l = self.total_length + extend
        b_h = 15.
        b_v = np.asarray([
            [0, -b_w / 2, 0],
            [b_l, -b_w / 2, 0],
            [b_l, b_w / 2, 0],
            [0, b_w / 2, 0],
            [0, -b_w / 2, b_h],
            [b_l, -b_w / 2, b_h],
            [b_l, b_w / 2, b_h],
            [0, b_w / 2, b_h],
        ])
        b_f = self.units[0].faces.copy()

        return trimesh.Trimesh(vertices=b_v, faces=b_f)


class FOAMGripper:
    def __init__(self, fingers: List[Finger]):
        self.n_finger = len(fingers)
        self.fingers = fingers

    def assemble(self, export=True, bottom_thick=1.2, palm_height=10., palm_ratio=1.2):
        gripper_meshes = []
        for i in range(self.n_finger):
            cur_mesh = self.fingers[0].assemble(bottom_thick=bottom_thick, export=False)
            cur_v = cur_mesh.vertices
            cur_v = self.rotate(cur_v, self.fingers[i].orientation)
            cur_f = cur_mesh.faces
            cur_mesh = trimesh.Trimesh(vertices=cur_v, faces=cur_f)
            gripper_meshes.append(cur_mesh)

        cylinder_v, cylinder_f = self.create_cylinder(radius=palm_ratio * self.min_distance_to_center,
                                                      z1=self.max_unit_height, z2=-palm_height)
        cylinder_mesh = trimesh.Trimesh(vertices=cylinder_v, faces=cylinder_f)
        gripper_meshes.append(cylinder_mesh)

        gripper_mesh = trimesh.boolean.union(gripper_meshes)

        if export:
            stl_file = os.path.join(os.path.abspath('..'), "assets/gripper_" + _create_id() + ".stl")
            gripper_mesh.export(stl_file)

        return gripper_mesh

    @staticmethod
    def rotate(v, theta):
        rotate_matrix = np.asarray([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return v @ rotate_matrix.T

    @staticmethod
    def create_cylinder(radius, z1, z2=-10., resolution=100):
        phi = np.linspace(0, 2 * np.pi, resolution + 1)
        phi = phi[: -1]
        z = np.asarray([z2, z1])
        z_grid, phi_grid = np.meshgrid(z, phi)

        x = radius * np.cos(phi_grid)
        y = radius * np.sin(phi_grid)
        z = z_grid
        side_v = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        top_v = np.asarray([0, 0, z1])
        bottom_v = np.asarray([0, 0, z2])
        vertices = np.vstack([side_v, top_v, bottom_v])

        side_f = []
        for i in range(resolution):
            v1 = i * 2
            v2 = ((i + 1) % resolution) * 2
            v3 = v2 + 1
            v4 = v1 + 1
            side_f.extend([[v1, v2, v3], [v3, v4, v1]])

        bottom_f = [[i * 2, -1, ((i + 1) % resolution) * 2] for i in range(resolution)]
        top_f = [[i * 2 + 1, ((i + 1) % resolution) * 2 + 1, -2] for i in range(resolution)]

        faces = np.vstack([np.asarray(side_f), np.asarray(top_f), np.asarray(bottom_f)])

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

    def seal_mask(self, export=True, extend=20., wall_thick=4.):
        finger_masks = []
        for f in self.fingers:
            cur_mesh = f.mask(extend=extend)
            cur_v = cur_mesh.vertices
            cur_v = self.rotate(cur_v, f.orientation)
            finger_masks.append(trimesh.Trimesh(vertices=cur_v, faces= cur_mesh.faces))
        inner_mask = trimesh.boolean.union(finger_masks)

        outer_v = inner_mask.vertices.copy()
        outer_f = inner_mask.faces
        for v in outer_v:
            v[0] += wall_thick * np.sign(v[0])
            v[1] += wall_thick * np.sign(v[1])
        outer_mask = trimesh.Trimesh(vertices=outer_v, faces=outer_f)

        mask = trimesh.boolean.difference([outer_mask, inner_mask])

        if export:
            stl_file = os.path.join(os.path.abspath('..'), "assets/gripper_mask_" + _create_id() + ".stl")
            mask.export(stl_file)

        return mask


import pybullet as p
import pybullet_data

if __name__ == "__main__":
    # test
    unit = Unit(20., 7.5, 20., np.pi / 3, np.pi / 3, 5.)
    unit_root = Unit(20., 5., 20., np.pi / 3, np.pi / 3, 15.)
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

    for i in range(50):
        for i, f in enumerate(finger_id):
            for j in range(2):
                p.setJointMotorControl2(f, j, p.VELOCITY_CONTROL, targetVelocity=1., force=1.5)
        objPos, _ = p.getBasePositionAndOrientation(box_id)
        p.resetDebugVisualizerCamera(cameraDistance=.2, cameraYaw=180, cameraPitch=-30,
                                     cameraTargetPosition=[0, 0., .05 + objPos[2]])
        p.stepSimulation()
        time.sleep(1. / 240.)

    for i in range(50):
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
    for cp in cps:
        print(cp[9])  # normal force
    p.disconnect()
    gripper.clean()

    # gripper.assemble(bottom_thick=1., palm_height=1.)
    gripper.seal_mask()
