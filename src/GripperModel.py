import os
import hashlib
import time

import math
import numpy as np
import stl
from stl import mesh
import trimesh
from typing import List
from GripperInitialization import initialize_fingers, compute_skeleton
from GeometryUtils import ContactPoints


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

        # clamp height
        if self.length - self.height / np.tan(self.theta1) - self.height / np.tan(self.theta2) < 0:
            self.height = self.length / (1 / np.tan(self.theta1) + 1 / np.tan(self.theta2)) - .4

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
        self.filename = os.path.join(os.path.abspath('..'), "_cache_/unit_" + self.id + ".stl")
        self.mesh.save(self.filename, mode=stl.Mode.BINARY)

    @property
    def mass(self):
        rho = .0017637  # tpu 95, g/mm3
        V = ((self.length - self.height * (1. / np.tan(self.theta1) + 1. / np.tan(self.theta2)) / 2.)
             * self.height * self.width)
        return rho * V / 1000.  # kg

    def __del__(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)


class Finger:
    def __init__(self, units: List[Unit], orientation):
        self.n_unit = len(units)
        self.units = units
        self.orientation = orientation
        self.id = _create_id()
        self.filename = os.path.join(os.path.abspath('..'), "_cache_/finger_" + self.id + ".urdf")
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
            origin /= 1000.
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
<inertial>
    <mass value="{u.mass}"/>
    <inertia ixx="0." ixy="0." ixz="0.0" iyy="0." iyz="0." izz="0."/>
</inertial>
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
            #
            link = f"""
<joint name="joint_{i}_{i + 1}" type="revolute">
    <origin xyz="{origin} 0 0"/>
    <parent link="link_{i}"/>
    <child link="link_{i + 1}"/>
    <axis xyz="0 1 0"/>
    <limit lower="0." upper="{self.calc_joint_limit(self.units[i], self.units[i + 1])}" effort="10."/>
    <dynamics stiffness="100." damping="0.001" />
</joint>
"""
            urdf.write(link)
            urdf.write("\r\n")

        urdf.write("</robot>")

    def __del__(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def __repr__(self):
        return repr({'id': self.id, 'orientation': int(self.orientation / np.deg2rad(22.5)) * 22.5})

    @staticmethod
    def calc_joint_limit(unit1: Unit, unit2: Unit):
        d1 = math.sqrt(math.pow(unit1.height, 2) + math.pow(unit1.height / math.tan(unit1.theta2) + unit2.gap / 2, 2))
        d2 = math.sqrt(math.pow(unit2.height, 2) + math.pow(unit2.height / math.tan(unit2.theta1) + unit2.gap / 2, 2))
        theta1 = math.acos(unit1.height / d1)
        theta2 = math.acos(unit2.height / d2)

        return theta1 + theta2

    def assemble(self, bottom_thick=1.2, export=True):
        # units
        totLength = self.units[0].gap
        finger_meshes = []
        for i in range(self.n_unit):
            cur_v = self.units[i].vertices.copy()
            # bottom
            w1 = self.units[i].width
            w2 = w1 if i == self.n_unit - 1 else self.units[i + 1].width
            bottom_length = self.units[i].length
            if i != self.n_unit - 1:
                bottom_length += self.units[i + 1].gap

            bottom_v = np.asarray([
                [0, -w1 / 2, -bottom_thick],
                [bottom_length, -w2 / 2, -bottom_thick],
                [bottom_length, w2 / 2, -bottom_thick],
                [0, w1 / 2, -bottom_thick],
                [0, -w1 / 2, 0],
                [bottom_length, -w2 / 2, 0],
                [bottom_length, w2 / 2, 0],
                [0, w1 / 2, 0],
            ])

            # translation
            if i != 0:
                totLength += self.units[i - 1].length + self.units[i].gap
            cur_v[:, 0] += totLength
            bottom_v[:, 0] += totLength

            cur_f = self.units[i].faces.copy()
            cur_mesh = trimesh.Trimesh(vertices=cur_v, faces=cur_f)
            cur_bottom_mesh = trimesh.Trimesh(vertices=bottom_v, faces=cur_f)
            finger_meshes.append(cur_mesh)
            finger_meshes.append(cur_bottom_mesh)

        totLength += self.units[-1].length

        finger_mesh = trimesh.boolean.union(finger_meshes)

        # trench
        h1 = 0.
        h2 = min(h1 + 2., self.max_unit_height - 1.)
        trench_w = min(3., self.min_unit_width / 2)
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
            stl_file = os.path.join(os.path.abspath('..'), "results/finger_" + self.id + ".stl")
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

    @property
    def min_unit_width(self):
        w_min = self.units[0].width
        for u in self.units:
            w_min = min(u.width, w_min)
        return w_min

    def mask(self, extend=10.):
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
        self.id = _create_id()

    def assemble(self, export=True, bottom_thick=1.2):
        gripper_meshes = []
        for f in self.fingers:
            f.assemble(bottom_thick=bottom_thick, export=export)
            print(f)

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
        d_min = np.inf
        for i, f in enumerate(self.fingers):
            d_min = min(d_min, f.units[0].gap)
        return d_min

    @property
    def max_unit_height(self):
        h_max = 0.
        for f in self.fingers:
            for u in f.units:
                h_max = max(h_max, u.height)
        return h_max

    @property
    def min_unit_height(self):
        h_min = self.fingers[0].units[0].height
        for f in self.fingers:
            for u in f.units:
                h_min = min(h_min, u.height)
        return h_min

    def clean(self):
        for f in self.fingers:
            for u in f.units:
                if os.path.exists(u.filename):
                    os.remove(u.filename)
            if os.path.exists(f.filename):
                os.remove(f.filename)


def compute_unit_min_right_angle(length: float, height: float, l_angle: float) -> float:
    l_r = length - height / np.tan(l_angle) if not np.isclose(l_angle, np.pi / 2) else length
    if l_r > 1:
        return np.arctan2(height, l_r - 1)
    else:
        return np.pi / 2


def initialize_gripper(
        cps: ContactPoints, effector_pos,
        n_finger_joints: int,
        expand_dist=30.,
        root_length=30.,
        height_ratio=.5,
        width=20.,
        gap=2.,
        finger_skeletons=None,
):
    if finger_skeletons is None:
        finger_skeletons = initialize_fingers(cps, effector_pos, n_finger_joints, expand_dist / 1000, root_length / 1000)
    L, angle, ori = compute_skeleton(finger_skeletons, cps, effector_pos, n_finger_joints)
    angle *= .95
    unit_h = expand_dist * height_ratio
    fingers: List[Finger] = []

    for i, f in enumerate(finger_skeletons):
        n_joints = np.sum(~np.isnan(f).any(axis=1))
        units: List[Unit] = []

        for j in range(n_finger_joints - n_joints + 1, n_finger_joints):
            if j == n_finger_joints - n_joints + 1:
                # root
                r_angle = compute_unit_min_right_angle(L[i][j] - 30 - gap / 2., unit_h, np.pi / 2)
                u = Unit(L[i][j] - 20 - gap / 2., unit_h, width, np.pi / 2,
                         max(r_angle, angle[i][j] - np.deg2rad(85)), 20)
            elif j == n_finger_joints - 1:
                # end
                u = Unit(L[i][j] - gap / 2, unit_h, width, angle[i][j - 1] - units[-1].theta2, np.pi / 2, gap)
            else:
                l_angle = angle[i][j - 1] - units[-1].theta2
                r_angle = compute_unit_min_right_angle(L[i][j] - gap, unit_h, l_angle)
                u = Unit(L[i][j] - gap, unit_h, width, angle[i][j - 1] - units[-1].theta2,
                         max(r_angle, angle[i][j] - np.deg2rad(85)), gap)
            units.append(u)

        fingers.append(Finger(units, orientation=round(ori[i] * 8 / np.pi) * np.pi / 8))

    return finger_skeletons, fingers
