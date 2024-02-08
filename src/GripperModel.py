import hashlib
import os
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
        self.filename = os.path.join(os.path.abspath('..'), "assets/unit_" + self.id + ".stl")
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
    <limit lower="0." upper="{self.calc_joint_limit(self.units[i], self.units[i + 1])}" effort="10." velocity=".1"/>
    <dynamics stiffness="100." damping="0.001" />
</joint>
"""
            urdf.write(link)
            urdf.write("\r\n")

        urdf.write("</robot>")

    def __del__(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

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

    def assemble(self, export=True, bottom_thick=1.2, palm_ratio=1.2):
        gripper_meshes = []
        for i in range(self.n_finger):
            cur_mesh = self.fingers[i].assemble(bottom_thick=bottom_thick, export=False)
            cur_v = cur_mesh.vertices
            cur_v = self.rotate(cur_v, self.fingers[i].orientation)
            cur_f = cur_mesh.faces
            cur_mesh = trimesh.Trimesh(vertices=cur_v, faces=cur_f)
            gripper_meshes.append(cur_mesh)

        # palm
        t_h = min(3., self.min_unit_height - 1.)
        hPalm = self.min_unit_height
        r_palm = palm_ratio * self.min_distance_to_center
        cylinder_v, cylinder_f = self.create_cylinder(radius=r_palm, z1=hPalm, z2=-bottom_thick)
        cylinder_mesh = trimesh.Trimesh(vertices=cylinder_v, faces=cylinder_f)
        gripper_meshes.append(cylinder_mesh)

        gripper_mesh = trimesh.boolean.union(gripper_meshes)

        # phi8.4 hole
        r = min(4.3, palm_ratio * self.min_distance_to_center - .4)

        cylinder_v, cylinder_f = self.create_cylinder(radius=r, z1=t_h, z2=-bottom_thick)
        cylinder_mesh = trimesh.Trimesh(vertices=cylinder_v, faces=cylinder_f)
        gripper_mesh = trimesh.boolean.difference([gripper_mesh, cylinder_mesh])

        t_w = 3.
        for f in self.fingers:
            t_h = min(3., f.min_unit_width / 2)
            t_v = np.asarray([
                [0, -t_w / 2, 0],
                [r_palm, -t_w / 2, 0],
                [r_palm, t_w / 2, 0],
                [0, t_w / 2, 0],
                [0, -t_w / 2, t_h],
                [r_palm, -t_w / 2, t_h],
                [r_palm, t_w / 2, t_h],
                [0, t_w / 2, t_h],
            ])
            t_v = self.rotate(t_v, f.orientation)
            t_f = f.units[0].faces.copy()
            t_mesh = trimesh.Trimesh(vertices=t_v, faces=t_f)
            gripper_mesh = trimesh.boolean.difference([gripper_mesh, t_mesh])

        if export:
            stl_file = os.path.join(os.path.abspath('..'), "assets/gripper_" + self.id + ".stl")
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

    def seal_mask(self, export=True, extend=10., wall_thick=4.):
        inner_masks = []
        for f in self.fingers:
            cur_mesh = f.mask(extend=extend)
            cur_v = cur_mesh.vertices
            cur_v = self.rotate(cur_v, f.orientation)
            inner_masks.append(trimesh.Trimesh(vertices=cur_v, faces=cur_mesh.faces))
        inner_mask = trimesh.boolean.union(inner_masks)

        outer_masks = []
        for f in self.fingers:
            cur_mesh = f.mask(extend=extend + wall_thick)
            cur_v = cur_mesh.vertices
            cur_v = self.rotate(cur_v, f.orientation)
            outer_masks.append(trimesh.Trimesh(vertices=cur_v, faces=cur_mesh.faces))

        outer_mask = trimesh.boolean.union(outer_masks)

        mask = trimesh.boolean.difference([outer_mask, inner_mask])

        if export:
            stl_file = os.path.join(os.path.abspath('..'), "assets/gripper_mask_" + self.id + ".stl")
            mask.export(stl_file)

        return mask


def initialize_gripper(
        cps: ContactPoints, effector_pos,
        n_finger_joints: int,
        height=20.,
        width=20.,
        gap=2.,
):
    finger_skeletons = initialize_fingers(cps, effector_pos, n_finger_joints, height / 1000.)
    L, angle, ori = compute_skeleton(finger_skeletons, cps, effector_pos, n_finger_joints)
    fingers: List[Finger] = []

    for i, f in enumerate(finger_skeletons):
        n_joints = np.sum(~np.isnan(f).any(axis=1))
        units: List[Unit] = []

        for j in range(n_finger_joints - n_joints + 1, n_finger_joints):
            if j == n_finger_joints - n_joints + 1:
                u = Unit(10. - gap / 2., height, width, np.pi / 2, angle[i][j] / 2, 20.)
            elif j == n_finger_joints - 1:
                u = Unit(L[i][j] - gap / 2., height, width, angle[i][j - 1] / 2., angle[i][j], gap)
            else:
                u = Unit(L[i][j] - gap, height, width, angle[i][j - 1] / 2., angle[i][j] / 2., gap)
            units.append(u)

        fingers.append(Finger(units, orientation=ori[i]))

    return finger_skeletons, fingers


import pybullet as p
import pybullet_data
from GeometryUtils import GraspingObj

if __name__ == "__main__":
    # test
    unit_10 = Unit(9., 20., 20., np.pi / 2, 0.96206061, 20.)
    unit_11 = Unit(69.76291, 20., 20., 0.96206061, 1.3867712, 2.)
    unit_12 = Unit(30.33847, 20., 20., 1.3867712, 1.352039875, 2.)
    unit_13 = Unit(44.67522, 20., 20., 1.352039875, 0.41261234, 2.)
    finger_1 = Finger([unit_10, unit_11, unit_12, unit_13], -2.83609789)

    unit_20 = Unit(9., 20., 20., np.pi / 2, 0.93984609, 20.)
    unit_21 = Unit(70.57716, 20., 20., 0.93984609, 1.443382455, 2.)
    unit_22 = Unit(38.50861, 20., 20., 1.443382455, 1.00161288, 2.)
    unit_23 = Unit(14.48294, 20., 20., 1.00161288, 0.94417541, 2.)
    finger_2 = Finger([unit_20, unit_21, unit_22, unit_23], -0.88539239)

    unit_30 = Unit(9., 20., 20., np.pi / 2, 0.965114735, 20.)
    unit_31 = Unit(68.89493, 20., 20., 0.965114735, 1.24382005, 2.)
    unit_32 = Unit(79.6328, 20., 20., 1.24382005, 0.29255161, 2.)
    finger_3 = Finger([unit_30, unit_31, unit_32], 0.40175351)

    unit_40 = Unit(9., 20., 20., np.pi / 2, 0.91361506, 20.)
    unit_41 = Unit(37.52607, 20., 20., 0.91361506, 0.933024, 2.)
    unit_42 = Unit(22.03892, 20., 20., 0.933024, 1.0873191, 2.)
    finger_4 = Finger([unit_40, unit_41, unit_42], 1.3071038)
    gripper = FOAMGripper([finger_1, finger_2, finger_3, finger_4])

    stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.stl")
    test_obj = GraspingObj(friction=0.4)
    test_obj.read_from_stl(stl_file)
    # cps = ContactPoints(test_obj, [176, 306, 959, 2036])
    # end_effector_pos = np.asarray([test_obj.center_of_mass[0], test_obj.center_of_mass[1], test_obj.maxHeight + .02])
    # _, fingers = initialize_gripper(cps, end_effector_pos, 4)
    # gripper = FOAMGripper(fingers)

    # begin pybullet test
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")

    startPos = [0, 0., 0.07504227100332876]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box_id = p.loadURDF(os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle.urdf"), startPos, startOrientation,
                        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
    p.changeDynamics(box_id, -1, mass=50e-3)

    finger_id = []
    for f in gripper.fingers:
        startPos = [0, 0, test_obj.height + .02]
        startOrientation = p.getQuaternionFromEuler([0, 0, f.orientation])
        f_id = p.loadURDF(f.filename, startPos, startOrientation, useFixedBase=1,
                          flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)
        finger_id.append(f_id)

    p.setRealTimeSimulation(0)
    # p.performCollisionDetection()
    p.enableJointForceTorqueSensor(finger_id[0], 1, 1)

    for i in range(500):
        for f in finger_id:
            for j in range(2):
                limit = p.getJointInfo(f, j)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL, targetPosition=min(0.05 * i, .98 * limit))
        objPos, _ = p.getBasePositionAndOrientation(box_id)
        p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=50, cameraPitch=-30,
                                     cameraTargetPosition=[0, 0., .05 + objPos[2]])
        # print(p.getJointState(finger_id[0], 1))
        p.stepSimulation()
        time.sleep(1. / 240.)

    for i in range(500):
        for f in finger_id:
            for j in range(2):
                limit = p.getJointInfo(f, j)[9]
                p.setJointMotorControl2(f, j, p.POSITION_CONTROL, targetPosition=.98 * limit)
            p.resetBaseVelocity(f, [0, 0, .05])
        objPos, _ = p.getBasePositionAndOrientation(box_id)
        p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=50, cameraPitch=-30,
                                     cameraTargetPosition=[0, 0., .05 + objPos[2]])
        p.stepSimulation()
        # print(p.getJointState(finger_id[0], 1))
        time.sleep(1. / 240.)

    print(p.getBasePositionAndOrientation(box_id))
    for f in finger_id:
        cps = p.getContactPoints(box_id, f)
        for cp in cps:
            print(cp[9])  # normal force
        # print("\n")

    p.disconnect()
    # end pybullet test

    # gripper.assemble(bottom_thick=1.2)
    # gripper.seal_mask()
