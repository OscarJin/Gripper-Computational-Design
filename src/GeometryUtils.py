import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List


class ContactPoint:
    def __init__(self, obj, fid):
        self.fid = fid
        self.position = np.average(obj._faces[fid], axis=0)
        self.normal = obj._normals[fid]
        self.normal /= np.linalg.norm(self.normal)


class ContactCone:
    def __init__(self, fid, position, normal):
        self.fid = fid
        self.position = position
        self.normal = normal


class GraspingObj:
    def __init__(self, friction=.4, cone_num=4):
        self._mesh = None
        self._faces = None
        self._normals = None
        self._volume = None
        self._cog = None
        self._cps = []
        self._contact_cones = []
        self._friction = friction
        self._cone_num = cone_num

    def read_from_stl(self, filename):
        self._mesh = mesh.Mesh.from_file(filename)
        self._faces = self._mesh.vectors
        self._normals = self._mesh.normals
        self._cog = self.calc_cog()

    def visualization(self):
        figure = plt.figure()
        ax = figure.add_axes(mplot3d.Axes3D(figure))
        ax.add_collection3d(Poly3DCollection(self._faces, alpha=.3, facecolors="lightgrey"))
        scale = self._mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        x = [cp.position[0] for cp in self._cps]
        y = [cp.position[1] for cp in self._cps]
        z = [cp.position[2] for cp in self._cps]
        print(x, y, z)
        ax.scatter3D(x, y, z, marker="v", c='r')
        ax.add_collection3d(Poly3DCollection(self._faces[[cp.fid for cp in self._cps]], facecolors="r"))
        # plt.axis('off')
        plt.show()

    def calc_cog(self):
        volume = 0.
        center = np.asarray([0, 0, 0], dtype=float)
        for i in range(0, self._faces.shape[0]):
            a = self._faces[i][0]
            b = self._faces[i][1]
            c = self._faces[i][2]
            tet_volume = -a.dot(np.cross(b, c)) / 6.
            tet_center = (a + b + c) / 4.
            center += tet_center * tet_volume
            volume += tet_volume
        return center / volume

    def _generate_contact_cone(self, contactPoint: ContactPoint):
        stepSize = 2 * np.pi / self._cone_num
        Y = np.asarray([0, 1, 0])
        B = np.cross(contactPoint.normal, Y)
        B /= np.linalg.norm(B)
        T = np.cross(B, contactPoint.normal)
        T /= np.linalg.norm(T)
        coeff = max(contactPoint.normal.dot(Y), 1e-6)
        B *= self._friction * coeff
        T *= self._friction * coeff
        contact_cone = []
        for i in range(self._cone_num):
            curStep = i * stepSize
            normal = contactPoint.normal + B * np.cos(curStep) + T * np.sin(curStep)
            cone = ContactCone(contactPoint.fid, contactPoint.position, normal)
            contact_cone.append(cone)

        return contact_cone

    def generate_contact_cones(self, contactPoints: List[ContactPoint]):
        self._cps = contactPoints
        for cp in contactPoints:
            for cone in self._generate_contact_cone(cp):
                self._contact_cones.append(cone)
