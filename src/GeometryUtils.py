import numpy as np
import math
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List
import cvxpy as cp


class GraspingObj:
    def __init__(self, friction=.4):
        self._mesh = None
        self.faces = None
        self.normals = None
        self._volume = None
        self._cog = None
        self.mu = friction

    def read_from_stl(self, filename):
        self._mesh = mesh.Mesh.from_file(filename)
        self.faces = self._mesh.vectors
        self.normals = self._mesh.normals
        self._cog = self.calc_cog()

    def calc_cog(self):
        volume = 0.
        center = np.asarray([0, 0, 0], dtype=float)
        for i in range(0, self.faces.shape[0]):
            a = self.faces[i][0]
            b = self.faces[i][1]
            c = self.faces[i][2]
            tet_volume = -a.dot(np.cross(b, c)) / 6.
            tet_center = (a + b + c) / 4.
            center += tet_center * tet_volume
            volume += tet_volume
        return center / volume

    def visualization(self):
        figure = plt.figure()
        ax = figure.add_axes(mplot3d.Axes3D(figure))
        ax.add_collection3d(Poly3DCollection(self.faces, alpha=.3, facecolors="lightgrey"))
        scale = self._mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        # plt.axis('off')
        plt.show()


class ContactPoints:
    def __init__(self, obj: GraspingObj, fid: List[int]):
        self._obj = obj
        self._fid = fid
        self.position = np.asarray([np.average(obj.faces[f], axis=0) for f in fid])
        self.normals = -np.asarray([obj.normals[f] / np.linalg.norm(obj.normals[f]) for f in fid])
        x1, y1, z1 = self.position[0]
        x2, y2, z2 = self.position[1]
        x3, y3, z3 = self.position[2]
        self.f = np.transpose(np.asarray([[0, 0, 1., 0, 0, 0]]))
        self.W = np.asarray([[1, 0, 0, 1, 0, 0, 1, 0, 0],
                             [0, 1, 0, 0, 1, 0, 0, 1, 0],
                             [0, 0, 1, 0, 0, 1, 0, 0, 1],
                             [0, -z1, y1, 0, -z2, y2, 0, -z3, y3],
                             [z1, 0, -x1, z2, 0, -x2, z3, 0, -x3],
                             [-y1, x1, 0, -y2, x2, 0, -y3, x3, 0]])
        self.Omega = self.W @ self.W.T
        self.N = np.asarray([[x2 - x1, x3 - x1, 0],
                             [y2 - y1, y3 - y1, 0],
                             [z2 - z1, z3 - z1, 0],
                             [x1 - x2, 0, x3 - x2],
                             [y1 - y2, 0, y3 - y2],
                             [z1 - z2, 0, z3 - z2],
                             [0, x1 - x3, x2 - x3],
                             [0, y1 - y3, y2 - y3],
                             [0, z1 - z3, z2 - z3]])
        self.G = self.N.T @ self.N
        self._FE = self.W.T @ np.linalg.inv(self.Omega) @ self.f
        self.K = 0.5 * self._FE.T @ self._FE
        self.eta = 1.0 + math.pow(self._obj.mu, 2)
        self.F = self.calc_force()

    def calc_force(self, verbose=False):
        lambdas = cp.Variable([3, 1])
        F = self._FE + self.N @ lambdas

        constraints = []
        for i in range(3):
            Fi = F[i * 3: (i + 1) * 3]
            gi = cp.norm(Fi) - cp.sqrt(self.eta) * (Fi.T @ self.normals[i])
            hi = -Fi.T @ self.normals[i]
            constraints += [gi <= 0]
            constraints += [hi <= 0]

        objective = cp.Minimize((1 / 2) * cp.quad_form(lambdas, self.G) + self.K)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)
        if verbose:
            print("Status: %s" % problem.status)
            print("Optimal Value: %s" % problem.value)
            print("Force:\n %s" % F.value)
        if problem.status not in ["infeasible", "unbounded"]:
            return F.value
        else:
            return None

    def visualisation(self, vector_ratio=1.):
        figure = plt.figure()
        ax = figure.add_axes(mplot3d.Axes3D(figure))
        ax.add_collection3d(Poly3DCollection(self._obj.faces, alpha=.3, facecolors="lightgrey"))
        scale = self._obj._mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        x = [p[0] for p in self.position]
        y = [p[1] for p in self.position]
        z = [p[2] for p in self.position]
        ax.scatter3D(x[0], y[0], z[0], marker="v", c='r')
        ax.add_collection3d(Poly3DCollection(self._obj.faces[self._fid], facecolors="r"))
        if self.F is not None:
            for i in range(3):
                ax.quiver(x[i], y[i], z[i],
                          self.F[i * 3] * vector_ratio,
                          self.F[i * 3 + 1] * vector_ratio,
                          self.F[i * 3 + 2] * vector_ratio,
                          arrow_length_ratio=0.1)
        plt.show()


if __name__ == "__main__":
    # test
    stl_file = 'E:/SGLab/Dissertation/Gripper-Computational-Design/assets/Cube.stl'
    test_obj = GraspingObj(friction=0.5)
    test_obj.read_from_stl(stl_file)
    cps = ContactPoints(test_obj, [31, 66, 99])
    cps.calc_force(verbose=True)
    cps.visualisation(vector_ratio=10)
