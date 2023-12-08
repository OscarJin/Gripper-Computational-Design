import os
import numpy as np
from scipy.linalg import null_space
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
        self.nContact = len(fid)
        self.position = np.asarray([np.average(obj.faces[f], axis=0) for f in fid])
        self.normals = -np.asarray([obj.normals[f] / np.linalg.norm(obj.normals[f]) for f in fid])
        self.f = np.transpose(np.asarray([[0, 0, 1., 0, 0, 0]]))
        self.W = self._create_grasp_matrix()
        self.Omega = self.W @ self.W.T
        self.N = null_space(self.W)
        self.G = self.N.T @ self.N
        self._FE = self.W.T @ np.linalg.inv(self.Omega) @ self.f
        self.K = 0.5 * self._FE.T @ self._FE
        self.eta = 1.0 + math.pow(self._obj.mu, 2)
        self.F = self.calc_force()

    def _create_grasp_matrix(self):
        W = np.zeros([6, 3 * self.nContact])
        for i in range(self.nContact):
            xi, yi, zi = self.position[i]
            W[:, 3 * i: 3 * (i + 1)] = np.asarray([[1, 0, 0],
                                                   [0, 1, 0],
                                                   [0, 0, 1],
                                                   [0, -zi, yi],
                                                   [zi, 0, -xi],
                                                   [-yi, xi, 0]])
        return W

    def calc_force(self, verbose=False):
        lambdas = cp.Variable([3 * (self.nContact - 2), 1])
        F = self._FE + self.N @ lambdas

        constraints = []
        for i in range(self.nContact):
            Fi = F[i * 3: (i + 1) * 3]
            gi = cp.norm(Fi) - cp.sqrt(self.eta) * (Fi.T @ self.normals[i])
            hi = -Fi.T @ self.normals[i]
            constraints += [gi <= 0]
            constraints += [hi <= 0]

        objective = cp.Minimize((1 / 2) * cp.quad_form(lambdas, self.G) + self.K)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)
        if verbose:
            print("Status: %s" % problem.status)
            print("Optimal Value: %s" % problem.value)
            print("Force:\n %s" % F.value)
            print("Solve time:", problem.solver_stats.solve_time)
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
        ax.scatter3D(x, y, z, marker="v", c='r')
        ax.add_collection3d(Poly3DCollection(self._obj.faces[self._fid], facecolors="r"))
        if self.F is not None:
            for i in range(self.nContact):
                ax.quiver(x[i], y[i], z[i],
                          self.F[i * 3] * vector_ratio,
                          self.F[i * 3 + 1] * vector_ratio,
                          self.F[i * 3 + 2] * vector_ratio,
                          arrow_length_ratio=0.2)
        plt.show()


if __name__ == "__main__":
    # test
    stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/google_16k/nontextured.stl")
    # stl_file = 'E:/SGLab/Dissertation/Gripper-Computational-Design/assets/StanfordBunny.stl'
    test_obj = GraspingObj(friction=0.4)
    test_obj.read_from_stl(stl_file)
    # cps = ContactPoints(test_obj, [31, 66, 99])
    cps = ContactPoints(test_obj, np.arange(0, 5000, 200).tolist())
    cps.calc_force(verbose=True)
    cps.visualisation(vector_ratio=.5)
