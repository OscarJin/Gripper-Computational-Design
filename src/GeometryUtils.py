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
from scipy.spatial import ConvexHull, Delaunay
from queue import PriorityQueue as PQ
from concurrent import futures
import warnings


class GraspingObj(object):
    def __init__(self, friction=.4):
        self._mesh = None
        self.faces = None
        self.normals = None     # outer
        self.vertices = None
        self._volume = None
        self.cog = None
        self.mu = friction
        self.minHeight = 0.
        self.maxHeight = 0.
        self.height = 0.
        # for compute connectivity
        self.dist = None
        self.parent = None

    def read_from_stl(self, filename):
        self._mesh = mesh.Mesh.from_file(filename)
        self.faces = self._mesh.vectors
        self.normals = self._mesh.normals

        zero_norm_i = []
        for i, n in enumerate(self.normals):
            if np.linalg.norm(n) < 1e-6:
                zero_norm_i.append(i)
        self.faces = np.delete(self.faces, zero_norm_i, axis=0)
        self.normals = np.delete(self.normals, zero_norm_i, axis=0)
        self.vertices = np.unique(self.faces.reshape([self.num_faces * 3, 3]), axis=0)

        self.cog = self.center_of_mass

        self.maxHeight = np.max(self.faces[:, :, 2])
        self.minHeight = np.min(self.faces[:, :, 2])
        self.height = self.maxHeight - self.minHeight

    @property
    def center_of_mass(self):
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

    @property
    def volume(self):
        volume, _, _ = self._mesh.get_mass_properties()
        return volume

    def visualisation(self):
        figure = plt.figure(dpi=300)
        ax = figure.add_axes(mplot3d.Axes3D(figure))
        ax.add_collection3d(Poly3DCollection(self.faces, alpha=.3, facecolors="lightgrey"))
        scale = self._mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        # plt.axis('off')
        plt.show()

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def furthest_dist(self):
        """get the furthest distance to cog"""
        dists = []
        for i in range(0, self.faces.shape[0]):
            for j in range(3):
                dists.append(np.linalg.norm(self.faces[i][j] - self.cog))

        return max(dists)

    def _segment_triangle_intersection(self, origin, direction, face_id) -> bool:
        A = self.faces[face_id][0]
        B = self.faces[face_id][1]
        C = self.faces[face_id][2]
        e_ = direction / np.linalg.norm(direction)

        segments = np.concatenate((origin, origin + direction), axis=0).reshape([2, 3])
        min_bound = np.min(segments, axis=0)
        max_bound = np.max(segments, axis=0)
        if not any(any(min_bound[i] <= P[i] <= max_bound[i] for i in range(3)) for P in [A, B, C]):
            return False

        AB = B - A
        AC = C - A
        n = np.cross(AB, AC)
        n_ = n / np.linalg.norm(n)
        if np.abs(np.dot(n_, e_)) < 1e-6:
            # parallel
            return False

        d = np.dot(n_, A - origin)
        d_end = np.dot(n_, A - (origin + direction))
        if np.abs(d) < 1e-6 or np.abs(d_end) < 1e-6:
            # one end on the surface excluded
            return False
        if d / d_end > 0:
            return False

        t = np.dot(n_, A - origin) / np.dot(n_, e_)
        P = origin + t * e_
        Pa = np.dot(np.cross(B - A, P - A), n_)
        Pb = np.dot(np.cross(C - B, P - B), n_)
        Pc = np.dot(np.cross(A - C, P - C), n_)
        if Pa < -1e-6 or Pb < -1e-6 or Pc < -1e-6:
            # Intersection point is outside the triangle
            return False

        return True

    def intersect_segment(self, origin, direction) -> bool:
        # for f in range(self.num_faces):
        #     intersects = self._segment_triangle_intersection(origin, direction, f)
        #     if intersects:
        #         return True
        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            for f in range(self.num_faces):
                # intersects = self._segment_triangle_intersection(origin, direction, f)
                intersects = executor.submit(self._segment_triangle_intersection, origin, direction, f)
                if intersects.result():
                    return True
        return False

    def compute_connectivity_from(self, origin):
        class VertexInfo(object):
            def __init__(self, id, dist):
                self.id = id
                self.dist = dist

            def __lt__(self, other):
                return self.dist > other.dist

        class EdgeInfo(object):
            def __init__(self, id, dist):
                self.id = id
                self.dist = dist

        dist = np.full(self.num_vertices, np.inf, dtype=float)
        parent = np.full(self.num_vertices, -2, dtype=int)
        edges: List[List[EdgeInfo]] = [[] for _ in range(self.num_vertices)]
        q = PQ()

        def calc_start_dist(i, v):
            direction = v - origin
            direction -= direction / np.linalg.norm(direction) * 1e-6
            if not self.intersect_segment(origin, direction):
                dist[i] = np.linalg.norm(v - origin)
                parent[i] = -1
                q.put(VertexInfo(i, dist[i]))
                print(f"{i}, ok")

        with futures.ThreadPoolExecutor(max_workers=8) as executor:
            _ = [executor.submit(calc_start_dist, i, v) for i, v in enumerate(self.vertices)]

        # for i, v in enumerate(self.vertices):
        #     direction = v - origin
        #     direction -= direction / np.linalg.norm(direction) * 1e-6
        #     if not self.intersect_segment(origin, direction):
        #         dist[i] = np.linalg.norm(v - origin)
        #         parent[i] = -1
        #         q.put(VertexInfo(i, dist[i]))
        #         print(f"{i}, ok")

        for i, f in enumerate(self.faces):
            for j in range(3):
                u = f[j]
                u_id = np.where(np.all(self.vertices == u, axis=1))[0][0]
                v = f[(j + 1) % 3]
                v_id = np.where(np.all(self.vertices == v, axis=1))[0][0]
                edges[u_id].append(EdgeInfo(v_id, np.linalg.norm(v - u)))

        # Dijkstra
        while not q.empty():
            now = q.get()
            for next in edges[now.id]:
                next_dist = dist[now.id] + next.dist
                if next_dist < dist[next.id]:
                    dist[next.id] = next_dist
                    parent[next.id] = now.id
                    q.put(VertexInfo(next.id, next_dist))

        self.dist = dist
        self.parent = parent


class ContactPoints(object):
    def __init__(self, obj: GraspingObj, fid: List[int]):
        self._obj = obj
        self._fid = fid
        self.nContact = len(fid)
        self.position = np.asarray([np.average(obj.faces[f], axis=0) for f in fid])
        self.normals = -np.asarray([obj.normals[f] / np.linalg.norm(obj.normals[f]) for f in fid])  # inner
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
        warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
        # lambdas = cp.Variable((3 * (self.nContact - 2), 1))
        lambdas = cp.Variable((self.G.shape[1], 1))
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
        # if problem.status not in ["infeasible", "unbounded"] | F.value is not None:
        if F.value is None:
            return None

        return np.reshape(F.value, (self.nContact, 3))

    def visualisation(self, vector_ratio=1.):
        figure = plt.figure(dpi=300)
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
                          self.F[i][0] * vector_ratio,
                          self.F[i][1] * vector_ratio,
                          self.F[i][2] * vector_ratio,
                          arrow_length_ratio=0.2)
        plt.show()

    @property
    def q_fcl(self):
        min_alpha = np.inf
        if self.F is not None:
            for i in range(self.nContact):
                fi = self.F[i]
                ni = self.normals[i]
                norm_f = np.linalg.norm(fi)
                norm_n = np.linalg.norm(ni)
                cos_a = fi.dot(ni) / (norm_f * norm_n)
                min_alpha = min(min_alpha, np.arccos(cos_a))
        else:
            min_alpha = 0.

        min_alpha /= (np.pi / 2)  # normalize
        return min_alpha

    @property
    def q_vgp(self):
        if self.nContact == 3:
            N = np.cross(self.position[1] - self.position[0], self.position[2] - self.position[0])
            return 0.5 * np.linalg.norm(N) / (self._obj.furthest_dist * self._obj.furthest_dist)

        hull = ConvexHull(self.position)
        q_vgp = hull.volume / self._obj.volume  # normalize
        return q_vgp

    @property
    def q_dcc(self):
        if self.nContact == 3:
            N = np.cross(self.position[1] - self.position[0], self.position[2] - self.position[0])
            if np.linalg.norm(N) < 1e-6:
                center = np.mean(self.position, axis=0)
                return np.linalg.norm(self._obj.cog - center)
            N /= np.linalg.norm(N)
            proj = (self._obj.cog - self.position[0]).dot(N)
            proj /= self._obj.furthest_dist     # normalize
            return np.linalg.norm(proj)

        T = Delaunay(self.position).simplices
        n = T.shape[0]
        W = 0.
        C = 0.

        for m in range(n):
            sp = self.position[T[m, :], :]
            w = ConvexHull(sp).volume
            C += w * np.mean(sp, axis=0)
            W += w

        center = C / W
        dist = center - self._obj.cog
        dist /= self._obj.furthest_dist     # normalize
        return np.linalg.norm(dist)

    @property
    def ferrari_canny(self):
        if self.F is None:
            return 0.

        F_norm = np.linalg.norm(self.F, ord=1, axis=1)
        # q = max(1 / F_norm)
        q = min(1 / F_norm)
        return q

    @property
    def is_too_low(self):
        h_threshold = self._obj.minHeight + self._obj.height * .1
        for p in self.position:
            if p[2] < h_threshold:
                return True

        return False


if __name__ == "__main__":
    # test
    stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.stl")
    test_obj = GraspingObj(friction=0.4)
    test_obj.read_from_stl(stl_file)
    print(test_obj.num_faces, test_obj.num_vertices,test_obj.volume, test_obj.cog, test_obj.furthest_dist)
    print(test_obj.minHeight, test_obj.maxHeight)
    # cps = ContactPoints(test_obj, [35, 56, 62])
    cps = ContactPoints(test_obj, np.arange(0, 2000, 600).tolist())
    cps.calc_force(verbose=True)
    # cps.visualisation(vector_ratio=.5)

    print(cps.is_too_low)

    # start = test_obj.center_of_mass
    # direction = (np.average(test_obj.faces[5], axis=0) - start) * 1.5
    # print(test_obj.intersect_segment(start, direction))

    end_effector_pos = np.asarray([test_obj.center_of_mass[0], test_obj.center_of_mass[1], test_obj.maxHeight + .1])
    test_obj.compute_connectivity_from(end_effector_pos)
