import os
import numpy as np
import stl
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
from tqdm import tqdm
import warnings
import pickle
from re import split


class GraspingObj(object):
    def __init__(self, friction=.4):
        self.short_name = None
        self._mesh = None
        self.faces = None
        self.normals = None     # outer
        self.vertices = None
        self.vertex2face = None
        self._volume = None
        self.cog = None
        self.mu = friction
        # bounding box
        self.minHeight = 0.
        self.maxHeight = 0.
        self.height = 0.
        self.x_min = 0.
        self.x_max = 0.
        self.x_span = 0.
        self.y_min = 0.
        self.y_max = 0.
        self.y_span = 0.
        # clamp height remap
        self.faces_mapping_clamp_height_and_radius: List[int] = []
        # for compute connectivity
        self.dist = {}
        self.parent = {}
        self.effector_pos = []

    def read_from_stl(self, filename):
        self.short_name = split('[/.]', filename)[-2]
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

        # construct vertex to face
        self.vertex2face = np.empty([self.num_faces, 3], dtype=int)
        for i, f in enumerate(self.faces):
            for j in range(3):
                self.vertex2face[i][j] = self._get_point_ind(f[j])

        self.cog = self._calc_center_of_mass()

        self.maxHeight = np.max(self.faces[:, :, 2])
        self.minHeight = np.min(self.faces[:, :, 2])
        self.height = self.maxHeight - self.minHeight
        self.x_max = np.max(self.faces[:, :, 0])
        self.x_min = np.min(self.faces[:, :, 0])
        self.x_span = self.x_max - self.x_min
        self.y_max = np.max(self.faces[:, :, 1])
        self.y_min = np.min(self.faces[:, :, 1])
        self.y_span = self.y_max - self.y_min

    def _calc_center_of_mass(self):
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

    def visualization(self):
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

    def clamp_height_and_radius(self, lower_bound=.2, upper_bound=.9, radius=np.inf):
        self.faces_mapping_clamp_height_and_radius.clear()
        centers = np.asarray([np.average(f, axis=0) for f in self.faces])
        for i, c in enumerate(centers):
            if (lower_bound * self.height < c[-1] - self.minHeight < upper_bound * self.height
                    and np.linalg.norm(c[:-1] - self.cog[:-1]) < radius):
                self.faces_mapping_clamp_height_and_radius.append(i)

    @property
    def num_middle_faces(self):
        return len(self.faces_mapping_clamp_height_and_radius)

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
        with futures.ThreadPoolExecutor(max_workers=32) as executor:
            for f in range(self.num_faces):
                # intersects = self._segment_triangle_intersection(origin, direction, f)
                intersects = executor.submit(self._segment_triangle_intersection, origin, direction, f)
                if intersects.result():
                    return True
        return False

    def _get_point_ind(self, p) -> int:
        return int(np.where(np.all(np.abs(self.vertices - p) < 1e-5, axis=1))[0][0])

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

        if any(np.array_equal(origin, pos) for pos in self.effector_pos):
            return
        self.effector_pos.append(origin.copy())
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

        with futures.ThreadPoolExecutor(max_workers=32) as executor:
            tasks = [executor.submit(calc_start_dist, i, v) for i, v in enumerate(self.vertices)]

            for _ in tqdm(futures.as_completed(tasks), total=len(tasks),
                          desc="Computing connectivity from end effector height {:.2f}".format(origin[-1] - self.maxHeight)):
                pass

        for i, f in enumerate(self.faces):
            for j in range(3):
                u = f[j]
                u_id = self.vertex2face[i][j]
                v = f[(j + 1) % 3]
                v_id = self.vertex2face[i][(j + 1) % 3]
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

        self.dist[tuple(origin)] = dist
        self.parent[tuple(origin)] = parent

    def compute_closest_point(self, fid: int) -> int:
        center = np.average(self.faces[fid], axis=0)
        dists = np.asarray([np.linalg.norm(center - self.faces[fid][i]) for i in range(3)])
        min_i = np.argmax(dists)
        v_id = self.vertex2face[fid][min_i]
        return int(v_id)

    def compute_vertex_normal(self, vid: int):
        face_ind = np.where(self.vertex2face == vid)[0]
        normal = self.normals[face_ind]
        v_n = np.average(normal, axis=0)
        v_n /= np.linalg.norm(v_n)
        return v_n

    def preprocess(self,
                   stl_path: str,
                   data_path: str,
                   height_lower_bound: float = .2,
                   height_upper_bound: float = .9,
                   radius: float = None,
                   end_effector_height_step: float = .01,
                   end_effector_max_height: float = 1,
                   ):
        self.read_from_stl(stl_path)
        xy_ratio = self.x_span / self.y_span if self.x_span > self.y_span else self.y_span / self.x_span
        r = ((self.x_span + self.y_span) / 4 if xy_ratio > 1.5 else np.inf) if radius is None else radius
        self.clamp_height_and_radius(height_lower_bound, height_upper_bound, r)
        print(f'Regions of interest: {len(self.faces_mapping_clamp_height_and_radius)}')
        _end_effector_pos = np.asarray([self.cog[0], self.cog[1], self.maxHeight + .04])
        while _end_effector_pos[-1] < self.maxHeight + end_effector_max_height * self.height:
            self.compute_connectivity_from(_end_effector_pos)
            _end_effector_pos[-1] += end_effector_height_step
        with open(data_path, 'wb') as _obj_data:
            pickle.dump(self, _obj_data)


class ContactPoints(object):
    def __init__(self, obj: GraspingObj, fid: List[int]):
        self._obj = obj
        self._fid = fid
        self.nContact = len(fid)
        self.position = np.asarray([np.average(obj.faces[f], axis=0) for f in fid])
        self.normals = -np.asarray([obj.normals[f] / np.linalg.norm(obj.normals[f]) for f in fid])  # inner

        self._full_fid = []
        for f in self._fid:
            for j in range(3):
                self._full_fid += list(np.where(self._obj.vertex2face == self._obj.vertex2face[f][j])[0])
        self._full_fid = list(set(self._full_fid))
        self._position_full = np.asarray([np.average(obj.faces[f], axis=0) for f in self._full_fid])
        self._normals_full = -np.asarray([obj.normals[f] / np.linalg.norm(obj.normals[f]) for f in self._full_fid])  # inner
        self.f = np.transpose(np.asarray([[0, 0, 9.8, 0, 0, 0]]))
        self.W = self._create_grasp_matrix()
        self.Omega = self.W @ self.W.T
        self.N = null_space(self.W)
        self.G = self.N.T @ self.N
        self._FE = self.W.T @ np.linalg.inv(self.Omega) @ self.f
        self.K = 0.5 * self._FE.T @ self._FE
        self.eta = 1.0 + math.pow(self._obj.mu, 2)
        self.F = self.calc_force()

    def _create_grasp_matrix(self):
        W = np.zeros([6, 3 * len(self._full_fid)])
        # for i, f in enumerate(self._fid):
        #     for _j in range(3):
        #         xi, yi, zi = self._obj.faces[f][_j]
        #         W[:, 3 * (i + _j): 3 * (i + _j + 1)] = np.asarray([[1, 0, 0],
        #                                                [0, 1, 0],
        #                                                [0, 0, 1],
        #                                                [0, -zi, yi],
        #                                                [zi, 0, -xi],
        #                                                [-yi, xi, 0]])
        for i in range(len(self._full_fid)):
            xi, yi, zi = self._position_full[i]
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
        for i in range(len(self._full_fid)):
            Fi = F[i * 3: (i + 1) * 3]
            # _normal = self._obj.compute_vertex_normal(self._obj.vertex2face[i // 3][i % 3])
            _normal = self._normals_full[i]
            mu_i = self._obj.mu if np.dot(_normal, self.f[:3]) > 0 else 0.
            eta_i = 1.0 + math.pow(mu_i, 2)
            gi = cp.norm(Fi) - cp.sqrt(eta_i) * (Fi.T @ _normal)
            hi = -Fi.T @ _normal
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

        return np.reshape(F.value, (len(self._full_fid), 3))

    def visualization(self, vector_ratio=1.):
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        figure = plt.figure(dpi=300)
        ax = figure.add_axes(mplot3d.Axes3D(figure), aspect='equal')
        ax.grid(None)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.xaxis.set_pane_color((1, 1, 1, 1))
        ax.yaxis.set_pane_color((1, 1, 1, 1))
        ax.zaxis.set_pane_color((1, 1, 1, 1))
        ax.tick_params(labelsize=8)
        ax.add_collection3d(Poly3DCollection(self._obj.faces, alpha=.3, facecolors="lightgrey"))
        scale = self._obj._mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        x = [p[0] for p in self._position_full]
        y = [p[1] for p in self._position_full]
        z = [p[2] for p in self._position_full]
        ax.scatter3D(x, y, z, marker="v", c='r')
        ax.add_collection3d(Poly3DCollection(self._obj.faces[self._fid], facecolors="r"))
        if self.F is not None:
            for i in range(len(self._full_fid)):
                # Fx, Fy, Fz = list(map(lambda j: np.linalg.norm(self.F[i * 3: (i + 1) * 3][j]), range(3)))
                Fx, Fy, Fz = list(map(lambda j: self.F[i][j], range(3)))
                ax.quiver(x[i], y[i], z[i],
                          Fx * vector_ratio,
                          Fy * vector_ratio,
                          Fz * vector_ratio,
                          arrow_length_ratio=0.2, linewidth=.8)
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
                cos_a = np.clip(cos_a, -1, 1)
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
    def q_dcc(self) -> float:
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
        h_threshold = self._obj.minHeight + self._obj.height * .2
        for p in self.position:
            if p[2] < h_threshold:
                return True
        return False

    @property
    def is_too_high(self):
        h_threshold = self._obj.minHeight + self._obj.height * .9
        for p in self.position:
            if p[2] > h_threshold:
                return True
        return False

    @property
    def obj(self) -> GraspingObj:
        return self._obj

    @property
    def fid(self) -> List[int]:
        return self._fid

from time import perf_counter

if __name__ == "__main__":
    """
        prepare the grasping object here
    """
    ycb_model = '001_tape'
    stl_file = os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}/{ycb_model}.stl")
    data_file = os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}/{ycb_model}.pickle")
    test_obj = GraspingObj(friction=0.5)
    t1 = perf_counter()
    test_obj.preprocess(stl_path=stl_file, data_path=data_file, end_effector_max_height=4, radius=np.inf)
    t2 = perf_counter()
    print(t2 - t1)

    # ycb_model = '011_banana'
    # with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}/{ycb_model}.pickle"),
    #           'rb') as f_test_obj:
    #     test_obj: GraspingObj = pickle.load(f_test_obj)
    # cps = ContactPoints(test_obj, np.take(test_obj.faces_mapping_clamp_height_and_radius, [410, 542, 1205, 1861]).tolist())
    # cps.calc_force(verbose=True)
    # cps.visualization(vector_ratio=.04)
