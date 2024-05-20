import os.path as osp
from GeometryUtils import GraspingObj, ContactPoints
import random
import numpy as np
from operator import attrgetter
import copy
from typing import List
from concurrent import futures
import matplotlib.pyplot as plt
from tqdm import tqdm
from GripperInitialization import initialize_fingers, compute_skeleton
from GripperModel import FOAMGripper, initialize_gripper
from GraspSim import multiple_gripper_sim
import pybullet as p
from abc import ABC, abstractmethod


class Chromosome(object):
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0.

    def __repr__(self):
        return repr((self.fitness, self.genes))


class DifferentialEvolution(ABC):
    def __init__(self,
                 lower_bound,
                 upper_bound,
                 population_size,
                 generations,
                 cross_prob,
                 mutation_factor,
                 adaptive=False,
                 maximize_fitness=True,
                 verbose=False,
                 random_state=None,
                 ):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        self._population_size = population_size
        self._generations = generations
        self._generations_stop = generations
        self._cross_prob = cross_prob
        self._mutation_factor_0 = mutation_factor
        self._mutation_factor = mutation_factor
        self._adaptive = adaptive

        self._cur_generation: List[Chromosome] = []
        self._history_fitness = []
        self._history_fitness_avg = []
        self._history_fitness_std = []

        self._max_fitness = maximize_fitness

        self._verbose = verbose
        self._random = random.Random(random_state)

    @abstractmethod
    def fitness(self, gene):
        pass

    @abstractmethod
    def create_individual(self):
        """create an individual randomly"""
        pass

    @abstractmethod
    def create_new_individual(self, ind):
        """create an individual using selection, mutation and crossover"""
        base_i = self._random.sample([i for i in range(self._population_size) if i != ind], k=3)
        x1, x2, x3 = self._cur_generation[base_i[0]].genes, self._cur_generation[base_i[1]].genes, \
            self._cur_generation[base_i[2]].genes
        x1, x2, x3 = np.asarray(x1), np.asarray(x2), np.asarray(x3)
        mutant = x1 + self._mutation_factor * (x2 - x3)

        # crossover
        trial = np.round(np.where(np.random.rand(len(mutant)) <= self._cross_prob,
                                  mutant, np.asarray(self._cur_generation[ind].genes))).astype(int)
        j_rand = self._random.choice(range(len(mutant)))
        trial[j_rand] = mutant[j_rand]  # ensure crossover
        trial = np.clip(trial, self._lower_bound, self._upper_bound)
        trial = trial.tolist()

        # selection
        trial_fitness = self.fitness(trial)
        if ((self._max_fitness and trial_fitness >= self._cur_generation[ind].fitness) |
                (not self._max_fitness and trial_fitness <= self._cur_generation[ind].fitness)):
            new_individual = Chromosome(trial)
            new_individual.fitness = trial_fitness
        else:
            new_individual = copy.deepcopy(self._cur_generation[ind])

        return new_individual

    def create_initial_generation(self):
        """create members of the first population randomly"""
        initial_generation = []
        for _ in range(self._population_size):
            gene = self.create_individual()
            individual = Chromosome(gene)
            initial_generation.append(individual)
        self._cur_generation = initial_generation

    def calculate_population_fitness(self, n_workers=None, parallel_type="processing"):
        if n_workers == 1:
            for i in self._cur_generation:
                i.fitness = self.fitness(i.genes)
        else:
            if "process" in parallel_type.lower():
                executor = futures.ProcessPoolExecutor(max_workers=n_workers)
            else:
                executor = futures.ThreadPoolExecutor(max_workers=n_workers)

            genes = [i.genes for i in self._cur_generation]

            with executor as pool:
                results = pool.map(self.fitness, genes)

            for individual, res in zip(self._cur_generation, results):
                individual.fitness = res

    def rank_population(self):
        """sort the population by fitness according to the order defined by maximizeFitness"""
        self._cur_generation.sort(
            key=attrgetter('fitness'), reverse=self._max_fitness
        )

    def create_new_population(self, n_workers=None, parallel_type="processing"):
        """create a new population using selection, mutation and crossover"""
        new_population = []

        if n_workers == 1:
            for i in range(self._population_size):
                new_population.append(self.create_new_individual(i))
        else:
            if "process" in parallel_type.lower():
                executor = futures.ProcessPoolExecutor(max_workers=n_workers)
            else:
                executor = futures.ThreadPoolExecutor(max_workers=n_workers)

            with executor as pool:
                new_population = list(pool.map(self.create_new_individual, range(self._population_size)))

        self._cur_generation = new_population

    def create_first_generation(self, n_workers=None, parallel_type="processing"):
        self.create_initial_generation()
        self.calculate_population_fitness(n_workers=n_workers, parallel_type=parallel_type)
        self.rank_population()

    def create_next_generation(self, n_workers=None, parallel_type="processing"):
        self.create_new_population(n_workers=n_workers, parallel_type=parallel_type)
        self.rank_population()
        if self._verbose:
            print(f"Best: {self.best_individual}")

    def check_convergence(self):
        if len(self._history_fitness) < 5:
            return False
        for fit in self._history_fitness[-2:-6:-1]:
            if not np.isclose(fit, self.best_individual[0]):
                return False
        return True

    def run(self, n_workers=None, parallel_type="processing"):
        """solve the GA"""
        self.create_first_generation(n_workers=n_workers, parallel_type=parallel_type)
        # for visualisation
        fits = [ind.fitness for ind in self._cur_generation]
        avg = np.mean(fits)
        std = np.std(fits)
        self._history_fitness.append(self.best_individual[0])
        self._history_fitness_avg.append(avg)
        self._history_fitness_std.append(std)
        if self._verbose:
            print(f'Generation: 0 Best: {self.best_individual}')

        for g in (tqdm(range(1, self._generations), desc="Searching optimal grasp configuration") if not self._verbose
                  else range(1, self._generations)):
            if self._adaptive:
                self._mutation_factor = (self._mutation_factor_0 *
                                         np.power(2, np.exp(1 - self._generations / (self._generations + 1 - g))))
            self._mutation_factor = self._mutation_factor_0 * (2 if self.check_convergence() else 1)
            if self._verbose:
                print(f'Generation: {g} Mutation: {self._mutation_factor}', end=" ")
            self.create_next_generation(n_workers=n_workers, parallel_type=parallel_type)

            # for visualisation
            fits = [ind.fitness for ind in self._cur_generation]
            avg = np.mean(fits)
            std = np.std(fits)
            self._history_fitness.append(self.best_individual[0])
            self._history_fitness_avg.append(avg)
            self._history_fitness_std.append(std)
            if self._verbose:
                print(f'avg: {avg}')

            # if self.check_convergence(avg, std):
            #     self._generations_stop = g + 1
            #     break

    def visualization(self):
        # plot
        plt.figure(dpi=300)
        plt.plot(range(self._generations_stop), self._history_fitness, color='r', linewidth=1.5)
        plt.plot(range(self._generations_stop), self._history_fitness_avg, color='b', linewidth=1.5)
        r1 = list(map(lambda x: x[0] - x[1], zip(self._history_fitness_avg, self._history_fitness_std)))
        r2 = list(map(lambda x: x[0] + x[1], zip(self._history_fitness_avg, self._history_fitness_std)))
        plt.fill_between(range(self._generations_stop), r1, r2, color='b', alpha=0.2)
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.show()

    @property
    def best_individual(self):
        best = self._cur_generation[0]
        return best.fitness, best.genes

    @property
    def last_generation(self):
        """return members of the last generation"""
        return ((member.fitness, member.genes) for member in self._cur_generation)


class SingleObjectGripperDE(DifferentialEvolution):
    """
        Differential evolution algorithm for designing a gripper to grasp a single object.

        A gene consists of n+3 integers, the first n are contact points,
        and the last three are width, height and end effector position in the form of indices.
    """
    def __init__(
            self,
            grasping_obj: GraspingObj,
            grasping_obj_urdf: str,
            numContact,
            n_finger_joints=8,
            population_size=50,
            generations=100,
            cross_prob=.9,
            mutation_factor=.5,
            adaptive=False,
            maximize_fitness=True,
            verbose=False,
            random_state=None,
    ):
        self._graspObj = grasping_obj
        self._graspObjUrdf = grasping_obj_urdf
        self._numContact = numContact
        self._n_finger_joints = n_finger_joints
        self.widths = np.linspace(12.5, 25., 6)
        _min_height_ratio = int(25e-2 / (self._graspObj.effector_pos[-1][-1] - self._graspObj.maxHeight)) / 10
        self.height_ratio = np.arange(_min_height_ratio, .7, .1)
        _lower_bound = [0] * (self._numContact + 3)
        _upper_bound = ([self._graspObj.num_middle_faces - 1] * self._numContact +
                        [len(self.widths) - 1, len(self.height_ratio) - 1, len(self._graspObj.effector_pos) - 1])

        super().__init__(_lower_bound, _upper_bound, population_size, generations, cross_prob, mutation_factor,
                         adaptive, maximize_fitness, verbose, random_state)

    def fitness(self, gene) -> float:
        cps = ContactPoints(obj=self._graspObj,
                            fid=np.take(self._graspObj.faces_mapping_clamp_height_and_radius, gene[0: self._numContact]).tolist())
        if cps.F is None:
            return 0

        # pybullet sim
        width = self.widths[gene[-3]]
        height_ratio = self.height_ratio[gene[-2]]
        end_effector_pos = self._graspObj.effector_pos[gene[-1]]
        end_height = end_effector_pos[-1] - self._graspObj.maxHeight
        if end_height * height_ratio < 15e-3:
            for i in range(gene[-2] + 1, self._upper_bound[-2] + 1):
                if end_height * self.height_ratio[i] > 15e-3:
                    height_ratio = self.height_ratio[i]
                    gene[-2] = i
                    break
        elif end_height * height_ratio > 25e-3:
            for i in range(gene[-2])[::-1]:
                if end_height * self.height_ratio[i] < 25e-3:
                    height_ratio = self.height_ratio[i]
                    gene[-2] = i
                    break

        skeletons = initialize_fingers(cps, end_effector_pos, self._n_finger_joints,
                                       expand_dist=end_height, root_length=.04, grasp_force=1e-3)

        L, _, ori = compute_skeleton(skeletons, cps, end_effector_pos, self._n_finger_joints)
        L_avg = np.average(np.nansum(L, axis=1))
        L_avg /= ((end_effector_pos[-1] - self._graspObj.minHeight) * 1000)  # normalize
        ori_sort = np.sort(ori, kind='heapsort')
        ori_sort = np.round(ori_sort * 8 / np.pi) * np.pi / 8
        min_ori_diff = np.inf
        for i, ori in enumerate(ori_sort):
            diff = (ori_sort[0] + 2 * np.pi) - ori_sort[-1] if i == len(ori_sort) - 1 else ori_sort[i + 1] - ori
            min_ori_diff = min(min_ori_diff, diff)
        if min_ori_diff < np.deg2rad(45):
            return 0
        if width > 20 * np.tan(min_ori_diff / 2) * 2 - 2:
            if gene[-3] == 0:
                return 0
            for i in range(gene[-3])[::-1]:
                if self.widths[i] < 20 * np.tan(min_ori_diff / 2) * 2 - 2:
                    width = self.widths[i]
                    gene[-3] = i
                    break

        max_height = 0.
        _, fingers = initialize_gripper(cps, end_effector_pos, self._n_finger_joints,
                                        expand_dist=end_height * 1000,
                                        height_ratio=height_ratio, width=width, finger_skeletons=skeletons)
        gripper = FOAMGripper(fingers)
        success_cnt = 0
        final_pos = multiple_gripper_sim(self._graspObj, self._graspObjUrdf, [gripper] * 20,
                                         end_height, max_deviation=.1, mode=p.DIRECT)
        for pos in final_pos:
            if .5 * (.05 * 500 / 240) < pos[-1] - self._graspObj.cog[-1] < 1.5 * (.05 * 500 / 240):
                success_cnt += 1
                if pos[-1] > max_height:
                    max_height = pos[-1]
        if success_cnt == 0:
            return 0

        # return 2 * success_cnt / 20 + max_height / (self._graspObj.cog[-1] + .05 * 500 / 240) - .5 * L_avg
        return success_cnt / 20

    def create_individual(self):
        """create an individual randomly"""
        contact_points = self._random.sample(range(0, self._graspObj.num_middle_faces), self._numContact)
        w = self._random.choice(range(len(self.widths)))
        h = self._random.choice(range(len(self.height_ratio)))
        pos = self._random.choice(range(len(self._graspObj.effector_pos)))
        gene = contact_points + [w, h, pos]
        return gene

    def create_new_individual(self, ind):
        """create an individual using selection, mutation and crossover"""
        base_i = self._random.sample([i for i in range(self._population_size) if i != ind], k=3)
        x1, x2, x3 = self._cur_generation[base_i[0]].genes, self._cur_generation[base_i[1]].genes, \
            self._cur_generation[base_i[2]].genes
        x1, x2, x3 = np.asarray(x1), np.asarray(x2), np.asarray(x3)
        mutant = x1 + self._mutation_factor * (x2 - x3)

        # crossover
        trial = np.round(np.where(np.random.rand(len(mutant)) <= self._cross_prob,
                                  mutant, np.asarray(self._cur_generation[ind].genes))).astype(int)
        j_rand = self._random.choice(range(len(mutant)))
        trial[j_rand] = mutant[j_rand]  # ensure crossover
        trial = np.clip(trial, self._lower_bound, self._upper_bound)
        trial_2 = copy.deepcopy(trial[self._numContact:])
        trial = trial[0: self._numContact]
        trial = np.unique(trial)
        if trial.shape[0] < self._numContact:
            more_i = self._random.sample([i for i in range(self._graspObj.num_middle_faces) if i not in trial],
                                         k=self._numContact - trial.shape[0])
            trial = np.concatenate((trial, more_i), axis=None)
        trial = np.concatenate((trial, trial_2), axis=None)
        trial = trial.tolist()

        # selection
        trial_fitness = self.fitness(trial)
        if ((self._max_fitness and trial_fitness >= self._cur_generation[ind].fitness) |
                (not self._max_fitness and trial_fitness <= self._cur_generation[ind].fitness)):
            new_individual = Chromosome(trial)
            new_individual.fitness = trial_fitness
        else:
            new_individual = copy.deepcopy(self._cur_generation[ind])

        return new_individual

    def visualization(self):
        # save numpy data
        training_data_file = osp.join(osp.abspath('..'), f"results/training_data_{self._graspObj.short_name}.npz")
        np.savez(training_data_file, self._generations_stop, self._history_fitness,
                 self._history_fitness_avg, self._history_fitness_std)
        # plot
        super().visualization()


class MultiObjectsGripperDE(DifferentialEvolution):
    """
        Differential evolution algorithm for designing a gripper to grasp m objects.

        A gene consists of m+n+3 integers. The first one is the index of the reference object,
        the next n are contact points, the next two are width and height,
        and last m are end effector heights for grasping each object.
    """
    def __init__(self,
                 grasping_objects: List[GraspingObj],
                 grasping_objects_urdf: List[str],
                 numContact,
                 n_finger_joints=8,
                 population_size=50,
                 generations=100,
                 cross_prob=.9,
                 mutation_factor=.5,
                 adaptive=False,
                 maximize_fitness=True,
                 verbose=False,
                 random_state=None,
                 ):
        self._grasping_objects = grasping_objects
        self._grasping_objects_urdf = grasping_objects_urdf
        self._numContact = numContact
        self._n_finger_joints = n_finger_joints
        self.widths = np.linspace(15., 25., 5)
        self.height_ratio = np.linspace(0.4, 0.7, 4)
        _lower_bound = [0] * (self.num_grasping_objects + self._numContact + 3)
        _upper_bound = []
        _end_effector_pos_upper_bound = [len(_obj.effector_pos) - 1 for _obj in self._grasping_objects]
        for _obj in self._grasping_objects:
            _obj_upper_bound = ([self.num_grasping_objects - 1] + [_obj.num_middle_faces - 1] * self._numContact +
                                [len(self.widths) - 1, len(self.height_ratio) - 1] + _end_effector_pos_upper_bound)
            _upper_bound.append(_obj_upper_bound)

        super().__init__(_lower_bound, _upper_bound, population_size, generations, cross_prob, mutation_factor,
                         adaptive, maximize_fitness, verbose, random_state)

    @property
    def num_grasping_objects(self):
        return len(self._grasping_objects)
