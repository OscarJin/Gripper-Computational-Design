from GeometryUtils import GraspingObj
from GeometryUtils import ContactPoints
import random
import numpy as np
from operator import attrgetter
import copy
from typing import List
from concurrent import futures
from scipy.spatial._qhull import QhullError
import matplotlib.pyplot as plt
from tqdm import tqdm
from GripperInitialization import initialize_fingers, compute_skeleton


class Chromosome(object):
    def __init__(self, genes):
        self.genes = genes
        self.fitness = 0.

    def __repr__(self):
        return repr((self.fitness, self.genes))


class ContactPointsGA(object):
    def __init__(
            self,
            graspingObj: GraspingObj,
            numContact,
            effector_pos,
            population_size=20,
            generations=100,
            cross_prob=.9,
            mutation_factor=.5,
            adaptive=False,
            maximizeFitness=True,
            verbose=False,
            random_state=None,
    ):
        self._graspObj = graspingObj
        self._numContact = numContact
        self._effector_pos = effector_pos
        self._lower_bound = 0
        self._upper_bound = self._graspObj.num_faces

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

        self._max_fitness = maximizeFitness

        self._verbose = verbose
        self._random = random.Random(random_state)

    def fitness(self, gene):
        cps = ContactPoints(obj=self._graspObj, fid=gene)
        if cps.F is None or cps.is_too_low:
            return -2.

        skeletons = initialize_fingers(cps, self._effector_pos, 4)
        L, _, _ = compute_skeleton(skeletons, cps, self._effector_pos, 4)
        L_avg = np.average(np.nansum(L, axis=1))
        L_avg /= (self._graspObj.height * 1000)  # normalize

        try:
            return 2 * cps.q_fcl + cps.q_vgp - cps.q_dcc + cps.ferrari_canny - .5 * L_avg
        except QhullError:
            return -2.

    def create_individual(self):
        """create an individual randomly"""
        gene = self._random.sample(range(self._lower_bound, self._upper_bound), self._numContact)
        return gene

    def create_new_individual(self, ind):
        """create an individual using selection, mutation and crossover"""
        base_i = self._random.sample([i for i in range(self._population_size) if i != ind], k=3)
        x1, x2, x3 = self._cur_generation[base_i[0]].genes, self._cur_generation[base_i[1]].genes, \
            self._cur_generation[base_i[2]].genes
        x1, x2, x3 = np.asarray(x1), np.asarray(x2), np.asarray(x3)
        mutant = x1 + self._mutation_factor * (x2 - x3)

        # crossover
        trial = np.where(np.random.rand(self._numContact) <= self._cross_prob,
                         mutant, np.asarray(self._cur_generation[ind].genes)).astype(int)
        j_rand = self._random.choice(range(self._numContact))
        trial[j_rand] = mutant[j_rand]  # ensure crossover
        trial = np.clip(trial, self._lower_bound, self._upper_bound - 1)
        trial = np.unique(trial)
        if trial.shape[0] < self._numContact:
            more_i = self._random.sample([i for i in range(self._upper_bound) if i not in trial],
                                         k=self._numContact - trial.shape[0])
            trial = np.concatenate((trial, more_i), axis=None)
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
        # self.calculate_population_fitness()
        self.rank_population()
        if self._verbose:
            print("Fitness: %f" % self.best_individual[0])

    def check_convergence(self, std):
        if std < .05 * self.best_individual[0]:
            return True
        else:
            return False

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

        for g in (tqdm(range(1, self._generations), desc="Searching optimal grasp configuration") if not self._verbose
                    else range(1, self._generations)):
            if self._adaptive:
                self._mutation_factor = (self._mutation_factor_0 *
                                         np.power(2, np.exp(1 - self._generations / (self._generations + 1 - g))))
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

            if self.check_convergence(std):
                self._generations_stop = g + 1
                break

    def visualisation(self):
        plt.figure(dpi=300)
        plt.plot(range(self._generations_stop), self._history_fitness, color='r', linewidth=1.5)
        plt.plot(range(self._generations_stop), self._history_fitness_avg, color='b', linewidth=1.5)
        r1 = list(map(lambda x: x[0] - x[1], zip(self._history_fitness_avg, self._history_fitness_std)))
        r2 = list(map(lambda x: x[0] + x[1], zip(self._history_fitness_avg, self._history_fitness_std)))
        plt.fill_between(range(self._generations_stop), r1, r2, color='b', alpha=0.2)
        # plt.xticks(range(self._generations), range(1, self._generations + 1))
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


import os
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from itertools import combinations
import pickle

if __name__ == "__main__":
    # test
    # stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.stl")
    # test_obj = GraspingObj(friction=0.5)
    # test_obj.read_from_stl(stl_file)
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)
    print(f'Faces: {test_obj.num_faces}')
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .02])
    test_obj.compute_connectivity_from(end_effector_pos)

    # tune
    # crs = np.arange(0.3, 0.9, 0.1)
    # fs = np.arange(0.4, 0.9, 0.1)
    # CR, F = np.meshgrid(crs, fs)
    # bests = np.zeros(CR.shape)
    # cnt = 0
    # for i in range(CR.shape[0]):
    #     for j in range(CR.shape[1]):
    #         ga = ContactPointsGA(test_obj, 4, cross_prob=CR[i][j], mutation_factor=F[i][j],
    #                              population_size=30, generations=50, verbose=False)
    #         ga.run(n_workers=8)
    #         cnt += 1
    #         print(cnt, ga.best_individual)
    #         bests[i][j] = ga.best_individual[0]
    #
    # figure = plt.figure(dpi=300)
    # ax = figure.add_axes(mplot3d.Axes3D(figure))
    # ax.plot_surface(CR, F, bests, cmap='cool')
    # ax.set_xlabel('Crossover Probability')
    # ax.set_ylabel('Mutation Factor')
    # ax.set_zlabel('Fitness')
    # plt.show()

    # single test
    t1 = time.time()
    ga = ContactPointsGA(test_obj, 3, end_effector_pos,
                         cross_prob=.8, mutation_factor=.6, maximizeFitness=True,
                         population_size=1000, generations=20, verbose=False, adaptive=False)
    ga.run(n_workers=8)
    t2 = time.time()
    print(f"Elapsed time: {t2 - t1} seconds")
    print(ga.best_individual)
    ga.visualisation()
    bestCP = ContactPoints(test_obj, ga.best_individual[1])
    bestCP.calc_force(verbose=True)
    bestCP.visualisation(vector_ratio=.2)

    for m in list(ga.last_generation):
        if m[0] != -np.inf:
            print(m[1])
