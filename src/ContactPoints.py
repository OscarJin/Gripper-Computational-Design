from GeometryUtils import GraspingObj
from GeometryUtils import ContactPoints
import random
import numpy as np
from operator import attrgetter
import copy
from typing import List
from concurrent import futures
from scipy.spatial._qhull import QhullError


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
            population_size=20,
            generations=100,
            cross_prob=.9,
            mutation_factor=.5,
            elitism=True,
            maximizeFitness=True,
            verbose=False,
            random_state=None,
    ):
        self._graspObj = graspingObj
        self._numContact = numContact
        self._lower_bound = 0
        self._upper_bound = self._graspObj.num_faces

        self._population_size = population_size
        self._generations = generations
        self._cross_prob = cross_prob
        self._mutation_factor = mutation_factor
        self._elitism = elitism

        self._cur_generation: List[Chromosome] = []

        self._max_fitness = maximizeFitness

        self._verbose = verbose
        self._random = random.Random(random_state)

    def fitness(self, gene):
        try:
            cps = ContactPoints(obj=self._graspObj, fid=gene)
        except np.linalg.LinAlgError:
            return 0
        else:
            if cps.F is None:
                return 0.
            else:
                try:
                    return 2 * cps.q_fcl + cps.q_vgp - cps.q_dcc
                except QhullError:
                    return 0.

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
        elite = copy.deepcopy(self._cur_generation[0])

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

        if self._elitism:
            new_population[0] = elite

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

    def run(self, n_workers=None, parallel_type="processing"):
        """solve the GA"""
        self.create_first_generation(n_workers=n_workers, parallel_type=parallel_type)

        for _ in range(1, self._generations):
            self.create_next_generation(n_workers=n_workers, parallel_type=parallel_type)

    @property
    def best_individual(self):
        best = self._cur_generation[0]
        return best.fitness, best.genes

    def last_generation(self):
        """return members of the last generation"""
        return ((member.fitness, member.genes) for member in self._cur_generation)


import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
from itertools import combinations

if __name__ == "__main__":
    # test
    stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/google_16k/nontextured.stl")
    # stl_file = os.path.join(os.path.abspath('..'), "assets/Cube.stl")
    test_obj = GraspingObj(friction=0.4)
    test_obj.read_from_stl(stl_file)
    print(f'Faces: {test_obj.num_faces}')

    # tune
    # crs = np.arange(0.3, 0.9, 0.1)
    # fs = np.arange(0.4, 0.9, 0.1)
    # CR, F = np.meshgrid(crs, fs)
    # bests = np.zeros(CR.shape)
    # for i in range(CR.shape[0]):
    #     for j in range(CR.shape[1]):
    #         ga = ContactPointsGA(test_obj, 4, cross_prob=CR[i][j], mutation_factor=F[i][j],
    #                              population_size=30, generations=20, verbose=False)
    #         ga.run()
    #         print(ga.best_individual)
    #         bests[i][j] = ga.best_individual[0]
    #
    # figure = plt.figure()
    # ax = figure.add_axes(mplot3d.Axes3D(figure))
    # ax.plot_surface(CR, F, bests, cmap='cool')
    # ax.set_xlabel('Crossover Probability')
    # ax.set_ylabel('Mutation Factor')
    # ax.set_zlabel('Fitness')
    # plt.show()

    # single test
    t1 = time.time()
    ga = ContactPointsGA(test_obj, 4, population_size=30, generations=20, verbose=True)
    ga.run(n_workers=8)
    t2 = time.time()
    print(f"Elapsed time: {t2 - t1} seconds")
    print(ga.best_individual)
    bestCP = ContactPoints(test_obj, ga.best_individual[1])
    bestCP.calc_force(verbose=True)
    bestCP.visualisation(vector_ratio=.5)

    """enumerate Cube.stl 1324.5218"""
    # maxFitness = 0.
    # for i in range(test_obj.num_faces - 2):
    #     for j in range(i + 1, test_obj.num_faces - 1):
    #         for k in range(j + 1, test_obj.num_faces):
    #             cur_fitness = ga.fitness([i, j, k])
    #             if cur_fitness > maxFitness:
    #                 print(f'Fitness: {cur_fitness}')
    #                 maxFitness = cur_fitness
    #
    # print(f'Optimal Fitness: {maxFitness}')
