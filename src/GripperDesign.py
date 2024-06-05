import os
import numpy as np
import pickle
from GripperEA import SingleObjectGripperDE
from GeometryUtils import GraspingObj
from time import perf_counter

if __name__ == "__main__":
    ycb_model = '012_strawberry'
    with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}/{ycb_model}.pickle"),
              'rb') as f_test_obj:
        test_obj: GraspingObj = pickle.load(f_test_obj)
    test_obj_urdf = os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}.urdf")

    design_cnt = 0
    for _ in range(1):
        ga = SingleObjectGripperDE(test_obj, test_obj_urdf, 4, n_finger_joints=8,
                                   population_size=100, generations=50,
                                   cross_prob=.8, mutation_factor=.6, adaptive=False, maximize_fitness=True,
                                   verbose=True)
        t1 = perf_counter()
        ga.run(n_workers=5)
        t2 = perf_counter()
        print(t2 - t1)
        last_gen = list(ga.last_generation)
        for ind in last_gen:
            # if np.isclose(ind[0], 1):
            if ind[0] > 0.5:
                print(ind)
        ga.visualization()

