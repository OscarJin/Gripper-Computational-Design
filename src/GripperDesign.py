import os
import numpy as np
import pickle
from GripperEA import SingleObjectGripperDE
from GeometryUtils import GraspingObj

if __name__ == "__main__":
    ycb_model = '011_banana'
    with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}/{ycb_model}.pickle"),
              'rb') as f_test_obj:
        test_obj: GraspingObj = pickle.load(f_test_obj)
    test_obj_urdf = os.path.join(os.path.abspath('..'), f"assets/ycb/{ycb_model}.urdf")

    design_cnt = 0
    for _ in range(1):
        ga = SingleObjectGripperDE(test_obj, test_obj_urdf, 4, n_finger_joints=8,
                                   population_size=100, generations=100,
                                   cross_prob=.8, mutation_factor=.6, adaptive=False, maximize_fitness=True,
                                   verbose=True)
        ga.run(n_workers=5)
        last_gen = list(ga.last_generation)
        for ind in last_gen:
            if np.isclose(ind[0], 1):
                print(ind[1])
        ga.visualization()

