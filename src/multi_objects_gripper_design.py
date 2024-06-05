import os
import pickle
from GripperEA import MultiObjectsGripperDE
from GeometryUtils import GraspingObj

if __name__ == "__main__":
    ycb_models = ['011_banana', '013_apple']
    test_objects = []
    test_objects_urdf = []
    for _model in ycb_models:
        with open(os.path.join(os.path.abspath('..'), f"assets/ycb/{_model}/{_model}.pickle"),
                  'rb') as f_obj:
            _model_obj: GraspingObj = pickle.load(f_obj)
            test_objects.append(_model_obj)
            test_objects_urdf.append(os.path.join(os.path.abspath('..'), f"assets/ycb/{_model}.urdf"))

    ga = MultiObjectsGripperDE(test_objects, test_objects_urdf, 4, n_finger_joints=8,
                               population_size=100, generations=50,
                               cross_prob=.8, mutation_factor=.6, adaptive=False, maximize_fitness=True,
                               verbose=True)
    ga.run(n_workers=4)

    last_gen = list(ga.last_generation)
    for ind in last_gen:
        if ind[0] > 0:
            print(ind)
    ga.visualization()
