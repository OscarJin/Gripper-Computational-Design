import os
import numpy as np
import pickle
from ContactPoints import ContactPointsGA
from GeometryUtils import ContactPoints
from GripperModel import initialize_gripper
from GripperModel import FOAMGripper
from GraspSim import gripper_sim
import pybullet as p

if __name__ == "__main__":
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle/006_mustard_bottle.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .02])
    # test_obj.compute_connectivity_from(end_effector_pos)
    test_obj_urdf = os.path.join(os.path.abspath('..'), "assets/ycb/006_mustard_bottle.urdf")

    physicsClient = p.connect(p.DIRECT)

    success_cnt = 0
    for _ in range(1):
        ga = ContactPointsGA(test_obj, 4, end_effector_pos,
                             cross_prob=.8, mutation_factor=.6, maximizeFitness=True,
                             population_size=1000, generations=20, verbose=False, adaptive=False)
        ga.run(n_workers=8)
        last_gen = list(set(ga.last_generation))

        for i in range(100):
            if last_gen[i][0] > -np.inf:
                cur_gene = last_gen[i][1]
                cps = ContactPoints(test_obj, cur_gene)
                _, fingers = initialize_gripper(cps, end_effector_pos, 4, width=20.)
                gripper = FOAMGripper(fingers)
                final_pos = gripper_sim(test_obj, test_obj_urdf, gripper)
                if final_pos[-1] > .1:
                    success_cnt += 1
                    print(cur_gene, final_pos)
    print(success_cnt)
    p.disconnect()
