import os
import numpy as np
import pickle
from ContactPoints import ContactPointsGA
from GeometryUtils import ContactPoints, GraspingObj
from GripperModel import initialize_gripper
from GripperModel import FOAMGripper
from GraspSim import gripper_sim
import pybullet as p

if __name__ == "__main__":
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)
    # stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.stl")
    # test_obj = GraspingObj(friction=0.5)
    # test_obj.read_from_stl(stl_file)
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .02])
    # test_obj.compute_connectivity_from(end_effector_pos)
    # save test_obj
    # with open(os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.pickle"),
    #           'wb') as f_test_obj:
    #     pickle.dump(test_obj, f_test_obj)
    test_obj_urdf = os.path.join(os.path.abspath('..'), "assets/ycb/013_apple.urdf")

    physicsClient = p.connect(p.DIRECT)

    design_cnt = 0
    for _ in range(1):
        ga = ContactPointsGA(test_obj, 4, end_effector_pos,
                             cross_prob=.8, mutation_factor=.6, maximizeFitness=True,
                             population_size=200, generations=50, verbose=False, adaptive=False)
        ga.run(n_workers=8)
        last_gen = list(ga.last_generation)
        cp_tested = []
        widths = np.arange(17.5, 27.5, 2.5)
        del ga

        for i in range(100):
            if last_gen[i][1] not in cp_tested and last_gen[i][0] > -2.:
                cur_gene = last_gen[i][1]
                cp_tested.append(cur_gene)
                cps = ContactPoints(test_obj, cur_gene)
                for w in widths:
                    _, fingers = initialize_gripper(cps, end_effector_pos, 4, width=w)
                    gripper = FOAMGripper(fingers)
                    success_cnt = 0
                    final_pos = None
                    for _ in range(10):
                        final_pos = gripper_sim(test_obj, test_obj_urdf, gripper)
                        if final_pos[-1] > test_obj.cog[-1] + .5 * (.05 * 500 / 240):
                            success_cnt += 1
                    if success_cnt > 5:
                        design_cnt += 1
                        print(cur_gene, f'Width: {w}', final_pos)
    print(design_cnt)
    p.disconnect()
