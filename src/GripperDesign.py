import os
import numpy as np
import pickle
from ContactPoints import ContactPointsGA
from GeometryUtils import ContactPoints, GraspingObj
from GripperModel import initialize_gripper
from GripperInitialization import initialize_fingers
from GripperModel import FOAMGripper
from GraspSim import multiple_gripper_sim
import pybullet as p

if __name__ == "__main__":
    with open(os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.pickle"),
              'rb') as f_test_obj:
        test_obj = pickle.load(f_test_obj)
    # stl_file = os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.stl")
    # test_obj = GraspingObj(friction=0.5)
    # test_obj.read_from_stl(stl_file)
    end_effector_pos = np.asarray([test_obj.cog[0], test_obj.cog[1], test_obj.maxHeight + .04])
    # test_obj.compute_connectivity_from(end_effector_pos)
    # save test_obj
    # with open(os.path.join(os.path.abspath('..'), "assets/ycb/013_apple/013_apple.pickle"),
    #           'wb') as f_test_obj:
    #     pickle.dump(test_obj, f_test_obj)
    test_obj_urdf = os.path.join(os.path.abspath('..'), "assets/ycb/013_apple.urdf")

    design_cnt = 0
    for _ in range(1):
        ga = ContactPointsGA(test_obj, test_obj_urdf, 4, end_effector_pos, n_finger_joints=8,
                             cross_prob=.8, mutation_factor=.6, maximizeFitness=True,
                             population_size=100, generations=50, verbose=True, adaptive=False)
        ga.run(n_workers=4)
        last_gen = list(ga.last_generation)
        del ga
        cp_tested = []
        widths = np.linspace(15., 25., 5)
        height_ratio = np.linspace(1.25, 2., 4)
        ww, rr = np.meshgrid(widths, height_ratio)

        for i in range(2):
            if last_gen[i][1] not in cp_tested and last_gen[i][0] > -1.:
                cur_gene = last_gen[i][1]
                cp_tested.append(cur_gene)
                cps = ContactPoints(test_obj, cur_gene)
                skeleton = initialize_fingers(cps, end_effector_pos, 8, root_length=.05)
                for i, w in np.ndenumerate(ww):
                    _, fingers = initialize_gripper(cps, end_effector_pos, 8, height_ratio=rr[i], width=w, finger_skeletons=skeleton)
                    gripper = FOAMGripper(fingers)
                    success_cnt = 0
                    final_pos = multiple_gripper_sim(test_obj, test_obj_urdf, [gripper] * 50, p.DIRECT)
                    for pos in final_pos:
                        if pos[-1] > test_obj.cog[-1] + .5 * (.05 * 500 / 240):
                            success_cnt += 1
                    if success_cnt > 0:
                        design_cnt += 1
                        print(cur_gene, f'Width: {w} Ratio:{rr[i]}', success_cnt)
                    gripper.clean()
    print(design_cnt)
