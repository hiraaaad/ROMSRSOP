from unittest import TestCase
# import matplotlib.pyplot as plt
import numpy as np

class TestParallel_Controller():

    def __init__(self, Parallel_Controller):
        self.alg = Parallel_Controller

    def test_rwheel_lr(self):
        param_greedy = 50
        p_rank_greedy = np.divide(p_rank ** param_greedy, np.sum(p_rank ** param_greedy))  #
        x = np.random.choice(r, int(1e5), p=p_rank_greedy[::-1])
        plt.hist(x)

    def test_selection_rga(self, obj_neighbor):
        feasible = np.where((obj_neighbor[:, 1] == 0) & (obj_neighbor[:, 2] == 0))[-1]

    def validate(self):
        # this function tests the validation of results
        # check utility
        req = 0
        temp = dict.fromkeys(self.alg.machine_idx, [])

        for idm in np.arange(self.alg.problem.number_machine):

            if req == 0:
                initial_cut_tuple = self.alg.machine[idm].solution[req].solution_path[0]
                utility_start = np.divide(initial_cut_tuple[0].cost_reclaim, initial_cut_tuple[0].cut_tonnage)
            else:
                initial_cut_tuple = self.alg.machine[idm].solution[req].solution_path[0]
                cut_start = self.alg.machine[idm].solution[req-1].solution_path[0][0]
                cut_start_direction = self.alg.machine[idm].solution[req-1].solution_path[0][1]
                cost_moving = np.float(
                    self.alg.problem.D_cost[cut_start.index, initial_cut_tuple[0].index, cut_start_direction, initial_cut_tuple[1]])
                cost_reclaim = np.float(initial_cut_tuple[0].cost_reclaim)
                cost = cost_reclaim + cost_moving
                utility_start = np.round(np.divide(cost, initial_cut_tuple[0].cut_tonnage), self.alg.problem.precision)

            temp[idm].append(np.round(utility_start,self.alg.problem.precision))

            for i, e in enumerate(self.alg.machine[idm].solution[req].solution_path):
                if i < len(self.alg.machine[idm].solution[req]) - 1:
                    cut_start = self.alg.machine[idm].solution[req].solution_path[i][0]
                    cut_start_direction = self.alg.machine[idm].solution[req].solution_path[i][1]
                    cut_end = self.alg.machine[idm].solution[req].solution_path[i + 1][0]
                    cut_end_direction = self.alg.machine[idm].solution[req].solution_path[i + 1][1]
                    cost_moving = np.float(
                        self.alg.problem.D_cost[cut_start.index, cut_end.index, cut_start_direction, cut_end_direction])
                    cost_reclaim = np.float(cut_end.cost_reclaim)
                    cost = cost_reclaim + cost_moving
                    utility = np.round(np.divide(cost, cut_end.cut_tonnage), self.alg.problem.precision)
                    temp[idm].append(utility)


    #
    #
    #         # :: (cut,direction)
    #         if idm == 0:
    #             utility_start = np.divide(initial_cut_tuple[0].cost_reclaim, initial_cut_tuple[0].cut_tonnage)
    #             temp[idm].append(np.round(utility_start,self.problem.precision))
    #         # sol 0
    #
    #
    #     utility_1 =
    #
    #     cost_reclaim = np.float(end_node.cost_reclaim)  # no moving cost for initial cut
    #
    # utility = np.divide(cost_reclaim, end_node.cut_tonnage)
    # utility = np.round(utility, self.problem.precision)
    #
    # cost = np.round(cost_reclaim, self.problem.precision)
