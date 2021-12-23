from src.machine import Machine
from src.utils import Solution, np

class LocalSearch():
    """
    this class takes a dictionary of solutions, total obtained path and current fitness of total solution and performs local search
    """
    def __init__(self, localsolution):
        self.solver = localsolution
        self.problem = localsolution.problem
        self.size_fitness = [1,2]
        self.range_demand = np.arange(self.problem.number_demand)
        self.range_machine = np.arange(self.problem.number_machine)

    # def show_obj(self, localsolution, id_machine, id_solution):
    #     utility = np.round(localsolution.machine[id_machine].solution[id_solution].utility,self.problem.precision)
    #     if self.problem.parcel_input:
    #         penalty_parcel = np.round(sum(localsolution.solution[id_solution].nodes_viol_parcel),self.problem.precision)
    #         return (utility, penalty_parcel)
    #     else:
    #         return utility
    #
    # def show_total_obj(self, localsolution):
    #     utility = []
    #     viol_parcel = []
    #     for id_solution in self.range_demand:
    #         if self.problem.parcel_input:
    #             viol_parcel.append(np.round(sum(localsolution.solution[id_solution].nodes_viol_parcel),self.problem.precision))
    #         else:
    #             viol_parcel.append(0)
    #
    #         for id_machine in self.range_machine:
    #             utility.append(localsolution.machine[id_machine].solution[id_solution].utility)
    #
    #     return (sum(utility), sum(viol_parcel))

    def swap_iterative(self, localsolution, id_machine, id_solution):
        best_obj = localsolution.show_obj(id_machine=id_machine, id_solution=id_solution)
        # print('original delivery {} machine {}:'.format(id_solution, id_machine), best_obj[::-1]) # original

        improvement = True
        while improvement:
            localsolution = self.swap_adj_full(localsolution=localsolution, id_machine=id_machine, id_solution=id_solution)
            show_obj_new = localsolution.show_obj(id_machine=id_machine, id_solution=id_solution)

            if best_obj == show_obj_new:
                improvement = False
            else:
                improvement = True
                best_obj = show_obj_new
                # print('improved delivery {} machine {}:'.format(id_solution, id_machine), best_obj[::-1])


        return localsolution



    def swap_adj_full(self, localsolution, id_machine, id_solution):
        len_solution = len(localsolution.machine[id_machine].solution[id_solution].solution_path)

        valid_improved_swaps = {}

        for i in range(len_solution):
            j = i + 1 # j > 1 always
            if i >= 3 and j < len_solution:
                if self.problem.direction == 2:
                    (success, D) = self.swap_adj_bidirection(id_machine=id_machine, id_solution=id_solution, len_solution=len_solution, index_i=i, index_j=j)
                else:
                    (success, D) = self.swap_adj(id_machine=id_machine, id_solution=id_solution, len_solution=len_solution, index_i=i, index_j=j)

                if success:
                    valid_improved_swaps[(i,j)]=D

        K = list(valid_improved_swaps.keys())

        fitness = np.zeros((len(K),self.size_fitness[localsolution.problem.parcel_input]))

        for i,k in enumerate(K):
            fitness[i] = valid_improved_swaps[k]['fitness']

        # find the best swap
        if len(fitness) > 0:
            best_swap_idx = self.my_lex_sort(fitness)[0]
            best_swap_key = K[best_swap_idx]
            # replace values in solver

            localsolution.replace_values(D=valid_improved_swaps[best_swap_key], id_machine=id_machine, id_solution=id_solution)


        return localsolution


    def calc_viol(self, avg, case, id_demand):
        # try:
        # :: ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
        # we calculate the violation of bounds by bracket operator
        avg = np.round(avg, self.problem.precision)
        type_current_demand = self.problem.demand_type[id_demand]
        size = avg.shape

        if case == 'avg':
            min_limit = np.full(size,self.problem.limits_lower_solution[type_current_demand])
            max_limit = np.full(size,self.problem.limits_upper_solution[type_current_demand])
        else:
            min_limit = np.full(size,self.problem.limits_lower_window[type_current_demand])
            max_limit = np.full(size,self.problem.limits_upper_window[type_current_demand])

        # we see only Fe has a lower bound therefore the lower bound violation is calculated only considering Fe
        viol_lower = np.abs(np.minimum(np.divide(avg[:,1] - min_limit[:,1], min_limit[:,1]), np.zeros(size[0])))
        viol_upper = np.sum(np.abs(np.minimum(np.divide(max_limit - avg, max_limit), np.zeros(size))),axis=1)
        viol = np.round(viol_lower + viol_upper, self.problem.precision)

        return viol


    def my_lex_sort(self, obj):

        if obj.shape[-1] == 2:
            sorted_idx = np.lexsort(
                (obj[:, 0], obj[:, 1]))  # utility, penalty_ parcel
        else:
            sorted_idx = np.argsort(obj[:,0])

        return sorted_idx

    def calc_swap_time(self,id_machine, id_solution, index_i, index_j, direction_i, direction_j):
        # return a tuple of two tuple: ((t_start_i, t_finish_i, t_start_j, t_finish_j)) for new i,j
        # return a tuple of t0, t1, t2 representing the new completion time for edges (i-1,j), (j,i) and (i,j+1) wrt. new indexes


        E = [(index_i-1, index_j), (index_j ,index_i)] # all affected edges # not considered index_j + 1 not exist

        if index_j+1 < len(self.solver.machine[id_machine].solution[id_solution]):
            E.append((index_i, index_j+1))

        time_swap = []

        for edge in E:

            edge_start = self.solver.machine[id_machine].solution[id_solution].solution_path[edge[0]]
            edge_end = self.solver.machine[id_machine].solution[id_solution].solution_path[edge[1]]

            time_move = self.solver.problem.D_cost[(edge_start[0].index, edge_end[0].index, direction_i, direction_j)]
            time_reclaim = edge_end[0].cost_reclaim
            t_finish_i = (time_move + time_reclaim) # + t_ref
            # t_ref = t_finish_i
            time_swap.append(t_finish_i)

        # t0, t1, t2 = time_swap
        time_completion = np.array(time_swap)
        t_ref = self.solver.machine[id_machine].solution[id_solution].time_start[index_i]
        time_completion[0] = time_completion[0] + t_ref
        time_completion = np.cumsum(time_completion)

        # # validate time_completion:
        # for i,t in enumerate(time_completion):
        #     if i+1<len(time_completion):
        #         if time_completion(i+1)<time_completion(i):
        #             raise EOFError

        return (np.array(time_swap), time_completion)
        # separate time for each edge, time of completion wrt. the reference time

    def calc_change_fitness_swap(self, id_machine, id_solution, index_i, index_j, time_swap, time_completion, L, P):
        success = False
        D = None
        orig_viol_parcel = np.inf
        new_viol_parcel = np.inf

        # calc penalty violation
        # adopt time_completion
        T = self.solver.machine[id_machine].solution[id_solution].time_finish.copy()

        T[index_i:index_j+2] = time_completion # time completion new for the new machine

        T_new = [] # new completion time for this solution after swap
        L_new = [] # new index of cuts for this solution after swap
        for idm in range(self.solver.problem.number_machine):
            if id_machine == idm:
                T_new.extend(T)
                L_new.extend(L)
            else:
                T_new.extend(self.solver.machine[idm].solution[id_solution].time_finish)
                L_new.extend(self.solver.machine[idm].solution[id_solution].solution_ids)

        # make the new solution array for this demand for all machines
        new_solution = [x for _, x in sorted(zip(T_new,L_new))]

        if self.solver.problem.parcel_input:
            # we should compare new_solution with solver.solution[id_solution].solution_ids :: to find different parcels
            diff_parcel_index = []
            diff_parcel = []
            r = range(len(self.solver.solution[id_solution].solution_ids))
            for i in r:
                if i >= 3:
                    orig = self.solver.solution[id_solution].solution_ids[i-2:i+1]
                    new = new_solution[i-2:i+1]
                    if orig != new:
                        if set(orig) != set(new):
                            # check how many real different paracels exist : because sometime the order is different in a parcel but all items are the same
                            diff_parcel_index.append(i-3)
                            diff_parcel.append(new)

            viol_new = []

            for parcel in diff_parcel:
                parcel_avg = np.mean([self.solver.problem.D_nodes[x].chemical for x in parcel],axis=0)
                viol_new.append(self.calc_viol(avg=parcel_avg.reshape(1,6), case='parcel', id_demand=id_solution)[-1])

            # replace new parcel violations in the swapped solution
            viol_solution_new = self.solver.solution[id_solution].nodes_viol_parcel.copy()

            viol_solution_new[diff_parcel_index] = viol_new


            orig_viol_parcel = np.round(sum(self.solver.solution[id_solution].nodes_viol_parcel),self.problem.precision)
            new_viol_parcel =  np.round(sum(viol_solution_new),self.problem.precision)

            if orig_viol_parcel < new_viol_parcel:
                # ignore this purturbation and no need to calculate the utilitu
                return (False, None)
            # elif orig_viol_parcel > new_viol_parcel:
            #     success = True
            # else: equal : do nothing

        # calc utility change
        # for utility we only consider one machine separately : three edges at most are changing
        index_cuts = [index_j, index_i]

        if index_j+1 < len(self.solver.machine[id_machine].solution[id_solution]):
            index_cuts.append(index_j+1)

        tonnage_swap = [self.solver.machine[id_machine].solution[id_solution].solution_path[idx][0].cut_tonnage for idx in index_cuts]
        utility_swap = time_swap/tonnage_swap
        orig_utility =  self.solver.machine[id_machine].solution[id_solution].utility
        temp_nodes_utility = self.solver.machine[id_machine].solution[id_solution].nodes_utility.copy()
        temp_nodes_utility[index_i:index_j + 2] = utility_swap
        new_utility = orig_utility - \
                      sum(self.solver.machine[id_machine].solution[id_solution].nodes_utility[index_i:index_j + 2]) + sum(utility_swap)

        # if new_utility < orig_utility or success:
            # record this swap change

        # compare the solution with orig
        obj = np.zeros([2,self.size_fitness[self.problem.parcel_input]])
        if obj.shape[-1] == 2:
            obj[0,:] = [orig_utility, orig_viol_parcel]
            obj[1,:] = [new_utility, new_viol_parcel]
        else:
            obj[0]=[orig_utility]
            obj[1] =[new_utility]

        rank = self.my_lex_sort(obj)

        if rank[0] == 1:
            success = True
            if self.solver.problem.parcel_input:
                fitness = (new_utility, new_viol_parcel)
                D = {'fitness':fitness, 'completion_time':T, 'solution_machine':L, 'solution_path':P, 'nodes_utility':temp_nodes_utility, 'nodes_viol_parcel':viol_solution_new}
                # print('fitness',fitness)
            else:
                fitness = (new_utility)
                D = {'fitness':fitness, 'completion_time':T, 'solution_machine':L, 'solution_path':P,
                     'nodes_utility':temp_nodes_utility, 'nodes_viol_parcel':None}
        else:
            success = False
            D = None

        return (success, D)


    def handle_swap_common(self, id_machine, id_solution, index_cut_main, time_completion, index_time_completion, cut_main):
        valid = True
        machine_rows_common = {2: {0,1}, 3: {1,2}}
        common_machine = machine_rows_common[cut_main.node_row].copy()
        common_machine.remove(id_machine)
        id_other_machine = list(common_machine)[-1]

        if id_other_machine < self.problem.number_machine:

            t_finish_old = self.solver.machine[id_machine].solution[id_solution].time_finish[index_cut_main]
            t_finish_new = time_completion[index_time_completion]

            suspect_cuts_ids = np.where((self.solver.machine[id_other_machine].solution[id_solution].time_start>=min(t_finish_old, t_finish_new)) &
                                        (self.solver.machine[id_other_machine].solution[id_solution].time_start<=max(t_finish_old, t_finish_new)))[-1]

            for index_cut in suspect_cuts_ids:
                edge_suspect = self.solver.machine[id_other_machine].solution[id_solution].solution_path[index_cut]
                cut_suspect = edge_suspect[0]
                direction_suspect = edge_suspect[-1]
                if cut_main.index in set(cut_suspect.prec[direction_suspect]):
                    valid = False
                    break


        return valid

    def swap_adj_bidirection(self, id_machine, id_solution, len_solution, index_i,index_j):
        success = False
        D = None
        L = self.solver.machine[id_machine].solution[id_solution].solution_ids.copy()
        P = self.solver.machine[id_machine].solution[id_solution].solution_path.copy()
        temp = L[index_i]
        temp_P = P[index_i]
        # perform swap
        L[index_i] = L[index_j]
        L[index_j] = temp
        #
        P[index_i] = P[index_j]
        P[index_j] = temp_P
        # possible directions
        list_directions = [('SN','SN'), ('SN','NS'), ('NS','SN'), ('NS','NS')]
        list_obj = []

        cut_common = np.zeros(3)

        if index_j + 1 < len_solution:
            edge_j1 = self.solver.machine[id_machine].solution[id_solution].solution_path[index_j+1]
            cut_j1 = edge_j1[0]
            direction_j1 = edge_j1[-1] # j+1 cut

            if cut_j1.node_common:
                cut_common[2] = 1

        for direction in list_directions:
            P[index_i] = (P[index_i][0],direction[0])
            P[index_j] = (P[index_j][0], direction[1])

            edge_i = (self.solver.machine[id_machine].solution[id_solution].solution_path[index_i][0], direction[0])
            edge_j = (self.solver.machine[id_machine].solution[id_solution].solution_path[index_j][0], direction[1])

            cut_i = edge_i[0]
            cut_j = edge_j[0]

            direction_i = edge_i[-1]
            direction_j = edge_j[-1]

            # print(direction_i, direction_j)

            if cut_i.node_common:
                cut_common[0] = 1

            if cut_j.node_common:
                cut_common[1] = 1

            # if index_i is not a precedence for index_j with the direction_i : it is valid
            if not cut_i.index in set(cut_j.prec[direction_j]):
                time_swap, time_completion = self.calc_swap_time(id_machine=id_machine, id_solution=id_solution,
                                                                 index_i=index_i, index_j=index_j,direction_i=direction_i, direction_j=direction_j)

                if not np.any(cut_common):
                    # valid
                    # calculate the change in the fitness, if better than the original, record the fitness function and the swap indicator
                    (success, D) = self.calc_change_fitness_swap(id_machine=id_machine, id_solution=id_solution, index_i=index_i, index_j=index_j,
                                                                 time_swap=time_swap, time_completion=time_completion, L=L, P=P)
                    # return (success, D)
                    list_obj.append((success, D))
                else:
                    valid = True
                    if cut_i.node_common and valid:
                        valid = self.handle_swap_common(id_machine=id_machine, id_solution=id_solution, index_cut_main=index_i, time_completion=time_completion, index_time_completion = 0, cut_main=cut_i)
                    if cut_j.node_common and valid:
                        valid = self.handle_swap_common(id_machine=id_machine, id_solution=id_solution, index_cut_main=index_j, time_completion=time_completion, index_time_completion=1, cut_main=cut_j)

                    if index_j + 1 < len_solution:
                        if cut_j1.node_common and valid:
                            valid = self.handle_swap_common(id_machine=id_machine, id_solution=id_solution, index_cut_main=index_j+1, time_completion=time_completion, index_time_completion=2, cut_main=cut_j1)

                    if valid:
                        (success, D) = self.calc_change_fitness_swap(id_machine=id_machine, id_solution=id_solution, index_i=index_i, index_j=index_j,
                                                                     time_swap=time_swap, time_completion=time_completion, L=L, P=P)
                        list_obj.append((success, D))
                    else:
                        # return (False, None)
                        list_obj.append((False, None))

        swap_true = [x[1] for x in list_obj if x[0]==True]

        if len(swap_true) > 0:
            # find the best one wrt. utility:
            idx = np.argmin([x['fitness'][0] for x in swap_true])
            success = True
            D = swap_true[0]

            return (success, D)

        else:

            return (False, None)


    def swap_adj(self, id_machine, id_solution, len_solution, index_i,index_j):
        success = False
        D = None
        # only adjacent cuts
        # when we do the swap,
        L = self.solver.machine[id_machine].solution[id_solution].solution_ids.copy()
        P = self.solver.machine[id_machine].solution[id_solution].solution_path.copy()
        temp = L[index_i]
        temp_P = P[index_i]

        L[index_i] = L[index_j]
        L[index_j] = temp

        P[index_i] = P[index_j]
        P[index_j] = temp_P

        # check validity of precedence constraint for this machine
        edge_i = self.solver.machine[id_machine].solution[id_solution].solution_path[index_i]
        edge_j = self.solver.machine[id_machine].solution[id_solution].solution_path[index_j]

        cut_common = np.zeros(3)

        if index_j + 1 < len_solution:
            edge_j1 = self.solver.machine[id_machine].solution[id_solution].solution_path[index_j+1]
            cut_j1 = edge_j1[0]
            direction_j1 = edge_j1[-1] # j+1 cut

            if cut_j1.node_common:
                cut_common[2] = 1

        cut_i = edge_i[0]
        cut_j = edge_j[0]

        direction_i = edge_i[-1]
        direction_j = edge_j[-1]

        if cut_i.node_common:
            cut_common[0] = 1

        if cut_j.node_common:
            cut_common[1] = 1

        # if index_i is not a precedence for index_j with the direction_i : it is valid
        if not cut_i.index in set(cut_j.prec[direction_j]):
            time_swap, time_completion = self.calc_swap_time(id_machine=id_machine, id_solution=id_solution,
                                                             index_i=index_i, index_j=index_j, direction_i = direction_i, direction_j = direction_j)

            if not np.any(cut_common):
                # valid
                # calculate the change in the fitness, if better than the original, record the fitness function and the swap indicator
                (success, D) = self.calc_change_fitness_swap(id_machine=id_machine, id_solution=id_solution, index_i=index_i, index_j=index_j,
                                                             time_swap=time_swap, time_completion=time_completion, L=L, P=P)
                return (success, D)
            else:
                valid = True
                if cut_i.node_common and valid:
                    valid = self.handle_swap_common(id_machine=id_machine, id_solution=id_solution, index_cut_main=index_i, time_completion=time_completion, index_time_completion = 0, cut_main=cut_i)
                if cut_j.node_common and valid:
                    valid = self.handle_swap_common(id_machine=id_machine, id_solution=id_solution, index_cut_main=index_j, time_completion=time_completion, index_time_completion=1, cut_main=cut_j)

                if index_j + 1 < len_solution:
                    if cut_j1.node_common and valid:
                        valid = self.handle_swap_common(id_machine=id_machine, id_solution=id_solution, index_cut_main=index_j+1, time_completion=time_completion, index_time_completion=2, cut_main=cut_j1)

                if valid:
                    (success, D) = self.calc_change_fitness_swap(id_machine=id_machine, id_solution=id_solution, index_i=index_i, index_j=index_j,
                                                                 time_swap=time_swap, time_completion=time_completion, L=L, P=P)
                    return (success, D)
                else:
                    return (False, None)

        return (success, D)


class LocalSolution():

    def __init__(self, solver):
        self.problem = solver.problem
        self.range_machine = np.arange(self.problem.number_machine)
        self.range_demand = np.arange(self.problem.number_demand)
        self.machine = {k: Machine(problem=self.problem, id_machine=k) for k in self.range_machine}
        self.solution = [Solution(k) for k in self.range_demand]
        self.copy_solution(solver=solver)

    # def generate_solution_path(self, path):
        # path_new = []
        # for edge in path:
        #     path_new.append((edge[0].index,edge[1]))
        #
        # return path_new

    def copy_solution(self, solver):
        for ids in self.range_demand:
            for idm in self.range_machine:
                self.machine[idm].solution[ids].solution_path = solver.machine[idm].solution[ids].solution_path.copy()
                self.machine[idm].solution[ids].time_start = solver.machine[idm].solution[ids].time_start.copy()
                self.machine[idm].solution[ids].time_finish = solver.machine[idm].solution[ids].time_finish.copy()
                self.machine[idm].solution[ids].solution_ids = solver.machine[idm].solution[ids].solution_ids.copy()
                self.machine[idm].solution[ids].utility = solver.machine[idm].solution[ids].utility
                self.machine[idm].solution[ids].nodes_utility = solver.machine[idm].solution[ids].nodes_utility.copy()

            self.solution[ids].solution_ids = solver.solution[ids].solution_ids.copy()
            self.solution[ids].nodes_viol_parcel = solver.solution[ids].nodes_viol_parcel.copy()

        self.total_obj = np.zeros(solver.total_obj.shape)+np.inf
        self.total_obj[1] = solver.total_obj[1] # average


    def replace_values(self, D, id_machine, id_solution):

        self.machine[id_machine].solution[id_solution].time_finish = D['completion_time']
        self.machine[id_machine].solution[id_solution].nodes_utility = D['nodes_utility']
        self.machine[id_machine].solution[id_solution].solution_ids = D['solution_machine']
        self.machine[id_machine].solution[id_solution].solution_path = D['solution_path']


        if self.problem.parcel_input:
            self.solution[id_solution].nodes_viol_parcel = D['nodes_viol_parcel']
            self.machine[id_machine].solution[id_solution].utility = D['fitness'][0]
        else:
            self.machine[id_machine].solution[id_solution].utility = D['fitness']

        # generate time_start
        if id_solution == 0:
            initial_time = 0
        else:
            id_previous_solution = id_solution-1
            while len(self.machine[id_machine].solution[id_previous_solution].time_finish) == 0:
                id_previous_solution -= 1

            initial_time = self.machine[id_machine].solution[id_previous_solution].time_finish[-1]

        self.machine[id_machine].solution[id_solution].time_start = \
            [initial_time]+self.machine[id_machine].solution[id_solution].time_finish[:-1]


    def show_obj(self, id_machine, id_solution):
        utility = np.round(self.machine[id_machine].solution[id_solution].utility,self.problem.precision)
        if self.problem.parcel_input:
            penalty_parcel = np.round(sum(self.solution[id_solution].nodes_viol_parcel),self.problem.precision)
            return (utility, penalty_parcel)
        else:
            return utility

    def calc_total_obj(self):
        utility = []
        viol_parcel = []
        for id_solution in self.range_demand:
            if self.problem.parcel_input:
                viol_parcel.append(np.round(sum(self.solution[id_solution].nodes_viol_parcel),self.problem.precision))

            for id_machine in self.range_machine:
                utility.append(self.machine[id_machine].solution[id_solution].utility)

        self.total_obj[0] = np.sum(utility)

        if self.problem.parcel_input:
            self.total_obj[-1] = np.sum(viol_parcel)
