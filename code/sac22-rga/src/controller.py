from src.utils import np, Solution, pickle, bz2, cPickle, pd
from src.machine import Machine
from src.test_controller import TestParallel_Controller

class Parallel_Controller():
    """
    This class controls the machine interactions in the stockpile recovery problem
    """

    def __repr__(self):
        pass

    def __contains__(self, node_id):
        return node_id in self.visited

    def __init__(self, problem, greedy_param, selection_pressure, ant_kw):
        self.problem = problem
        self.greedy_param = greedy_param
        self.SP = selection_pressure
        self.solution = [Solution(k) for k in np.arange(self.problem.number_demand)]
        self.active_demand = 0  # demands start at 0
        self.active_type = self.problem.demand_type[self.active_demand]
        self.machine_idx = set(np.arange(self.problem.number_machine))  # +1
        self.machine = {k: Machine(problem=self.problem, id_machine=k) for k in self.machine_idx}
        # self.edges = [set() for _ in np.arange(self.problem.number_machine)]
        self.edges = set()
        self.idle_machines = []
        self.machine_path = []
        self.visited = set()
        self.tonnage_queue = np.zeros([self.problem.number_demand, self.problem.number_machine], dtype=float)
        self.solution_path = []
        self.solution_machine_ids = []  # to track the recovery solution by machines
        self.initiate_queue() # to track the queue for machines
        self.time_release = np.zeros(self.problem.number_machine)
        self.time_start = np.zeros(self.problem.number_machine)
        self.stockpile_busy = dict.fromkeys(self.machine_idx, ())
        self.time_current = np.zeros(1)  # time of all jobs starts from 0.
        self.status = np.array([True] * self.problem.number_machine)
        self.generate_str()

        if self.problem.type_algorithm == 'MMAS':
            self.ant_kw = ant_kw


        # print('{} algorithm initialisation : succesful'.format(self.problem.type_algorithm))


    def initiate_queue(self):
        self.queue_keys_cat = ['end_node', 'end_direction'] # categorical
        self.queue_keys_num = ['end_utility', 'end_cost', 'end_completion', 'end_demand'] # numerical
        self.queue = {k:[None]*self.problem.number_machine for k in self.queue_keys_cat}
        self.queue.update({k:np.zeros(self.problem.number_machine)-1 for k in self.queue_keys_num})
        self.queue['end_demand'].dtype=int

    def generate_str(self):
        alg = self.problem.type_algorithm
        self.string_save = '{}_{}_{}'.format(alg, self.greedy_param, self.problem.number_demand)

    def next_time_step(self):
        self.time_current = self.time_release.min()
        self.status = (self.time_release <= self.time_current)

    def my_lex_sort(self, obj_neighbor):
        if self.problem.parcel_input:
            sorted_idx = np.lexsort(
                (obj_neighbor[:, 0], obj_neighbor[:, 2], obj_neighbor[:, 1]))  # utility, penalty_ parcel, penalty_avg
        else:
            sorted_idx = np.lexsort((obj_neighbor[:, 0], obj_neighbor[:, 1]))

        return sorted_idx

    def selection_MMAS(self, obj_neighbor, key_edges):
        # MMAS algorithm selection
        if self.problem.parcel_input:
            cond = np.sum((obj_neighbor[:, 1] == 0) & (obj_neighbor[:, 2] == 0)) == obj_neighbor.shape[0] # all feasible : we employ utility

        else:
            cond = np.sum(obj_neighbor[:, 1] == 0) == obj_neighbor.shape[0] # some infeasible : we employ ranking

        # if cond:
        #     # all are feasible -- we use fitness proportionate on the utility
        #     idx_next = self.rwheel_FP_MMAS(obj_neighbor[:, 0], key_edges=key_edges)
        # else:
        #     # we use linear ranking based method
        #     idx_next = self.rwheel_LR_MMAS(obj_neighbor, key_edges=key_edges)
        idx_next = self.rwheel_LR_MMAS(obj_neighbor, key_edges=key_edges)

        return idx_next

    def rwheel_LR_MMAS(self, obj_neighbor, key_edges):
        n = obj_neighbor.shape[0]
        sorted_idx = self.my_lex_sort(obj_neighbor)  # best to worst
        r = np.arange(n)
        p_rank = (2 - self.SP) / n + (2 * r * (self.SP - 1)) / (
                n * (n - 1))  # probability based on ranks -- worst to best
        second_term =  p_rank[::-1] ** self.ant_kw['beta'] # best to worst
        tau = np.array([self.ant_kw['tau'][key_edges[j]] for j in sorted_idx]) # best to worst
        # since 0 < tau < 1: to consider the effect of pheromone we should consider 1/alpha, the same applies to p_rank but
        # it has already been considered in p_rank calculation
        first_term = tau ** (1/self.ant_kw['alpha']) # best to worst
        term = first_term * second_term
        selection_probs = term/np.sum(term)
        # index_sorted_idx = np.random.choice(r, p=selection_probs)  # choose a random index from best to worst
        index_sorted_idx = self.problem.rng.choice(r, p=selection_probs)  # choose a random index from best to worst
        output = sorted_idx[index_sorted_idx]
        return output

    def selection_GA(self, obj_neighbor):
        # Greedy algorithm selection
        sorted_idx = self.my_lex_sort(obj_neighbor)

        return sorted_idx[0]

    def selection_RGA(self, obj_neighbor):
        if self.problem.parcel_input:
            cond = np.sum((obj_neighbor[:, 1] == 0) & (obj_neighbor[:, 2] == 0)) == obj_neighbor.shape[0]
        else:
            cond = np.sum(obj_neighbor[:, 1] == 0) == obj_neighbor.shape[0]

        if cond:
        #     # all are feasible -- we use fitness proportionate on the utility
            idx_next = self.rwheel_FP_RGA(obj_neighbor)
            # print('X')
        else:
        # we use linear ranking based method
            idx_next = self.rwheel_LR_RGA(obj_neighbor)
            # print('Y')

        return idx_next

    def rwheel_FP_MMAS(self, utility, key_edges):
        temp = 1 / utility
        tau = np.array([self.ant_kw['tau'][j] for j in key_edges])
        second_term = temp ** self.ant_kw['beta']
        first_term = tau ** (1/self.ant_kw['alpha'])
        term = first_term * second_term
        selection_probs = term/np.sum(term)
        # output = np.random.choice(np.arange(len(utility)), p=selection_probs)
        output = self.problem.rng.choice(np.arange(len(utility)), p=selection_probs)
        return output

    def rwheel_FP_RGA(self, obj_neighbor):
        n = obj_neighbor.shape[0]
        sorted_idx = self.my_lex_sort(obj_neighbor) # best to worst
        temp = 1/obj_neighbor[:,0]
        selection_probs = np.divide(temp ** self.greedy_param, np.sum(temp ** self.greedy_param))
        s = selection_probs[sorted_idx] # from best to worst
        index_sorted_idx = self.problem.rng.choice(n, size=1, replace=False, p=s)[-1]
        output = sorted_idx[index_sorted_idx]

        # n = obj_neighbor.shape[0]
        # temp = 1/obj_neighbor[:,0]
        # selection_probs = np.divide(temp ** self.greedy_param, np.sum(temp ** self.greedy_param))
        # index_sorted_idx = self.problem.rng.choice(n, size=1, replace=False, p=selection_probs)[-1]
        # output = index_sorted_idx

        return output

    def rwheel_LR_RGA(self, obj_neighbor):
        n = obj_neighbor.shape[0]
        sorted_idx = self.my_lex_sort(obj_neighbor)  # best to worst
        r = np.arange(n)
        p_rank = (2 - self.SP) / n + (2 * r * (self.SP - 1)) / (
                    n * (n - 1))  # probability based on ranks -- worst to best
        p_rank_greedy = np.divide(p_rank ** self.greedy_param, np.sum(p_rank ** self.greedy_param))#
        # index_sorted_idx = np.random.choice(r, p=p_rank_greedy[::-1])  # choose a random index from best to worst
        index_sorted_idx = self.problem.rng.choice(r, p=p_rank_greedy[::-1])  # choose a random index from best to worst
        output = sorted_idx[index_sorted_idx]
        return output

    def run(self):
        self.termination = False
        # termination_demand = False
        available_machines_idx = np.arange(self.problem.number_machine)  # +1
        self.construct_solution_initial(available_machines_idx=available_machines_idx)
        while self.termination == False or np.any(self.queue['end_demand']>0):

            self.next_time_step()  # increase time unit
            available_machines_idx = np.where(self.status == True)[-1]

            for id_machine in available_machines_idx:
                self.release_machine(
                    id_machine=id_machine)  # active machines should be released and finish reclaiming their cuts

            tonnage_queue = np.sum(self.tonnage_queue[self.active_demand])
            if self.solution[self.active_demand].tonnage + tonnage_queue >= self.problem.demand_capacity[self.active_demand]:
                self.terminate_demand(idm=id_machine)


            if not self.termination:
                self.find_neighbor_after_reclaim(direction_reclaimer='SN', id_machine=id_machine)

                if self.problem.direction == 2:
                    self.find_neighbor_after_reclaim(direction_reclaimer='NS', id_machine=id_machine)

                return_code = self.construct_solution_run(available_machines_idx=available_machines_idx)

                if return_code == 1:
                    if self.problem.parcel_input:
                        self.total_obj = np.array([1e12, 1e12, 1e12])
                    else:
                        self.total_obj = np.array([1e12, 1e12])
                    return

        self.calc_total_obj()
        # np.set_printoptions(suppress=True, precision=4)
        # self.validate(test_case=3)
        # Export sol
        # finish

    def calc_total_obj(self):
        r = np.arange(self.problem.number_demand)
        total_utility = np.array([np.sum(self.solution[id_demand].nodes_utility) for id_demand in r])
        total_penalty_avg = np.array([self.calc_viol(self.solution[id_demand].penalty_average.reshape(1,6),case='avg',id_demand=id_demand) for id_demand in r])
        if self.problem.parcel_input:
            total_penalty_parcel = np.array([np.sum(self.solution[id_demand].nodes_viol_parcel) for id_demand in r])
            self.total_obj = np.array([total_utility.sum(), total_penalty_avg.sum(), total_penalty_parcel.sum()]) # utility, penalty_ parcel, penalty_avg
        else:
            self.total_obj = np.array([total_utility.sum(), total_penalty_avg.sum()])


    def terminate_demand(self,idm):
            self.active_demand = self.active_demand+1

            if self.active_demand == self.problem.number_demand:
                self.termination = True
                self.terminate_machine(idm)
                self.active_demand = self.problem.number_demand - 1

    def construct_solution_initial(self, available_machines_idx):
        end_id_direction = 'SN'
        ## find initial cut for each machine and push it to the queue
        ## we assume initial direction of reclamation is all 'SN'
        for idm in available_machines_idx:
            initial_cut = self.machine[idm].find_active_neighborhood_initial()
            initial_queue = self.calc_initial_cut(end_id=initial_cut, end_id_direction=end_id_direction, id_machine=idm)
            self.push_cut_into_queue(cut_queue=initial_queue, id_machine=idm)

    def generate_neighbor_keys(self, available_machines_idx, idm):
        active_neighborhood = {k:[] for k in available_machines_idx}
        active_neighborhood[idm] = self.machine[idm].find_active_neighborhood_run(
            stockpile_busy=self.stockpile_busy)
        # idm = available_machines_idx[-1]
        key_neighbor = []
        neighbor_id_SN = active_neighborhood[idm]['SN']


        for neighbor in iter(neighbor_id_SN):
            key_neighbor.append((neighbor, 'SN'))

        if self.problem.direction == 2:
            neighbor_id_NS = active_neighborhood[idm]['NS']
            for neighbor in iter(neighbor_id_NS):
                key_neighbor.append((neighbor, 'NS'))

        return key_neighbor

    def terminate_machine(self,idm):
        self.time_release[idm] =  np.inf

    def construct_solution_run(self, available_machines_idx):

        idm = available_machines_idx[-1]

        # active_neighborhood is all the neighbors available to current position with respect to the idm

        if len(available_machines_idx) > 1:
            #   raise Exception('Not implemented') # two machines have become active at the same time
            return_code = 1
        else:
            return_code = 0
            key_neighbor = self.generate_neighbor_keys(available_machines_idx=available_machines_idx, idm=idm)

            while len(key_neighbor) == 0:
                new_machine_demand = self.machine[idm].machine_demand + 1
                if new_machine_demand < self.problem.number_demand:
                    self.machine[idm].machine_demand += 1
                    key_neighbor = self.generate_neighbor_keys(available_machines_idx=available_machines_idx,idm=idm)
                else:
                    self.terminate_machine(idm=idm)
                    break

            if len(key_neighbor) > 0:
                # evaluation
                obj_neighbor, end_node_all, cost_all = self.calc_obj(key_neighbor, idm)

                if len(obj_neighbor) == 1:
                    next_index = 0
                else:
                    # selection
                    if self.problem.type_algorithm in ['DGA', 'RGA']:
                        if self.greedy_param == 0:
                            next_index = self.selection_GA(obj_neighbor)
                        else:
                            next_index = self.selection_RGA(obj_neighbor)

                    elif self.problem.type_algorithm == 'MMAS':
                        #  MMAS selection specific
                        key_edges = [self.problem.D_edge[(self.machine[idm].current[0].index,x[0],self.machine[idm].current[1],x[1])] for x in key_neighbor]
                        next_index = self.selection_MMAS(obj_neighbor, key_edges)

                next_key = key_neighbor[next_index]
                queue_cut = {'end_node': end_node_all[next_index], 'end_utility': obj_neighbor[next_index, 0],
                         'end_cost': cost_all[next_index], 'end_completion': self.time_current + cost_all[next_index],
                         'end_direction': next_key[1], 'end_demand': self.machine[idm].machine_demand}
                self.push_cut_into_queue(cut_queue=queue_cut, id_machine=idm)  # push it into the queue
        return return_code

    def calc_initial_cut(self, end_id, end_id_direction, id_machine):
        (end_utility, end_cost, end_node) = self.machine[id_machine].calc_utility_initial(
            (end_id, end_id_direction))  # initial reclaim

        initial_cut = {'end_node': end_node, 'end_utility': end_utility, 'end_cost': end_cost, 'end_completion': self.time_current+end_cost,
                       'end_direction': end_id_direction, 'end_demand': self.machine[id_machine].machine_demand}

        return initial_cut

    def push_cut_into_queue(self, cut_queue, id_machine):
        # push
        id_demand = self.machine[id_machine].machine_demand
        self.remove_node(id_machine=id_machine, end_node=cut_queue['end_node'],
                         end_direction=cut_queue['end_direction'])

        for k in self.queue:
            self.queue[k][id_machine] = cut_queue[k]


        # self.queue[id_machine]['end_demand'] = self.machine[id_machine].machine_demand
        self.tonnage_queue[id_demand, id_machine] = np.float(cut_queue['end_node'].cut_tonnage)
        # self.queue[id_machine]['next_penalty'] = self.machine[id_machine].
        self.time_release[id_machine] = self.queue['end_completion'][id_machine]  # self.queue[id_machine]['end_cost'] + self.time_current
        self.time_start[id_machine] = float(self.time_current)
        self.stockpile_busy[id_machine] = (self.queue['end_node'][id_machine].node_row, self.queue['end_node'][id_machine].node_stockpile)

    def calc_obj(self, neighbor_key, id_machine):
        """
        :param neighbor_key:
        :param id_machine:
        :return:
        tuple: (objective function, cut objects, cost of nodes) for all nodes in the neighbor_key
        """

        output_obj = np.zeros([len(neighbor_key), self.problem.size_objective[int(self.problem.parcel_input)]])
        # 0:cost, 1:utility, 2: penalty_1, 3: penalty_2

        (utility_all, cost_all, end_node_all, chemical_all) = self.machine[id_machine].calc_utility(neighbor_key)
        # cost_all : completion time for all cuts
        output_obj[:, 0] = utility_all

        id_demand = self.machine[id_machine].machine_demand
        len_solution = len(self.solution[id_demand])
        penalty_avg_all = self.calc_penalty_avg(chemical_all, id_demand=id_demand, len_solution=len_solution)
        output_obj[:, 1] = penalty_avg_all

        if self.cond_parcel(id_demand):
            penalty_parcel_all = self.calc_penalty_parcel(chemical_all, cost_all, id_demand=id_demand)
            output_obj[:, 2] = penalty_parcel_all

            # self.validate_penalty_parcel(chemical_all, cost_all)

        return (output_obj, end_node_all, cost_all)

    def cond_parcel(self, id_demand):
        return (self.problem.parcel_input and len(self.solution[id_demand])>3)

    def calc_penalty_avg(self, end_node_chemical, id_demand, len_solution):

        temp_added_value_avg = np.divide(end_node_chemical - self.solution[id_demand].penalty_average,len_solution + 1)  # add an value to an existing average
        temp_avg = self.solution[id_demand].penalty_average + temp_added_value_avg
        viol_penalty_avg = self.calc_viol(avg=temp_avg, case='avg', id_demand=id_demand)

        return viol_penalty_avg

    def calc_penalty_parcel(self, end_node_chemical, cost_all, id_demand):

        window_avg = self.find_window(end_node_chemical,cost_all, id_demand)

        viol_penalty_parcel = self.calc_viol(avg=window_avg, case='parcel', id_demand=id_demand)

        return viol_penalty_parcel

    def calc_penalty_parcel_release(self, end_node_chemical, id_demand):

        if self.cond_parcel(id_demand):
            # avg_last_three = np.vstack([self.solution[id_demand].nodes_chemical[-2:], end_node_chemical]).mean(axis=0)
            avg_last_three = self.solution[id_demand].nodes_chemical[-3:].mean(axis=0)
            output = self.calc_viol(avg=np.array(avg_last_three).reshape(1,6), case='parcel', id_demand=id_demand)

            # return
        else:
            output = np.float(0)

        return output

    def find_window(self, end_node_chemical, cost_all, id_demand):
        size = end_node_chemical.shape[0] # number of neighbor cuts
        window_avg = np.zeros((size,6)) # output for mean of last three for each neighbor cut
        completion_time = self.time_current + cost_all # time required to complete the jobs for each neighbor cut
        completion_time = completion_time.reshape(size,1)
        this_demand_queue = np.where(self.queue['end_demand'] == id_demand)[-1] # to ensure we only look at cuts in the queue with current id_demand :: logical
        queue_ct = self.queue['end_completion'][this_demand_queue] # completion time recorded for jobs in the queue with current id_demand
        indices_active_machines = this_demand_queue[(queue_ct > 0) & (queue_ct != np.inf)]

        # indices_active_machines = this_demand_queue(queue > 0) & (queue != np.inf))[-1] # active machines where there are pending jobs and the machines are not terminated
        len_active = len(queue_ct)

        full_queue_ct = np.full((size,len_active),queue_ct)
        sort_queue_ct = np.argsort(np.hstack([full_queue_ct,completion_time])) # find out the order of reclamation for cuts in the queue and neighbor cuts

        ravel_last_two = self.solution[id_demand].nodes_chemical[-2:].ravel() # 1D of last two recorded cuts in the nodes_chemical
        for i,e in enumerate(end_node_chemical):
            sort_rank = sort_queue_ct[i] # order of recalmation (completion time)
            if len_active == 2:

                q1q2n = np.hstack([self.queue['end_node'][indices_active_machines[0]].chemical,self.queue['end_node'][indices_active_machines[1]].chemical,e])
                sorted_q1q2n = q1q2n.reshape(3,6)[sort_rank] # reshape of 3 cuts in the queue and neighbor cut :: in order of recalmation (completion time)
                temp = np.vstack([ravel_last_two.reshape(2,6), sorted_q1q2n]) # five cuts :: two last recorded + ordered of queue and neighbor cut
                pos_n = np.where(sort_rank == len_active)[-1] + 2 # position of neighbor cut in all five cuts considering last two cuts recorded in previous steps
                last_three = temp[int(pos_n-2):int(pos_n+1)]

            elif len_active == 1:

                q1n = np.hstack([self.queue['end_node'][indices_active_machines[-1]].chemical,e])
                sorted_q1n = q1n.reshape(2,6)[sort_rank]
                temp = np.vstack([ravel_last_two.reshape(2,6), sorted_q1n]) # four cuts :: two last recorded + ordered of queue and neighbor cut
                pos_n = np.where(sort_rank == len_active)[-1] + 2 # position of neighbor cut in all five cuts considering last two cuts recorded in previous steps
                last_three = temp[int(pos_n-2):int(pos_n+1)]

            elif len_active == 0:
                last_three = np.vstack([ravel_last_two.reshape(2,6), e])


            window_mean = last_three.mean(axis=0)

            window_avg[i,:] = window_mean

        return window_avg

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

    def remove_node(self, id_machine, end_node, end_direction):

        self.machine[id_machine].active_cuts[end_node.product][(end_node.node_row, end_node.node_stockpile)][
            end_direction].discard(end_node.index)

        if self.problem.direction == 2:

            if end_direction == 'SN':
                other_direction = 'NS'
            else:
                other_direction = 'SN'

            if end_node.index in self.machine[id_machine].active_cuts[end_node.product][
                (end_node.node_row, end_node.node_stockpile)][other_direction]:
                self.machine[id_machine].active_cuts[end_node.product][(end_node.node_row, end_node.node_stockpile)][
                    other_direction].discard(end_node.index)

        # if end_node.node_row in [2, 3]: # it is a common node
        if end_node.node_common:
            other_id = self.check_common(id_machine=id_machine, end_node_row = end_node.node_row)

            if other_id > 0:
                if end_node.index in \
                        self.machine[other_id].active_cuts[end_node.product][(end_node.node_row, end_node.node_stockpile)][
                            end_direction]:
                    self.machine[other_id].active_cuts[end_node.product][(end_node.node_row, end_node.node_stockpile)][
                        end_direction].discard(end_node.index)

                if self.problem.direction == 2:

                    if end_direction == 'SN':
                        other_direction = 'NS'
                    else:
                        other_direction = 'SN'

                    if end_node.index in self.machine[other_id].active_cuts[end_node.product][
                        (end_node.node_row, end_node.node_stockpile)][other_direction]:
                        self.machine[other_id].active_cuts[end_node.product][(end_node.node_row, end_node.node_stockpile)][
                            other_direction].discard(end_node.index)

    def check_common(self,id_machine,end_node_row):
        other_id = -1 # no common node

        if id_machine == 0 or id_machine == 2:
            if 1 in self.machine_idx:
                other_id = 1

        elif id_machine == 1:
            if end_node_row == 2:
                if 0 in self.machine_idx:
                    other_id = 0
            elif end_node_row == 3:
                if 2 in self.machine_idx:
                    other_id = 2

        return other_id

    def generate_edges(self):
        try:
            for idm in np.arange(self.problem.number_machine):
                temp = self.machine[idm].solution_path_id
                for i,e in enumerate(temp):
                    if i+1<len(temp):
                        edge = self.problem.D_edge[(temp[i][0],temp[i+1][0],temp[i][1],temp[i+1][1])]
                        # self.edges[idm].add(edge)
                        self.edges.add(edge)
        except:
            print('ERR')




    def release_machine(self, id_machine):
        ## reclaim

        end_id = self.queue['end_node'][id_machine].index
        end_direction = self.queue['end_direction'][id_machine]
        end_utility = self.queue['end_utility'][id_machine]
        end_id_demand = self.queue['end_demand'][id_machine]
        end_tonnage = np.float64(self.queue['end_node'][id_machine].cut_tonnage)

        # append

        self.machine[id_machine].solution[end_id_demand].solution_nodes.append(self.queue['end_node'][id_machine])
        self.solution[end_id_demand].solution_nodes.append(self.queue['end_node'][id_machine])

        self.machine[id_machine].solution[end_id_demand].solution_direction.append(end_direction)
        self.solution[end_id_demand].solution_direction.append(end_direction)

        self.machine[id_machine].solution[end_id_demand].solution_ids.append(end_id)
        self.solution[end_id_demand].solution_ids.append(end_id)
        self.visited.add(end_id)

        # Add time
        self.machine[id_machine].solution[end_id_demand].time_start.append(self.time_start[id_machine])
        self.machine[id_machine].solution[end_id_demand].time_finish.append(self.queue['end_completion'][id_machine])

        # Add tonnage
        self.solution[end_id_demand].tonnage_cut.append(end_tonnage)

        # self.machine[id_machine].solution[end_id_demand].solution_time.append(self.time_current)
        # self.solution[end_id_demand].solution_time.append(self.time_current)

        # end_penalty parcel calculation after appending
        end_penalty_parcel = self.calc_penalty_parcel_release(end_node_chemical=self.queue['end_node'][id_machine].chemical, id_demand=end_id_demand)

        # update chemical : change the average, update the array ::
        len_solution = len(self.solution[end_id_demand])

        if len_solution == 1:
            added_value_avg = self.queue['end_node'][id_machine].chemical

            # penalty_average depends on the demand
            # self.machine[end_id_demand].solution[end_id_demand].penalty_average = added_value_avg
            # self.machine[id_machine].solution[end_id_demand].nodes_chemical = self.queue[id_machine]['end_node'].chemical

            self.solution[end_id_demand].penalty_average += added_value_avg
            self.solution[end_id_demand].nodes_chemical += added_value_avg

            self.solution[end_id_demand].nodes_penalty_avg += added_value_avg

        else:
            #update penalty avg

            added_value_avg = np.divide(
                self.queue['end_node'][id_machine].chemical - self.solution[end_id_demand].penalty_average,
                len_solution) # not len_solution+1 because above we have already appended the cut to be reclaimed and release its serving machine

            self.solution[end_id_demand].penalty_average += added_value_avg  # add an value to an existing average


            # keep chemical of all nodes
            self.solution[end_id_demand].nodes_chemical = np.vstack(
                [self.solution[end_id_demand].nodes_chemical, self.queue['end_node'][id_machine].chemical])

            # keep penalty avg at every node
            self.solution[end_id_demand].nodes_penalty_avg = np.vstack(
            [self.solution[end_id_demand].nodes_penalty_avg, self.solution[end_id_demand].penalty_average]) # after placement of the node occured

        # update utility
        self.machine[id_machine].solution[end_id_demand].utility += end_utility
        self.machine[id_machine].solution[end_id_demand].nodes_utility = np.append(
            self.machine[id_machine].solution[end_id_demand].nodes_utility, end_utility)

        # self.machine[id_machine].solution[end_id_demand].nodes_utility.append(end_utility)

        self.solution[end_id_demand].utility += end_utility
        self.solution[end_id_demand].nodes_utility = np.append(self.solution[end_id_demand].nodes_utility, end_utility)
        # self.solution[end_id_demand].nodes_utility.append(end_utility)




        # update tonnage
        self.machine[id_machine].solution[end_id_demand].tonnage += end_tonnage
        self.solution[end_id_demand].tonnage += end_tonnage

        # update path
        self.machine[id_machine].solution[end_id_demand].solution_path.append(
            (self.queue['end_node'][id_machine], end_direction))

        self.machine[id_machine].solution_path_id.append((end_id,end_direction))

        # for evaluating the "current" position for a each machine
        self.machine[id_machine].solution_path.append((self.queue['end_node'][id_machine], end_direction))

        # update penalty parcel
        if self.cond_parcel(end_id_demand):
            self.solution[end_id_demand].nodes_viol_parcel = np.hstack(
                [self.solution[end_id_demand].nodes_viol_parcel, end_penalty_parcel])

        # data export save
        self.solution[end_id_demand].data_export['time'].append(self.time_current)
        self.solution[end_id_demand].data_export['machine'].append(id_machine)
        self.solution[end_id_demand].data_export['dir'].append(end_direction)
        self.solution[end_id_demand].data_export['cut_id'].append(end_id)

        self.machine[id_machine].solution[end_id_demand].data_export['time'].append(self.time_current)
        self.machine[id_machine].solution[end_id_demand].data_export['machine'].append(id_machine)
        self.machine[id_machine].solution[end_id_demand].data_export['dir'].append(end_direction)
        self.machine[id_machine].solution[end_id_demand].data_export['cut_id'].append(end_id)
        # self.machine[id_machine].solution[end_id_demand].data_export['cut'].append(self.queue['end_node'][id_machine])



        # reset
        self.stockpile_busy[id_machine] = ()

        for k in self.queue_keys_cat:
            self.queue[k][id_machine] = None

        for k in self.queue_keys_num:
            self.queue[k][id_machine] = -1

        # self.queue[id_machine] = {}
        self.tonnage_queue[self.machine[id_machine].machine_demand, id_machine] = np.float(0)

        # check increasing the number of demand
        if self.machine[id_machine].machine_demand < self.active_demand:
            self.machine[id_machine].machine_demand = self.active_demand

        # update machine id demands
        if self.machine[id_machine].machine_demand < self.active_demand:
            self.machine[id_machine].machine_demand = max(self.machine[id_machine].machine_demand, self.active_demand)

        # update objective function


    def find_neighbor_after_reclaim(self, direction_reclaimer, id_machine):
        node_current = self.machine[id_machine].current[0]
        neighbor_list = node_current.prec_1[direction_reclaimer]
        neighbor_type = node_current.product
        # nodes that are succeeded by the current node

        for neighbor_id in neighbor_list:
            neighbor = self.problem.D_nodes[neighbor_id]
            if neighbor_id not in self and neighbor_id not in self.machine[id_machine].active_cuts[neighbor_type][
                (node_current.node_row, node_current.node_stockpile)][direction_reclaimer]:
                prec = set(
                    neighbor.prec[direction_reclaimer])  # what are precedences of found nodes in neighborhood :: set
                if len(self.visited.intersection(prec)) == len(prec):  # all(node in self for node in prec)
                    ## this node should be added
                    self.machine[id_machine].active_cuts[neighbor_type][
                        (node_current.node_row, node_current.node_stockpile)][direction_reclaimer].add(neighbor_id)

                    if neighbor.node_common:# it is a common node
                        other_id = self.check_common(id_machine=id_machine, end_node_row=neighbor.node_row)

                        if other_id > 0:

                            if neighbor_id not in self.machine[other_id].active_cuts[neighbor_type][
                                (neighbor.node_row, neighbor.node_stockpile)][direction_reclaimer]:
                                self.machine[other_id].active_cuts[neighbor_type][
                                    (node_current.node_row, node_current.node_stockpile)][direction_reclaimer].add(
                                    neighbor.node_alias)
                                # it should be "add" instead of "append" :: intentional error

    def validate(self, test_case):

        if test_case == 1:
            # this function tests the validation of results
            # check utility
            temp = {req_k :{idm: [] for idm in self.machine_idx} for req_k in np.arange(self.problem.number_demand)}
            for req in np.arange(self.problem.number_demand):
                for idm in np.arange(self.problem.number_machine):

                    if req == 0:
                        initial_cut_tuple = self.machine[idm].solution[req].solution_path[0]
                        utility_start = np.divide(initial_cut_tuple[0].cost_reclaim, initial_cut_tuple[0].cut_tonnage)
                    else:
                        initial_cut_tuple = self.machine[idm].solution[req].solution_path[0]
                        cut_start = self.machine[idm].solution[req-1].solution_path[-1][0]
                        cut_start_direction = self.machine[idm].solution[req-1].solution_path[-1][1]
                        cost_moving = np.float(
                            self.problem.D_cost[cut_start.index, initial_cut_tuple[0].index, cut_start_direction, initial_cut_tuple[1]])
                        cost_reclaim = np.float(initial_cut_tuple[0].cost_reclaim)
                        cost = cost_reclaim + cost_moving
                        utility_start = np.round(np.divide(cost, initial_cut_tuple[0].cut_tonnage), self.problem.precision)

                    temp[req][idm].append(np.round(utility_start,self.problem.precision))

                    for i, e in enumerate(self.machine[idm].solution[req].solution_path):
                        if i < len(self.machine[idm].solution[req]) - 1:
                            cut_start = self.machine[idm].solution[req].solution_path[i][0]
                            cut_start_direction = self.machine[idm].solution[req].solution_path[i][1]
                            cut_end = self.machine[idm].solution[req].solution_path[i + 1][0]
                            cut_end_direction = self.machine[idm].solution[req].solution_path[i + 1][1]
                            cost_moving = np.float(
                                self.problem.D_cost[cut_start.index, cut_end.index, cut_start_direction, cut_end_direction])
                            cost_reclaim = np.float(cut_end.cost_reclaim)
                            cost = cost_reclaim + cost_moving
                            utility = np.round(np.divide(cost, cut_end.cut_tonnage), self.problem.precision)
                            temp[req][idm].append(utility)

                    # print('all utility for machine {}: {}, real:'.format(idm, self.machine[idm].solution[req].nodes_utility))
                    print('total utility for machine {}: {}, real: {}'.format(idm, sum(self.machine[idm].solution[req].nodes_utility), sum(temp[req][idm])))

        if test_case == 2:
            # check penalty avg viol
            temp = np.zeros(self.problem.number_demand)

            for req in np.arange(self.problem.number_demand):
                temp[req] = self.calc_viol(avg=self.solution[req].nodes_chemical.mean(axis=0).reshape(1,6), case='avg', id_demand=req)

                print('total penalty avg for solution {}: {}, real: {}'.format(req, self.calc_viol(avg=self.solution[req].penalty_average.reshape(1,6), case='avg', id_demand=req),
                                                                        temp[req]))

        if test_case == 3:
            # check penalty parcel viol
            temp = {req_k :[] for req_k in np.arange(self.problem.number_demand)}
            for req in np.arange(self.problem.number_demand):
                for i,e in enumerate(self.solution[req].nodes_chemical):
                    if i >= 3:
                        last_three_chemical = self.solution[req].nodes_chemical[i-3:i]
                        parcel_avg = last_three_chemical.mean(axis=0)
                        temp[req].append(self.calc_viol(avg=parcel_avg.reshape(1,6), case='parcel', id_demand=req))

                print('total penalty parcel for solution {}: {}, real: {}'.format(req, np.round(sum(self.solution[req].nodes_viol_parcel),self.problem.precision), sum(temp[req])))

    def export(self):
        str_instance = '({},{},{},{})'.format(self.problem.number_machine, self.problem.number_demand,
                                        int(self.problem.parcel_input), self.problem.direction)
        # export to csv

        if self.greedy_param > 0:
            str_alg = '({},{})'.format(self.greedy_param, self.SP)
            instance_save = './rga_data/{}_RGA_{}_seed_{}.csv'.format(str_instance, str_alg, abs(self.problem.seed_record))
        else:
            instance_save = './rga_data/{}_DGA.csv'.format(str_instance)

        DF = []
        for id_demand in np.arange(self.problem.number_demand):
            size = len(self.solution[id_demand].data_export['time'])
            self.solution[id_demand].data_export['req']= np.ones(size)*id_demand
            self.solution[id_demand].data_export['type'] = [self.problem.demand_type[id_demand]]*size
            DF.append(pd.DataFrame(self.solution[id_demand].data_export))

        df = pd.concat(DF)
        df = df.sort_values(by='time')
        df.to_csv(instance_save,index=False)

    # def save_solution(self):
        # self.final_solution = {k: {demand: {'solution_ids': self.machine[k].solution[demand].solution_ids,
        #                                     'time_ids': self.machine[k].solution[demand].solution_time} for demand in
        #                            np.arange(self.problem.number_demand) + 1} for k in self.machine_idx}

    def export_best_solution(self):
        str_instance = '({},{})'.format(self.problem.number_machine, self.problem.number_demand,
                                              int(self.problem.parcel_input), self.problem.direction)

        str_alg = '({},{})'.format(self.greedy_param, self.SP)

        instance_save = './rga_data/{}_RGA_{}_seed_{}'.format(str_instance, str_alg, abs(self.problem.seed_record))

        self.compressed_save_variable(self, instance_save)

    def export_best_solution_simple(self):
        str_instance = '({},{})'.format(self.problem.number_machine, self.problem.number_demand,
                                        int(self.problem.parcel_input), self.problem.direction)

        str_alg = '({},{})'.format(self.greedy_param, self.SP)

        instance_save = './rga_data/{}_RGA_{}_seed_{}.pickle'.format(str_instance, str_alg, abs(self.problem.seed_record))

        self.save_variable(self.solution, instance_save)

    def save_variable(self, variable, string):
        with open(string, 'wb') as pickle_str:
            pickle.dump(variable, pickle_str)

    def load_variable(self, string):
        return pickle.load(open(string, "rb"))

    def compressed_save_variable(self, variable, string):
        with bz2.BZ2File(string + '.pbz2', 'w') as f:
            cPickle.dump(variable, f)

    def decompress_load_variable(self, string):
        data = bz2.BZ2File(string, 'rb')
        data = cPickle.load(data)
        return data

