from src.utils import np, Solution, it

class Machine():

    def __repr__(self):
        return f'Machine(idm={self.id_machine})'
    def __init__(self,problem,id_machine):
        self.problem = problem
        self.id_machine = id_machine
        self.solution = [Solution(k) for k in np.arange(self.problem.number_demand)]
        self.setup_active_cuts()
        self.solution_ids = []
        self.solution_time = [] # time that a solution got reclaimed
        self.solution_path = []
        self.solution_path_id = []
        self.common_neighbor_SN = ()
        self.common_neighbor_NS = ()
        self.machine_demand = 0 # which demand is being reclaimed for -- provided
        self.last_cut_demand = False


    def __len__(self):

        return len(self.solution[self.machine_demand].solution_nodes)

    def __contains__(self, node_id):
        return node_id in self.solution_ids


    @property
    def current(self):
        return self.solution_path[-1]

    def setup_active_cuts(self):
        """
        this function sets the entry points for the stockpiles wrt the direction for each machine
        :param number_stockpile:
        :return: setups a dictionary of active nodes in a stockyard which are the entry points at beginning
        """
        keys = it.product(self.problem.machine_rows[self.id_machine],np.arange(1,5))
        Keys = {k:[] for k in ['F','L']}

        for element in keys:
            if element in self.problem.type_stockpile['F']:
                Keys['F'].append(element)
            else:
                Keys['L'].append(element)

        if self.problem.direction == 1:
            self.active_cuts = {available_type:{K: dict(zip(['SN'],[{'0{}-0{}-01-01'.format(K[0], K[1])},
                                                                    {'0{}-0{}-01-10'.format(K[0], K[1])}])) for K in Keys[available_type]} for available_type in ['F','L']}
        elif self.problem.direction == 2:
            self.active_cuts = {available_type:{K: dict(zip(['SN','NS'],[{'0{}-0{}-01-01'.format(K[0], K[1])},
                                                                         {'0{}-0{}-01-10'.format(K[0], K[1])}])) for K in Keys[available_type]} for available_type in ['F','L']}


    def find_available_cuts(self,direction_reclaimer, stockpile_busy):
        # return active_neighborhood
        type_demand = self.problem.demand_type[self.machine_demand]
        neighbors_id = set() # we use set to avoid using "extend"
        for k,v in self.active_cuts[type_demand].items():
            # find stockpile_keys that are not busy (safety constraint) and they are the same type of product
            if k not in stockpile_busy.values():
                neighbors_id.update(v[direction_reclaimer])
        return neighbors_id

    def find_initial_cut(self,end_id_direction):
        # return initial active_neighborhood
        initial_cut = self.problem.initial_cut[self.problem.demand_type[0]][end_id_direction][self.id_machine]
        return initial_cut

    def find_active_neighborhood_initial(self):
        # if initial return objective of initial cuts, if run: return neighbor ids
        end_id_direction = 'SN'
        end_id = self.find_initial_cut(end_id_direction=end_id_direction) # reclaim global entry : '01-01-01-01'
        return end_id

    def find_active_neighborhood_run(self,stockpile_busy):

        neighbors_id_SN = self.find_available_cuts(direction_reclaimer='SN',stockpile_busy=stockpile_busy)
        if self.problem.direction == 2:
            neighbors_id_NS = self.find_available_cuts(direction_reclaimer='NS',stockpile_busy=stockpile_busy)
            return {'SN':neighbors_id_SN, 'NS':neighbors_id_NS}
        else:
            return {'SN':neighbors_id_SN}

    def calc_utility_initial(self, id_cut):

        end_id = id_cut[0]
        end_direction = id_cut[1]
        end_node = self.problem.D_nodes[end_id]
        cost_reclaim = np.float(end_node.cost_reclaim) # no moving cost for initial cut

        utility = np.divide(cost_reclaim,end_node.cut_tonnage)
        utility = np.round(utility,self.problem.precision)

        cost = np.round(cost_reclaim,self.problem.precision)

        return (utility,cost,end_node)


    def calc_utility(self, id_cut):
        """
        this function calculates the utility in reclamation for a node
        :param id_cut : a set containing all cuts including tuples of (id,direction)
        :param start_id: index id of the current position of reclaimer
        :param end_id: id of the node to be evaluated
        :param start_direction: current direction of the reclaimer
        :param end_direction: direction of reclaiming
        :return: the tuple of fitness for a node
        """
        # to evlauate fitness for each node in the neighborhood
        utility_all = np.zeros(len(id_cut))
        cost_all = np.zeros(len(id_cut))
        chemical_all = np.zeros([len(id_cut),6])
        end_node_all = []


        for index_cut, id in enumerate(id_cut):

            end_id = id[0]
            end_direction = id[1]
            end_node = self.problem.D_nodes[end_id]

            cost_moving = np.float(self.problem.D_cost[self.current[0].index,end_id,self.current[1],end_direction])

            cost_reclaim = np.float(end_node.cost_reclaim)

            cost = cost_reclaim + cost_moving
            cost_all[index_cut] = np.round(cost,self.problem.precision)

            utility = np.divide(cost,end_node.cut_tonnage)
            utility_all[index_cut] = np.round(utility,self.problem.precision)
            chemical_all[index_cut] = end_node.chemical

            end_node_all.append(end_node)

            # for reproducibility we should sort all to preserve order?

        return (utility_all,cost_all,end_node_all,chemical_all)