from src.utils import pickle, np

class Problem:
    def __init__(self, number_demand : int, number_machine : int, parcel : bool, local : str, method: str, direction: int):
        """

        :param scenario: number of scenari
        :param number_demand: number of demands
        :param number_stockpile: number of stockpiles
        :param greedy_factor: greedy factor for RGA
        :param parcel: scenario 3 is activated or not
        """
        if method == 'DGA':
            self.type_algorithm = 'DGA'
        elif method == 'RGA':
            self.type_algorithm = 'RGA'
        elif method == 'MMAS':
            self.type_algorithm = 'MMAS'
            
        self.direction = direction # 1 means only SN , 2 means both SN-NS

        self.number_demand = number_demand
        self.number_machine = number_machine

        self.demand_capacity = np.array([200,100,100,200,150,200,100,200,200,100,100,200])*1000
        self.demand_type = np.asarray(['F','F','F','L','L','F','F','F','F','F','F','L'])

        if self.number_machine == 2:
            self.demand_capacity = self.demand_capacity[:11]
            self.demand_type = self.demand_type[:11]

        self.type_stockpile = {'F':[(1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,1), (3,2), (4,2), (4,4)],
                               'L': [(2,1), (3,3), (3,4), (4,1), (4,3)]}


        self.machine_rows = {0: np.asarray([1,2]), 1:np.asarray([2,3]), 2:np.asarray([3,4])} # machine : rows

        self.machine_rows_common = {2: [0,1], 3: [1,2]} # row: machine

        self.parcel_input = bool(parcel)

        self.size_objective = np.array([2,3]) # if parcel false: 2 | if parcel true : 3

        self.local = local
        self.precision = 4
                      
        
        if number_machine < 2 and number_machine > 3:
            print('number machine is in wrong range')
            
        pickle.load(open('src/eka_testproblem/D_nodes_{}_machines.pickle'.format(self.number_machine),'rb'))

        self.D_nodes = pickle.load(open('src/eka_testproblem/D_nodes_{}_machines.pickle'.format(self.number_machine),'rb')) # only one stockpile
        
        if self.direction == 2:
            self.D_cost = pickle.load(open('src/eka_testproblem/D_cost_{}_machines.pickle'.format(self.number_machine),'rb'))
        elif self.direction == 1:
            self.D_cost = pickle.load(open('src/eka_testproblem/D_cost_{}_SN_machines.pickle'.format(self.number_machine),'rb'))

        if self.direction == 2:
            self.D_edge = pickle.load(open('src/eka_testproblem/D_edge_{}_machines.pickle'.format(self.number_machine),'rb')) # only one stockpile
        elif self.direction == 1:
            self.D_edge = pickle.load(open('src/eka_testproblem/D_edge_{}_SN_machines.pickle'.format(self.number_machine),'rb')) # only one stockpile

        # mineral limits :: ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']

        self.limits_upper_solution = {'F':np.asarray([2.3886, 100, 0.2191, 0.106, 0.0311, 4.0246]), 'L':np.asarray([1.662, 100.000, 0.221, 0.093, 0.028, 3.372])}
        self.limits_lower_solution = {'F':np.asarray([0, 61.165, 0, 0, 0, 0]), 'L':np.asarray([0, 61.881, 0, 0, 0, 0])}
        self.limits_upper_window = {'F':np.asarray([2.4677, 100, 0.2686, 0.109, 0.03615, 4.1777]), 'L':np.asarray([1.743, 100.000, 0.271, 0.096, 0.033, 3.519])}
        self.limits_lower_window = {'F':np.asarray([0, 60.9636, 0, 0, 0, 0]), 'L':np.asarray([0, 601.682, 0, 0, 0, 0])}

        self.initial_cut = {'F': {'SN': {0: '01-01-01-01', 1: '03-01-01-01', 2: '04-02-01-01'},
                                  'NS': {0: '01-01-01-10', 1: '03-01-01-10', 2: '04-02-01-10'}
                                  },
                            'L': {'SN': {0: '02-01-01-01', 1: '03-03-01-01', 2: '04-01-01-01'},
                                  'NS': {0: '02-01-01-01', 1: '03-03-01-01', 2: '04-01-01-01'}
                                  }
                            }


        # print('problem initialisation : successful')

    def initialise_seed(self, seed):

        if seed >= 0:
            self.rng = np.random.default_rng(seed)
            # print('seed initialisation {} : successful'.format(seed))
            self.seed_record = seed
        elif seed < 0:
            raise Exception('this seed has not been considered in preseed allocation')


    def find_type_cut(self,node_row,node_stockpile):

        if (node_row, node_stockpile) in self.type_stockpile['F']:
            return 'F'
        else:
            return 'L'


    def my_lex_sort(self, obj_neighbor):
        if self.parcel_input:
            sorted_idx = np.lexsort(
                (obj_neighbor[:, 0], obj_neighbor[:, 2], obj_neighbor[:, 1]))  # utility, penalty_ parcel, penalty_avg
        else:
            sorted_idx = np.lexsort((obj_neighbor[:, 0], obj_neighbor[:, 1]))

        return sorted_idx