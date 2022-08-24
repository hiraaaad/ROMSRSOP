from src.utils import np, Solution, pickle, bz2, cPickle
from src.utils_local import LocalSearch, LocalSolution
from src.ant import Ant
from pathlib import Path


class MMAS_Solver():
    def __init__(self, problem, alpha, beta, rho, tau0, selection_pressure,
                 population_size,iteration_max, type_best_ant, local_search):
        self.problem = problem
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.sp = selection_pressure
        self.size = len(self.problem.D_edge.keys())
        self.tau_min, self.tau_max = 1/self.size, 1-(1/self.size)
        self.iteration_max = iteration_max
        self.type_best_ant = type_best_ant
        self.best_ant_index = None
        self.globalBestAnt = None
        self.globalBest = np.full((self.problem.size_objective[int(self.problem.parcel_input)],),np.inf)
        self.population_size = population_size
        self.tau0 = tau0
        self.init_pheromone()
        self.init_population()
        self.ls = local_search
        Path('./raw_mmas_local').mkdir(parents=True, exist_ok=True)
        Path('./raw_mmas').mkdir(parents=True, exist_ok=True)
        print('MMAS algorithm initialisation : succesful')


    def init_population(self):
        self.population = [Ant(problem=self.problem, alpha=self.alpha, beta=self.beta, rho=self.rho, tau=self.tau,
                               greedy_param = -1, selection_pressure=self.sp)
                           for _ in range(self.population_size)]

    def run_population(self):
        for ant in self.population:
            ant.constructSolution()
            # print(self.problem.rng.random(4))

    def init_pheromone(self):
    # https://www.sciencedirect.com/science/article/pii/S0167739X00000431: the larger exploration of the search space
    # due to setting τ(1)=τmax improves its performance
        if self.tau0 == 0:
            self.tau = np.full((self.size,),self.tau_max, dtype=np.float32)
        else:
            self.tau = np.full((self.size,),0.5, dtype=np.float32)

        # self.tau = np.full((self.size,),self.tau_max, dtype=np.float32)

    def update_pheromone(self, best_ant):
        # https://cs.adelaide.edu.au/~markus/pub/2011foga-ants.pdf
        self.tau *= (1-self.rho) # decay pheromone
        best_ant.sim.generate_edges()
        included_edges = np.array(list(best_ant.sim.edges),dtype=np.int)
        # add pheromone trail to included edges
        self.tau[included_edges] += self.rho
        self.tau = np.clip(self.tau, self.tau_min, self.tau_max)

    def find_best_ant(self):
        obj_population = np.array([ant.sim.total_obj for ant in self.population])
        lex_sort = self.problem.my_lex_sort(obj_population)
        return (lex_sort[0], obj_population[0])

    def improvement_global_best(self, best_ant_iteration):
        temp = self.problem.my_lex_sort(np.array([self.globalBest, best_ant_iteration.sim.total_obj]))
        improvement = False
        if temp[0] == 1:
            improvement = True

        return improvement

    def update_global_best(self, best_ant_iteration):
        # globalBest should be updated
            self.globalBest = best_ant_iteration.sim.total_obj
            self.globalBestAnt = best_ant_iteration

    def apply_local_search(self, best_ant):
        best_ant_sim = best_ant.sim
        localsolution = LocalSolution(solver=best_ant_sim)
        localsearch = LocalSearch(localsolution=localsolution)
        for id_solution in range(localsolution.problem.number_demand):
            for id_machine in range(localsolution.problem.number_machine):
                localsolution = localsearch.swap_iterative(localsolution=localsolution, id_machine=id_machine, id_solution=id_solution)

        localsolution.calc_total_obj()
        return localsolution

    def assign_to_ant(self, localsolution):
        ant = Ant(problem=self.problem, alpha=self.alpha, beta=self.beta, rho=self.rho, tau=self.tau,
            greedy_param = -1, selection_pressure=self.sp)

        ant.sim.machine = localsolution.machine
        ant.sim.solution = localsolution.solution
        ant.sim.total_obj = localsolution.total_obj

        return ant





    def run(self):

        # if self.ls == 0:
        str_save_mmas = './raw_mmas/MMAS_({}-{}-{})_seed_{}.csv'.format(self.problem.number_demand,
                                                                        self.problem.number_machine,
                                                                        self.problem.direction,
                                                                        self.problem.seed_record)
        #     mod_iteration = 5
        # else:
        str_save_mmas_local = './raw_mmas_local/MMAS_local_({}-{}-{})_seed_{}.csv'.format(self.problem.number_demand,
                                                                                          self.problem.number_machine,
                                                                                          self.problem.direction,
                                                                                          self.problem.seed_record)
        #     mod_iteration = 10

        for iteration in range(self.iteration_max):
            self.init_population()
            self.run_population()

            # find best ant in current iteration
            (index_best_ant, obj_best_ant) = self.find_best_ant()

            if self.ls == 1:
                localsolution = self.apply_local_search(best_ant=self.population[index_best_ant])
                best_ant_iteration = self.assign_to_ant(localsolution=localsolution)
                improvement = self.improvement_global_best(best_ant_iteration=best_ant_iteration)
            else:
                improvement = self.improvement_global_best(best_ant_iteration=self.population[index_best_ant])

            if improvement:
                if self.ls == 1:
                    self.update_global_best(best_ant_iteration=best_ant_iteration)
                else:
                    self.update_global_best(best_ant_iteration=self.population[index_best_ant])

                # np.savetxt(str_save_mmas, self.globalBest, delimiter=',', fmt='%10.5f')
                print('Improvement: Generation: {:d} - Best-solution: {}'.format(iteration,self.globalBest))

            if self.type_best_ant == 'IBA':
                # it is the best one in current iteration
                self.update_pheromone(self.population[index_best_ant])
            elif self.type_best_ant == 'BSFA':
                self.update_pheromone(self.globalBestAnt)
            else:
                raise Exception

            # print('Generation: {}'.format(iteration))

        # self.export_best_global()
        self.total_obj = self.globalBest

        localsolution = self.apply_local_search(best_ant=self.globalBestAnt)
        print(localsolution.total_obj)
        best_ant_iteration = self.assign_to_ant(localsolution=localsolution)
        self.update_global_best(best_ant_iteration=best_ant_iteration)
        np.savetxt(str_save_mmas_local, self.globalBest, delimiter=',', fmt='%10.5f')

    def export_best_global(self):
        str_instance = '({},{},{},{})'.format(self.problem.number_machine, self.problem.number_demand,
                                                  int(self.problem.parcel_input), self.problem.direction)

        str_alg = '({},{},{},{},{},{},{})'.format(self.population_size, self.iteration_max, self.alpha, self.beta,
                                                  self.rho, self.tau0, self.sp)

        instance_save = './mmas_data/{}_MMAS_{}_seed_{}'.format(str_instance, str_alg, abs(self.problem.seed_record))

        self.compressed_save_variable(self.globalBestAnt, instance_save)


    def compressed_save_variable(self, variable, string):
        with bz2.BZ2File(string + '.pbz2', 'w') as f:
            cPickle.dump(variable, f)

    def decompress_load_variable(self, string):
        data = bz2.BZ2File(string, 'rb')
        data = cPickle.load(data)
        return data






