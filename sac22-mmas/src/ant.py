from src.utils import np
from src.controller import Parallel_Controller

class Ant():
    def __init__(self, problem, alpha, beta, rho, tau, greedy_param, selection_pressure):
        ant_kw = {'tau':tau, 'alpha':alpha, 'beta':beta, 'rho':rho}
        self.sim = Parallel_Controller(problem,greedy_param=greedy_param, selection_pressure=selection_pressure, ant_kw=ant_kw)

    def constructSolution(self):
        self.sim.run()




