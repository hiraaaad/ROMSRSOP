import argparse
import os
import sys
import pickle
import time
from utils import np
from problem import Problem
from controller import Parallel_Controller
from MMAS_solver import MMAS_Solver
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        choices=['DGA', 'RGA', 'MMAS'],
        default='DGA',
        type=str,
        required=True,
        help='algorithm')
    parser.add_argument(
        '--sp',
        default=1.5,
        type=float,
        required=False,
        help='Selection pressure')
    parser.add_argument(
        '--greedy-param',
        default=1,
        type=int,
        required=False,
        help='Greedy paramter for RGA')
    parser.add_argument(
        '--ls',
        choices=['none', '1'],
        default='none',
        type=str,
        required=False,
        help='Local search method')
    parser.add_argument(
        '--rho',
        default=0.5,
        type=float,
        required=False,
        help='rho of ACO')
    parser.add_argument(
        '--n-ants',
        default=10,
        type=int,
        required=False,
        help='number of ants for population size')
    parser.add_argument(
        '--alpha',
        default=1,
        type=int,
        required=False,
        help='alpha for ACO')
    parser.add_argument(
        '--beta',
        default=2,
        type=int,
        required=False,
        help='beta for ACO')
    parser.add_argument(
        '--tau0',
        default=0.5,
        type=float,
        required=False,
        help='initial value of pheromone for ACO')
    parser.add_argument(
        '--seed',
        default=0,
        type=int,
        required=False,
        help='Seed for the random number generator')
    parser.add_argument(
        '--maxiter',
        default=1000,
        type=int,
        required=False,
        help='Maximum number of iterations')
    parser.add_argument(
        '--n-machine',
        choices=[2, 3],
        default=2,
        type=int,
        required=True,
        help='number of machines')
    parser.add_argument(
        '--n-demand',
        choices=[2,3,4,5,6,7,8,9,10,11,12,13],
        default=2,
        type=int,
        required=True,
        help='number of demands')
    parser.add_argument(
        '--parcel',
        choices=[True,False],
        default=False,
        type=bool,
        required=True,
        help='number of demands')
    parser.add_argument(
        '--dir',
        choices=[1,2],
        default=1,
        type=int,
        required=True,
        help='type of directions')
    parser.add_argument(
        '--tba',
        choices=['BSFA','IBA'],
        type=str,
        required=False,
        help='type best algorithm for pheromone update')
    parser.add_argument(
        '--batch',
        choices=[0,1,2],
        default=2,
        type=int,
        required=True,
        help='size of batch of run for RGA')
        # if 0 : it takes the seed from input
        # if 2 : it generates a random seed
        # if 1 : it goes for 31 runs.


    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        import sys
        sys.exit(0)
    return args

if __name__ == '__main__':
    kwargs = {}
    args = parse_arguments()


    problem = Problem(number_machine = args.n_machine, number_demand = args.n_demand, parcel = args.parcel,
                                        local = args.ls, method=args.method, direction=args.dir)

    str_instance = '({},{},{},{})'.format(args.n_machine, args.n_demand, int(args.parcel), args.dir)
    data_batch = []

    # mkdir
    Path('./batch_data').mkdir(parents=True, exist_ok=True)

    # run

    if args.method == 'DGA':
        solver = Parallel_Controller(problem=problem, greedy_param = 0, selection_pressure=args.sp, ant_kw=None)
        solver.run()
        data_batch.append(solver.total_obj)
        solver.export()


    elif args.method == 'RGA':
        Path('./rga_data').mkdir(parents=True, exist_ok=True)

        str_alg = '({},{})'.format(args.greedy_param, args.sp)
        if args.batch == 1:
            batch_range = np.arange(31)
        else:
            batch_range = np.arange(1)

        for idx_batch in batch_range:
            solver = Parallel_Controller(problem=problem, greedy_param = args.greedy_param, selection_pressure=args.sp, ant_kw=None)

            if args.batch == 0:
                solver.problem.initialise_seed(seed=-args.seed)
            elif args.batch == 1:
                solver.problem.initialise_seed(seed=-idx_batch)
            elif args.batch == 2:
                # generate a seed
                seed = np.random.randint(1e9)
                solver.problem.initialise_seed(seed=seed)



            solver.run()
            print('Best-solution: {}'.format(solver.total_obj))
            data_batch.append(solver.total_obj)
            str_alg = '({},{})'.format(args.greedy_param, args.sp)


            # solver.export_best_solution_simple()
            # solver.export()



    elif args.method == 'MMAS':
        Path('./mmas_data').mkdir(parents=True, exist_ok=True)
        print('batch {} is running'.format(-args.seed))
        solver = MMAS_Solver(problem=problem, alpha=args.alpha, beta= args.beta, rho = args.rho, tau0 = args.tau0,
                             population_size= args.n_ants, iteration_max = args.maxiter, selection_pressure=args.sp,
                             type_best_ant=args.tba, local_search=args.ls)
        solver.problem.initialise_seed(seed=-args.seed)
        solver.run()
        print('Best-solution: {}'.format(solver.globalBest))
        data_batch.append(solver.globalBest)
        str_alg = '({},{},{},{},{},{},{})'.format(args.n_ants, args.maxiter, args.alpha, args.beta, args.rho, args.tau0, args.sp)

    if args.method =='DGA':
        instance_save = './batch_data/{}_DGA.csv'.format(str_instance)
    elif args.method == 'RGA':
        if args.batch == 1:
            instance_save = './batch_data/{}_RGA_{}.csv'.format(str_instance, str_alg)
        else:
            instance_save = './batch_data/{}_RGA_{}_seed_{}.csv'.format(str_instance, str_alg, abs(solver.problem.seed_record))
    elif args.method == 'MMAS':
        instance_save = './batch_data/{}_MMAS_{}_seed_{}.csv'.format(str_instance, str_alg, abs(solver.problem.seed_record))


    np.savetxt(instance_save,data_batch,delimiter=',',fmt='%10.5f')
    print('csv save: succssful')



