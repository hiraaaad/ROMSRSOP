import argparse
import os
import sys
import pickle
import time
import pandas as pd
import re
from src.utils import np
from src.problem import Problem
from src.controller import Parallel_Controller
from src.MMAS_solver import MMAS_Solver
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        choices=['DGA', 'RGA', 'MMAS'],
        default='MMAS',
        type=str,
        required=True,
        help='algorithm')
    parser.add_argument(
        '--idx',
        type=int,
        required=True,
        help='Index of slurm')

        '--instance',
        type=str,
        required=True,
        help='instance to apply the algorithm to it')
    # parser.add_argument(
        # '--batch',
        # # choices=[101],
        # # default=31,
        # type=int,
        # required=True,
        # help='size of batch of run for RGA')
        # if 0 : it takes the seed from input
        # if 2 : it generates a random seed
        # if 1 : it goes for 31 runs.
    parser.add_argument(
        '--ls',
        choices=['none', '1'],
        default='none',
        type=str,
        required=False,
        help='Local search method')
    parser.add_argument(
        '--n-ants',
        default=10,
        type=int,
        required=False,
        help='number of ants for population size')
    parser.add_argument(
        '--maxiter',
        default=1000,
        type=int,
        required=False,
        help='Maximum number of iterations')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        import sys
        sys.exit(0)
    return args

def parse_instance(instance_name):
    find_all = np.asarray(re.findall(r'\b\d+\b', instance_name),dtype=int)
    return tuple(find_all[-3:])

if __name__ == '__main__':
    df = pd.read_csv('mmas-irace-configs-found.csv')
    kwargs = {}
    # Parse command line arguments
    args = parse_arguments()
    (n_demand, n_machine, direction) = parse_instance(args.instance)
    conf_params = df.loc[df['Instance'] == '({}, {}, {})'.format(n_demand, n_machine, direction)]


    problem = Problem(number_machine = n_machine, number_demand = n_demand, parcel = True,
                                        local = 'none', method='MMAS', direction= direction)

    # str_instance = '({},{},{},{})'.format(args.n_machine, args.n_demand, int(args.parcel), args.dir)
    obj_batch = []

    # mkdir
    Path('./raw_mmas').mkdir(parents=True, exist_ok=True)
    Path('./raw_mmas_local').mkdir(parents=True, exist_ok=True)

    # run

    if args.method == 'MMAS':
        Path('./mmas_data').mkdir(parents=True, exist_ok=True)

        # str_alg = '({},{})'.format(args.greedy_param, args.sp)
        # if args.batch == 101:
        #     batch_range = np.arange(101)
        # else:
        #     batch_range = np.arange(1)

        # seeds for replication
        preseed = np.array([226024, 894631, 118599, 802361, 23414, 976405, 798742, 647772, 82428, 566941
                               , 175144, 435676, 331388, 428582, 873627, 41918, 7806, 562734, 523424, 609150
                               , 93564, 209194, 220472, 63488, 570335, 153744, 543934, 625362, 84325, 636283
                               , 464398, 529193, 318544, 205037, 852066, 988015, 15880, 665647, 658019, 690671
                               , 362619, 803845, 868070, 394902, 161626, 636900, 332690, 442120, 113993, 276401
                               , 942972, 134143, 137052, 921830, 727872, 61800, 943104, 108918, 233229, 936444
                               , 689071, 862780, 944836, 552032, 357025, 92066, 869317, 216829, 493700, 51734
                               , 691270, 146044, 728563, 471856, 132138, 736886, 77208, 443348, 224069, 656098
                               , 990195, 516716, 854011, 698891, 184790, 161487, 336484, 22868, 246949, 410368
                               , 194817, 318576, 98816, 312131, 22585, 889346, 900289, 789335, 25676, 591257
                               , 839707])

        # for idx_batch in batch_range:
            # solver = Parallel_Controller(problem=problem, greedy_param = args.greedy_param, selection_pressure=args.sp, ant_kw=None)
        if args.ls == 'none':
            local_search = 0
        else:
            local_search = 1

        # batch_range = np.arange(int(args.batch))

        if int(conf_params['bestant'])==0:
            tba = 'BSFA'
        else:
            tba = 'IBA'

        # print(conf_params)
        solver = MMAS_Solver(problem=problem,
                             alpha=int(conf_params['alpha']),
                             beta=int(conf_params['beta']),
                             rho=float(conf_params['rho']),
                             tau0=1,
                             population_size=args.n_ants,
                             iteration_max=args.maxiter,
                             selection_pressure=float(conf_params['sp']),
                             type_best_ant=tba,
                             local_search=local_search)

        print(conf_params)
        try:
            seed = preseed[args.idx]
            # print(seed)
            solver.problem.initialise_seed(seed=seed)
        except:
        # else:
            raise Exception('Problem in seed initialisation')

        solver.run()
        print('Best-solution: {}'.format(solver.total_obj))
        obj_batch.append(solver.total_obj)
            # str_alg = '({},{})'.format(args.greedy_param, args.sp)
            # solver.export_best_solution_simple()
            # solver.export()




