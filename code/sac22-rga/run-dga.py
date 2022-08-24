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
# from src.MMAS_solver import MMAS_Solver
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
        '--instance',
        type=str,
        required=True,
        help='instance to apply the algorithm to it')
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
    kwargs = {}
    # Parse command line arguments
    args = parse_arguments()
    (n_demand, n_machine, direction) = parse_instance(args.instance)


    problem = Problem(number_machine = n_machine, number_demand = n_demand, parcel = True,
                                        local = 'none', method='DGA', direction= direction)

    obj_batch = []

    # mkdir
    Path('./raw_dga').mkdir(parents=True, exist_ok=True)

    # run

    if args.method == 'DGA':
        solver = Parallel_Controller(problem=problem, greedy_param = 0, selection_pressure=0, ant_kw=None)
        solver.run()
        obj_batch.append(solver.total_obj)
        # solver.export()


    if args.method == 'DGA':
        str_save = './raw_dga/DGA_({}-{}-{}).csv'.format(n_demand, n_machine, direction)

    np.savetxt(str_save,obj_batch,delimiter=',',fmt='%10.5f')
    print('DGA {}: csv save: succssful'.format(args.instance))



