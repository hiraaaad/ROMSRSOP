import numpy as np
import pandas as pd
import itertools as it
import bz2
import pickle
import _pickle as cPickle
np.set_printoptions(suppress=True, precision=4)

class Node():
    def __repr__(self):
        return f'Cut(id={self.index})'

    def __init__(self, row, node_prec_SN, node_prec_NS, node_prec_1_SN, node_prec_1_NS,node_machine , node_row, node_stockpile , node_bench, node_cut, node_common, node_alias):
        self.index = row.Index[3:]
        self.product = row.Product_Description # type of product description
        # for easier handling we remove the chemical names but through the code we consider their corresponding lower and upper bounds :: ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
        self.chemical = np.asarray([row.Al2O3, row.Fe, row.Mn, row.P, row.S, row.SiO2])
        self.cut_tonnage = np.float(row.Cut_Tonnage)
        self.cost_reclaim = np.float(row.Cost)
        self.prec = dict(zip(['SN','NS'],[node_prec_SN,node_prec_NS]))
        self.prec_1 = dict(zip(['SN','NS'],[node_prec_1_SN,node_prec_1_NS]))
        self.node_common = node_common
        self.node_alias = node_alias
        self.node_machine, self.node_row, self.node_stockpile, self.node_bench, self.node_cut = node_machine , node_row, node_stockpile , node_bench, node_cut

class Solution():

    def __repr__(self):
        return f'Solution(package={self.solution_number})'

    def __init__(self,number_demand):
        self.solution_number = number_demand
        self.solution_nodes = [] # all nodes in a solution
        self.solution_direction = [] # all direction in a solution
        self.penalty_average = np.zeros(6) # ['Al2O3', 'Fe', 'Mn', 'P', 'S', 'SiO2']
        self.nodes_chemical = np.array(np.zeros(6))
        self.nodes_utility = []
        self.nodes_viol_parcel = np.array([])
        self.nodes_penalty_avg = np.array(np.zeros(6))
        # self.penalty_avg_final = np.float(0) # it shows the final penalty avg for the whole package
        self.tonnage = np.float(0)
        self.utility = np.float(0)
        self.nodes_tonnage = np.array([])
        self.solution_path = []
        self.solution_ids = []
        self.time_start = []
        self.time_finish = []
        self.tonnage_cut = []
        self.data_export = {'time':[], 'machine':[], 'cut_id':[], 'dir':[]}

    def __contains__(self, node_id):
        return node_id in self.solution_ids

    def __len__(self):
        return len(self.solution_ids)


def parse_cut_id(cut_id: str) -> np.ndarray(shape=(1,5),dtype=int):
    cut_id_separated : list = cut_id.split("-")
    return np.asarray(cut_id_separated,dtype=np.int8)

def calc_pos_node(node_stockpile : int, node_bench : int, node_cut : int) -> tuple():
    x : int
    y : int
    x = -(node_bench-1)
    y = (10+2)*(node_stockpile-1) + node_cut-1
    return (x,y)


