'''
Hierarchical cost computation methods for single-label hierarchical
classification.
'''

import networkx as nx
import numpy as np
from scipy.stats import itemfreq


def compute_treecost_vector(train_node, y, graph, cost_type, imbalance):

    if cost_type == "lr":
        c1 = np.ones(len(y))
    elif cost_type == "trd":
        c1 = cost_trd(train_node, y, graph)
    elif cost_type == "etrd":
        c1 = cost_etrd(train_node, y, graph, k=1.25)
    elif cost_type == "nca":
        c1 = cost_nca(train_node, y, graph)
    else:
        raise Exception("Invalid cost type {}".format(cost_type))

    if imbalance:
        d1 = cost_imb(train_node, y)
        return d1*c1
    else:
        return c1

def get_tree_dist(train_node, y, graph):
    paths_dict = nx.single_source_shortest_path(graph.to_undirected(), train_node)
    path_len_dict = {k:(len(v)-1) for (k,v) in paths_dict.items()}

    gamma = np.zeros(len(y))
    for i in range(len(y)):
        gamma[i] = path_len_dict[y[i]]
    return gamma

def get_ancestor_dist(train_node, y, graph):
    num_common_anc = np.zeros(len(y))
    anc_tn = nx.ancestors(graph, train_node)
    anc_dict = dict([(k, nx.ancestors(graph,k)) for k in np.unique(y)])
    for i,li in enumerate(y):
        common = anc_dict[li] & anc_tn
        num_common_anc[i] = len(common)
    return num_common_anc

def get_num_examples_dict(y):
    counts = itemfreq(y)
    return dict(counts.astype(int))

def cost_imb(train_node, y):
    N0, L = 10.0,20.0
    numex_dict = get_num_examples_dict(y)
    tmp1 = np.array([numex_dict[lbl] for lbl in y])
    tmp2 = tmp1 - N0
    tmp2[tmp2 < 0] = 0
    tmp3 = 1+np.exp(tmp2)
    tmp4 = 1 + L/tmp3
    return tmp4


def cost_etrd(train_node, y, graph, k=1.25):
    # cost = ExTrD
    gamma = get_tree_dist(train_node, y, graph)
    cost = k**gamma
    cost[gamma==0] = np.max(cost)
    return cost


def cost_nca(train_node, y, graph):
    # cost = NCA
    alpha = get_ancestor_dist(train_node, y, graph)
    alpha_max = alpha.max()
    cost_vector = (alpha_max - alpha + 1)
    cost_vector[train_node == y] = alpha_max + 1
    return cost_vector


def cost_trd(train_node, y, graph):
    # cost = Tree Distance
    gamma = get_tree_dist(train_node, y, graph)
    gamma_max = 1.0*np.max(gamma)
    cost_vector = gamma
    cost_vector[train_node==y] = gamma_max
    return cost_vector
