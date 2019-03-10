'''
Hierarchical cost computation methods for multi-label hierarchical
classification.
'''

import networkx as nx
import numpy as np
from scipy.stats import itemfreq
from sklearn.preprocessing import MultiLabelBinarizer

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

def get_tree_dist(train_node, labels, graph, aggregate_function=np.min):
    paths_dict = nx.single_source_shortest_path(graph.to_undirected(), train_node)
    path_len_dict = {k:(len(v)-1) for (k,v) in paths_dict.items()}
    label_tree_dist = np.zeros(len(labels))
    for i, lbls in enumerate(labels):
        label_tree_dist[i] = aggregate_function([path_len_dict[l] for l in lbls])
    return label_tree_dist

def get_ancestor_dist(train_node, y, graph, aggregate_function=np.min):
    num_common_anc = np.zeros(len(y))
    anc_tn = nx.ancestors(graph, train_node)
    anc_dict = dict([(k, nx.ancestors(graph,k)) for k in set(flatten_list(y))])
    for i,li in enumerate(y):
        num_common_anc[i] = aggregate_function([len(anc_dict[ll] & anc_tn) for ll in li])
    return num_common_anc

def flatten_list(inlist):
    return [item for sublist in inlist for item in sublist]

def belong_to_node(train_node, y):
    return np.array([ train_node in lbl for lbl in y ])

def get_bin_lbl(train_node, y):
    lb = MultiLabelBinarizer(sparse_output=True)
    lbl_mat = lb.fit_transform(y)
    lbl_mat = lbl_mat.astype(bool).tocsc()
    y_bin = lbl_mat[:, lb.classes_ == train_node].toarray().flatten()
    return y_bin

def cost_imb(train_node, y, graph):
    N0, L = 10.0,20.0
    lb = MultiLabelBinarizer(sparse_output=True)
    lbl_mat = lb.fit_transform(y)
    lbl_mat = lbl_mat.astype(bool).tocsc()
    num_ex = lbl_mat.sum(axis=0).A1
    numex_dict = dict(zip(lb.classes_, num_ex))

    tmp1 = np.array( [ np.min([ numex_dict[l] for l in lbl]) for lbl in y]  )
    # tmp1 = np.array([numex_dict[lbl] for lbl in y])
    tmp2 = tmp1 - N0
    tmp2[tmp2 < 0] = 0
    tmp3 = 1+np.exp(tmp2)
    tmp4 = 1 + L/tmp3
    return tmp4


def cost_etrd(train_node, y, graph, k=1.25):
    label_tree_dist = get_tree_dist(train_node, y, graph)
    cost = k**label_tree_dist
    cost[belong_to_node(train_node, y)] = np.max(cost)
    return cost


def cost_nca(train_node, y, graph):
    # similar to cost102, all values scaled by alpha_max + 1
    alpha = get_ancestor_dist(train_node, y, graph)
    alpha_max = alpha.max()
    cost_vector = (alpha_max - alpha + 1)
    cost_vector[belong_to_node(train_node, y)] = alpha_max + 1
    return cost_vector


def cost_trd(train_node, y, graph):
    # similar to cost101, all values scaled by gamma_max
    gamma = get_tree_dist(train_node, y, graph)
    gamma_max = 1.0*np.max(gamma)
    cost_vector = gamma
    cost_vector[belong_to_node(train_node, y)] = gamma_max
    return cost_vector
