'''
Train hierarchical flat classifier
using cost sensitive learning based on hierarchical costs

REF:
Anveshi Charuvaka and Huzefa Rangwala "HierCost: Improving Large Scale
Hierarchical Classification with Cost Sensitive Learning"  European Conference
on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases, 2015
'''

import sys, os, pickle, time
from dataset_util import *
import numpy as np
import scipy.sparse
from slcost import *
from logcost import LogisticCost
from util import get_graph_and_leaf_nodes


def train_and_output_model(X, class_labels, train_node, graph, cost_type, imbalance, rho, outpath):
    '''
    Train model and save model to a pickled file.

    Args:
        X (np.ndarray[num_instances x num_features]:float): Training dataset
        class_labels (np.ndarray[num_instances]:int): class labels.
        train_node (int): positive training label.
        cost_type (str): cost type in ["lr", "nca", "trd", "etrd"]
        imbalance (bool): Include imbalance cost?
        rho (float): Regularization parameter
        outpath (str): output path of the model

    Returns:
        None
    '''
    # print("Training Model {} :  ".format(train_node), end='')
    start = time.time()

    y = 2*(class_labels == train_node).astype(int) - 1
    cost_vector = compute_treecost_vector(train_node, class_labels, graph,
            cost_type=cost_type, imbalance=imbalance)
    model = LogisticCost(rho=rho)
    model.fit(X, y, cost_vector)

    # save model
    safe_pickle_dump(model, outpath)
    end = time.time()
    # print(" time= {:.3f} sec".format(end-start))


def main(args):
    '''
    Driver function to
        - parse command line argumnets.
        - train models for all input nodes.
    '''

    mkdir_if_not_exists(args.model_dir)
    graph = safe_read_graph(args.hierarchy)
    X_train, labels_train = safe_read_svmlight_file(args.dataset, args.features)
    if args.nodes == "all":
        train_node_list = graph.nodes()
    elif args.nodes == "leaf":
        train_node_list = [ n for n in graph.nodes() if len(list(graph.successors(n))) == 0]
    elif args.nodes != '':
        train_node_list = [int(n) for n in args.nodes.split(",")]
    else:
        raise Exception("Need to assign nodes to train models")

    import time
    start = time.time()
    for train_node in train_node_list:
        model_save_path = '{}/model_{}.p'.format(args.model_dir, train_node)
        train_and_output_model(X_train, labels_train, train_node, graph,
                args.cost_type, args.imbalance, args.rho, model_save_path)
    print("Hiercost training time: " + str(time.time() - start))

