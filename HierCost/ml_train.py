'''
Train hierarchical flat classifier
using cost sensitive learning based on hierarchical costs
for hierarchical multi-label classification.

REF:
Anveshi Charuvaka and Huzefa Rangwala "HierCost: Improving Large Scale
Hierarchical Classification with Cost Sensitive Learning"  European Conference
on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases, 2015
'''

import time
from dataset_util import * #pylint: disable=W0614
from util import get_graph_and_leaf_nodes
import numpy as np
import scipy.sparse
from sklearn.preprocessing import MultiLabelBinarizer
from logscut import LogisticScut
from mlcost import *

def train_and_output_model(X, y, class_labels, train_node, graph, cost_type,
    imbalance, rho, outpath):

    # print("Training Model {} :  ".format(train_node), end='')
    start = time.time()

    cost_vector = compute_treecost_vector(train_node, class_labels, graph,
            cost_type=cost_type, imbalance=imbalance)
    model = LogisticScut(rho=rho)
    model.fit(X, y, cost_vector)

    # save model
    safe_pickle_dump(model, outpath)
    end = time.time()
    # print(" time= {:.3f} sec".format(end-start))

def main(args):

    mkdir_if_not_exists(args.model_dir)
    graph = safe_read_graph(args.hierarchy)
    X_train, labels_train = safe_read_svmlight_file_multilabel(
             args.dataset, args.features)
    lbin = MultiLabelBinarizer(sparse_output=True)
    label_matrix = lbin.fit_transform(labels_train)

    if args.nodes == "all":
        train_node_list = graph.nodes()
    elif args.nodes == "leaf":
        train_node_list = [ n for n in graph.nodes() if len(list(graph.successors(n))) == 0]
    elif args.nodes != '':
        train_node_list = [int(n) for n in args.nodes.split(",")]
    else:
        raise Exception("Need to assign nodes to train models")

    for train_node in train_node_list:
        model_save_path = '{}/model_{}.p'.format(args.model_dir, train_node)
        y_node = label_matrix[:, lbin.classes_ == train_node].toarray().flatten()
        train_and_output_model(X_train, y_node, labels_train, train_node,
                graph, args.cost_type, args.imbalance, args.rho, model_save_path)

