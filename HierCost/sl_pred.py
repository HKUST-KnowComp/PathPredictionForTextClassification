'''
Test hierarchical flat classifier
using cost sensitive learning based on hierarchical costs

REF:
Anveshi Charuvaka and Huzefa Rangwala "HierCost: Improving Large Scale
Hierarchical Classification with Cost Sensitive Learning"  European Conference
on Machine Learning and Principles and Practice of Knowledge Discovery in
Databases, 2015
'''

import numpy as np
import networkx as nx
from util import get_graph_and_leaf_nodes, compute_p_r_f1
from dataset_util import safe_read_svmlight_file, safe_pickle_load, safe_read_graph
import scipy.sparse
import warnings

def pred_multiclass(X_test, model_dir, leaf_nodes):
    '''
    Predict class labels for test set.

    Args:
        X_test (np.ndarray[num_examples x num_features]:float): test dataset features.
        model_dir (str): Directory containing pickled model files (ending with *.p)
                         belonging to class LogisticCost, with one *.p file per leaf node.
        leaf_nodes (np.ndarray[]:int):list of leaf nodes in the graph.

    Returns:
        np.ndarray[num_examples]: predicted labels for test dataset.

    '''
    num_examples = X_test.shape[0]
    y_pred = np.zeros(num_examples, int)
    best_scores = np.zeros(num_examples)
    for idx, node in enumerate(leaf_nodes):
        model_save_path = '{}/model_{}.p'.format(
            model_dir, node)
        node_model = safe_pickle_load(model_save_path)
        node_score = node_model.decision_function(X_test)
        if idx == 0:
            y_pred[:] = node
            best_scores = node_score
        else:
            select_index = node_score > best_scores
            y_pred[select_index] = node
            best_scores[select_index] = node_score[select_index]
    return y_pred


def main(args):
    '''
    Driver function to
        - parse command line argumnets.
        - obtain predictions for test set and write them to a file.
    '''
    X_test, y_test = safe_read_svmlight_file(args.dataset, args.features)
    graph = safe_read_graph(args.hierarchy)
    if args.nodes == "all":
        pred_node_list = graph.nodes()
    elif args.nodes == "leaf":
        pred_node_list = [ n for n in graph.nodes() if len(list(graph.successors(n))) == 0]
    elif args.nodes != '':
        pred_node_list = [int(n) for n in args.nodes.split(",")]
    else:
        raise Exception("Need to assign nodes to train models")

    import time
    start = time.time()
    y_pred = pred_multiclass(X_test, args.model_dir, pred_node_list)
    print("Hiercost predicting time: " + str(time.time() - start))
    np.savetxt(args.pred_path, y_pred, fmt="%d")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # suppress UndefinedMetricWarning for macro_f1
        metrics = compute_p_r_f1(y_test, y_pred)
        print("Macro-Precision = {:.5f}".format(metrics[0][0]))
        print("Micro-Precision = {:.5f}".format(metrics[0][1]))
        print("Macro-Recall = {:.5f}".format(metrics[1][0]))
        print("Micro-Recall = {:.5f}".format(metrics[1][1]))
        print("Macro-F1 = {:.5f}".format(metrics[2][0]))
        print("Micro-F1 = {:.5f}".format(metrics[2][1]))
    return metrics, y_pred