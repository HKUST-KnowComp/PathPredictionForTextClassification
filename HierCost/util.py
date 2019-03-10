import networkx as nx
import numpy as np 
from sklearn.metrics import f1_score, precision_recall_fscore_support

def get_graph_and_leaf_nodes(graph_path):
    '''
    Get graph and the list of leaf nodes.

    Returns:
        np.ndarray[]:int : list of leaf nodes in the graph.
    '''
    graph = nx.read_edgelist(graph_path, create_using=nx.DiGraph(),nodetype=int)
    leaf_nodes = np.array([ n for n in graph.nodes() if len(list(graph.successors(n))) == 0],dtype=int)
    return graph, leaf_nodes

def compute_p_r_f1(y_true, y_pre):
    p_M, r_M, f1_M, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pre, average="micro")
    return [(p_M, p_m), (r_M, r_m), (f1_M, f1_m)]