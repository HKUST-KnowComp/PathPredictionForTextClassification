import numpy as np
# from svmlight_loader  import load_svmlight_file, dump_svmlight_file
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
import networkx as nx
import scipy
import pickle
import os
import os.path
from scipy.sparse import csr_matrix

def mkdir_if_not_exists(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except:
        pass


def safe_operation(old_func):
    '''Decorator to make dicey operations repeat 3 times if they fail'''
    def new_func(*args,**kwargs):
        max_attempts = 3
        for i in range(max_attempts):
            try:
                result = old_func(*args, **kwargs)
                return result
            except Exception as e:
                print("Failed once", old_func)
                caught_excption = e
                continue
        else:
            print(args, kwargs)
            raise IOError("function failed "+ str(old_func), caught_excption)
    return new_func


def read_svmlight_file_multilabel(file_path, n_features=None):
    '''Reads multi-label svmlight file'''
    with open(file_path) as fin:
        line_index = 0
        data_indices = list()
        data = list()
        labels = []
        for line in fin:
            lbl_feat_str, sep, comment = line.strip().partition("#")
            tokens1 = lbl_feat_str.split(',')
            tokens2 = tokens1[-1].split()

            line_labels = [int(i) for i in tokens1[:-1] + tokens2[:1]]
            labels.append(line_labels)

            features = tokens2[1:]
            for f in features:
                fid, fval = f.split(':')
                data_indices.append([line_index, int(fid)-1])
                data.append(float(fval))
            line_index += 1
    if n_features == None:
        X = csr_matrix((np.array(data), np.array(data_indices).T))
    else:
        X = csr_matrix((np.array(data), np.array(data_indices).T), shape = (line_index, n_features))

    return X, labels

def dump_svmlight_file_multilabel(X, y, file_path):
    y_temp = np.zeros(len(y))
    dump_svmlight_file(X, y_temp, file_path, zero_based=False)

    data = []
    with open(file_path) as fin:
        counter = 0
        for line in fin:
            lbl_str = ",".join([str(lbl) for lbl in y[counter]])
            if line[0] == '0':
                out_line = lbl_str + line[1:]
                data.append(out_line)
            else:
                raise Exception("unexpected label")
            counter += 1

    with open(file_path,'w') as fout:
        fout.write("".join(data))


@safe_operation
def safe_read_graph(graph_path):
    '''Load a networkx.DiGraph from a file in edgelist format'''
    return  nx.read_edgelist(graph_path, create_using=nx.DiGraph(),nodetype=int)

@safe_operation
def safe_read_svmlight_file_multilabel(data_path, num_features=None):
    return read_svmlight_file_multilabel(data_path, num_features)


@safe_operation
def safe_read_svmlight_file(data_path, num_features=None):
    '''Reads dataset from file in libsvm sparse format, single label'''
    # X, y = load_svmlight_file(data_path, num_features, buffer_mb=300)
    X, y = load_svmlight_file(data_path, num_features)
    return X, y


@safe_operation
def safe_pickle_load(pickle_path):
    '''Reads a pickled object from file'''
    with open( pickle_path, "rb" ) as fin:
        model = pickle.load(fin)
    return model

@safe_operation
def safe_pickle_dump(pickle_object, output_path):
    '''Writes a pickled object to file'''
    with open( output_path, "wb" ) as fout:
        pickle.dump(pickle_object, fout, protocol=2)

def get_root_node(graph):
    root_cand = [n for n in graph.nodes() if not graph.predecessors(n)]
    if len(root_cand) > 1:
        raise Exception("Too many roots")
    else:
        return root_cand[0]
