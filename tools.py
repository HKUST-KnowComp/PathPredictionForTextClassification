import errno
import numpy as np
import os
import settings

try:
    import cpickle as pickle
except ImportError:
    import pickle
try:
    import ujson as json
except ImportError:
    import json


def safe_log(ary):
    return np.nan_to_num(np.log(ary))


def make_sure_path_exists(p):
    try:
        os.makedirs(p)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def label_to_delta(label):
    C = np.max(label) + 1
    X = label.shape[0]
    delta = np.zeros((X, C))
    delta[np.arange(X), label] = 1
    return delta

def save(output_path, data):
    file_extend = os.path.splitext(output_path)[1]
    if file_extend == '.npy':
        np.save(output_path, data)
    elif file_extend == '.npz':
        np.savez(output_path, **data)
    elif file_extend == '.json':
        with open(output_path, 'w',encoding='utf-8') as f:
            json.dump(data, f)
    elif file_extend == '.txt':
        with open(output_path, 'w',encoding='utf-8') as f:
            for x in data:
                f.write(x)
                f.write('\n')
    elif file_extend == '.pkl':
        from build_data_managers import DataManager
        with open(output_path, 'wb') as f:
            try:
                pickle.dump(data, f, protocol=4)
            except:
                pickle.dump(data, f)
    else:
        raise NotImplementedError

def load(input_path):
    file_extend = os.path.splitext(input_path)[1]
    if file_extend == '.npy':
        return np.load(input_path)
    elif file_extend == '.npz':
        return np.load(input_path)
    elif file_extend == '.json':
        with open(input_path, 'r',encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif file_extend == '.txt':
        with open(input_path, 'r',encoding='utf-8') as f:
            data = f.readlines()
        return data
    elif file_extend == '.pkl':
        from build_data_managers import DataManager
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            max_depth = len(data.labels)
            for depth in range(max_depth):
                data.deltas[depth] = data.deltas[depth].toarray()
                data.sims[depth] = data.sims[depth].toarray()
        return data
    else:
        raise NotImplementedError

def save_pre(pre, pre_file):
    with open(pre_file, 'w',encoding='utf-8') as f:
        for y in pre:
            f.write(str(y))
            f.write('\n')

def save_pres(pres, pre_file):
    max_depth = len(pres)
    with open(pre_file, 'w',encoding='utf-8') as f:
        N = len(pres[0]) if depth > 0 else 0
        for i in range(N):
            f.write(','.join([str(pres[j][i]) for j in range(max_depth)]))
            f.write('\n')

def load_pre(pre_file):
    pre = []
    with open(pre_file, 'r',encoding='utf-8') as f:
        for line in f:
            pre.append(int(line.strip()))
    return np.array(pre)

def load_pres(pre_file):
    pres = []
    with open(pre_file, 'r',encoding='utf-8') as f:
        for line in f:
            pres.append([int(y) for y in line.strip().split(',')])
    return np.array(pres).T

def find_available_filename(name,):
    """Get file or dir name that not exists yet"""
    file_name, file_extension = os.path.splitext(name)
    dirname = os.path.dirname(name)
    make_sure_path_exists(dirname)

    while True:
        tmp = "{}_{:04d}_{}".format(file_name, i, file_extension)
        if not os.path.exists(tmp):
            break
        i += 1

        tmp = "{}_{:03d}_{}".format(file_name, i, file_extension)
        assert not os.path.exists(tmp), "dir exists: {}".format(tmp)

    return tmp
