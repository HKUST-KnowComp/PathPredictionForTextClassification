import logging
import logging.config
import logconfig
import random
import os
import tools
import numpy as np
import settings
import math
from shutil import copyfile
from collections import defaultdict
from util import generate_hier_info

def build_class_tree(paths, output_path):
    id2text = dict()
    id2label = []
    id2sims = []
    add_sims = False

    classes = []
    deltas = []
    class2idx = []
    class_tree = defaultdict(lambda: set())
    
    for i in range(len(paths)):
        classes.append(set())
        deltas.append(np.zeros(((0,0)), dtype=np.int32))
        id2label.append(dict())
        id2sims.append(dict())
        with open(paths[i], 'r', encoding='utf-8') as f:
            # remove '\n'
            try:
                line = f.readline()
                while line:
                    line = line.strip()
                    if line:
                        line = line.split('\t')
                        try:
                            id2label[-1][line[0]] = line[2]
                        except:
                            print(paths[i], line)
                        if len(line) == 4:
                            id2sims[-1][line[0]] = line[3]
                            add_sims = True
                    line = f.readline()
            except Exception as e:
                raise(e)

    # last layer
    with open(paths[-1], 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip()
            if line:
                line = line.split('\t')
                id2text[line[0]] = line[1]
            line = f.readline()
    
    valid_id = set(id2text.keys())
    for i in range(len(paths)-1):
        valid_id = valid_id.intersection(id2label[i].keys())

    for i in range(len(paths)):
        for id_ in valid_id:
            classes[i].add(id2label[i][id_])
        classes[i] = list(classes[i])
        classes[i].sort()
        class2idx.append(dict())
        for j, c in enumerate(classes[i]):
            class2idx[-1][c] = j
        if i > 0:
            deltas[i-1] = np.zeros((len(classes[i-1]), len(classes[i])), dtype=np.int32)

    for id_ in valid_id:
        for i in range(len(paths) - 1):
            deltas[i][class2idx[i][id2label[i][id_]], class2idx[i+1][id2label[i+1][id_]]] = 1
    for i in range(len(paths)):
        for j in range(len(classes[i])):
            classes[i][j] = classes[i][j][2:-1]

    tools.make_sure_path_exists(output_path)
    tools.save(os.path.join(output_path, settings.deltas_file), deltas)
    tools.save(os.path.join(output_path, settings.classes_file), classes)
    try:
        valid_id_list = list(valid_id)
        valid_id_list.sort(key=lambda x: int(x))
        valid_id = valid_id_list
    except:
        valid_id_list = list(valid_id)
        valid_id_list.sort()
        valid_id = valid_id_list
    for i in range(len(paths) - 1):
        with open(os.path.join(output_path, 'depth%d.txt' % (i+1)), 'w', encoding='utf-8') as f:
            for id_ in valid_id:
                f.write(id_ + '\tNULL\t' + id2label[i][id_])
                if add_sims:
                    f.write('\t' + id2sims[i].get(id_, ''))
                f.write('\n')
    i = len(paths) - 1
    with open(os.path.join(output_path, 'depth%d.txt' % (i+1)), 'w', encoding='utf-8') as f:
        for id_ in valid_id:
            f.write(id_ + '\t' + id2text[id_] + '\t' + id2label[i][id_])
            if add_sims:
                f.write('\t' + id2sims[i].get(id_, ''))
            f.write('\n')
    return deltas, classes

def split(data, classes, rate, index=None, seed=0):
    index1 = []
    index2 = []
    classes_set = set(classes)
    np.random.seed(seed)
    
    if index is not None:
        len_label = math.ceil(len(index) * rate)
        shuffle_index = index.copy()
    else:
        len_label = math.ceil(len(data) * rate)
        shuffle_index = list(range(len(data)))
    np.random.shuffle(shuffle_index)
    if rate == 1.0:
        index1 = shuffle_index
    else:    
        index1 = shuffle_index[0:len_label]
        index2 = shuffle_index[len_label:]
    return index1, index2

def split_train_test(data, classes, rate, output_dir, seed=0):
    train_idx, test_idx = split(data, classes, rate, seed=seed)
    tools.make_sure_path_exists(output_dir)
    tools.save(os.path.join(output_dir, 'train_test_idx.npz'), {'train_idx': train_idx, 'test_idx': test_idx})
    return train_idx, test_idx

def main(input_dir='./datasets/rcv1org', output_dir=settings.data_dir_20ng, split_randomly=True):
    logger = logging.getLogger(__name__)
    logger.info(logconfig.key_log(logconfig.DATA_NAME, input_dir))

    paths = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith('filtered'):
            paths.append(os.path.join(input_dir, file_name))
    paths.sort()

    logger.info(logconfig.key_log(logconfig.FUNCTION_NAME, 'build_class_tree'))
    deltas, classes = build_class_tree(paths, output_dir)

    logger.info(logconfig.key_log(logconfig.FUNCTION_NAME, 'split'))
    if split_randomly:
        data = tools.load(os.path.join(output_dir, 'depth%d.txt' % (len(paths))))
        train_idx, test_idx = split_train_test(data, classes[-1], settings.train_ratio, output_dir)
    else:
        copyfile(os.path.join(input_dir, 'train_test_idx.npz'), os.path.join(output_dir, 'train_test_idx.npz'))

    logger.info(logconfig.key_log(logconfig.FUNCTION_NAME, 'generate_hier_info'))
    generate_hier_info(deltas, classes, output_dir)

if __name__ == "__main__":
    log_filename = os.path.join(settings.log_dir, 'split_data.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))
    assert len(settings.input_dirs) == len(settings.data_dirs)
    for i in range(len(settings.input_dirs)):
        main(input_dir=settings.input_dirs[i], output_dir=settings.data_dirs[i], split_randomly=True)
