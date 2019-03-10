import logging
import logging.config
import logconfig
import numpy as np
import scipy.sparse as sparse
import os
import tools
import split_data
import time
import string
from nltk.corpus import stopwords
from collections import Counter
try:
    import cPickle as pickle
except ImportError:
    import pickle
import settings
from multiprocessing import Pool

stop_words = set(stopwords.words('english') + list(string.punctuation))

class DataManager:
    def __init__(self, name, xit=None, labels=None, deltas=None, sims=None, true_idx=None):
        self.name = name
        self.xit = xit
        self.labels = labels
        self.deltas = deltas
        self.sims = sims
        self.true_idx = true_idx
        self._dict = {'name': self.name,
                      'xit': self.xit,
                      'deltas': self.deltas,
                      'labels': self.labels,
                      'sims': self.sims,
                      'true_idx': self.true_idx
                      }
    def __getitem__(self, key):
        return self._dict[key]


def split_label_unlabel(data, index, classes, rate, output_dir, seed=0):
    label_idx, unlabel_idx = split_data.split(data, classes, rate, index=index, seed=seed)
    tools.make_sure_path_exists(output_dir)
    tools.save(os.path.join(output_dir, 'label_unlabel_idx.npz'), {'label_idx': label_idx, 'unlabel_idx': unlabel_idx})
    return label_idx, unlabel_idx


def get_vocab_info(doc, labels, train_idx, output_path, sparse_format=False):
    if os.path.exists(os.path.join(output_path, settings.vocab_file)):
        vocab_info = tools.load(os.path.join(output_path, settings.vocab_file))
        if len(vocab_info['vocab_dict']) <= settings.max_vocab_size:
            return vocab_info
    tf = Counter()
    data_doc = [doc[i] for i in train_idx]
    leaf_label = labels[-1][train_idx]
    for i, x in enumerate(data_doc):
        for word_tuple in x:
            word, frequency = word_tuple
            if sparse_format or (word not in stop_words and not word.isnumeric()):
                tf[word] += frequency

    vocab_dict = dict()
    new_tf = Counter()
    for i, v in enumerate(tf.most_common(settings.max_vocab_size)):
        vocab_dict[v[0]] = i
        new_tf[v[0]] = tf[v[0]]
    tf = new_tf
    tf["<DF>"] = len(data_doc) # to store the number of documents
    vocab_info = {"vocab_dict": vocab_dict, "tf": tf}
    tools.save(os.path.join(output_path, settings.vocab_file), vocab_info)
    return vocab_info


def process_dataset(input_dir, output_dir, sparse_format=False):
    if os.path.exists(os.path.join(output_dir, settings.labeled_data_manager_file)) and \
        os.path.exists(os.path.join(output_dir, settings.unlabeled_data_manager_file)) and \
            os.path.exists(os.path.join(output_dir, settings.test_data_manager_file)) and \
                os.path.exists(os.path.join(input_dir, settings.vocab_file)):
        labeled_data_manager = tools.load(os.path.join(output_dir, settings.labeled_data_manager_file))
        unlabeled_data_manager = tools.load(os.path.join(output_dir, settings.unlabeled_data_manager_file))
        test_data_manager = tools.load(os.path.join(output_dir, settings.test_data_manager_file))
        vocab_info = tools.load(os.path.join(input_dir, settings.vocab_file))
        return [labeled_data_manager, unlabeled_data_manager, test_data_manager], vocab_info

    classes = tools.load(os.path.join(input_dir, settings.classes_file))
    classes_idx = list(map(lambda x: dict(zip(x, range(len(x)))), classes))
    train_test_idx = tools.load(os.path.join(input_dir, settings.train_test_idx_file))
    train_idx, test_idx = train_test_idx['train_idx'], train_test_idx['test_idx']
    label_unlabel_idx = tools.load(os.path.join(output_dir, settings.label_unlabel_idx_file))
    label_idx, unlabel_idx = label_unlabel_idx['label_idx'], label_unlabel_idx['unlabel_idx']
    
    data_size = len(train_idx) + len(test_idx)
    doc = []
    labels = []
    deltas = []
    sims = []

    for depth in range(len(classes)):
        labels.append(np.zeros((data_size, ), dtype=np.int32))
        deltas.append(sparse.lil_matrix((data_size, len(classes[depth])), dtype=np.float32))
        sims.append(sparse.lil_matrix((data_size, len(classes[depth])), dtype=np.float32))

        file_name = os.path.join(input_dir, 'depth%d.txt' % (depth+1))
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip().split('\t')
            c = classes_idx[depth][line[2][2:-1]]
            labels[depth][i] = c
            deltas[depth][i, c] = 1
            sim = line[3].split(';')[:-1] if len(line) == 4 else []
            for x in sim:
                x_sp = x.split(',')
                if x_sp[1] not in classes_idx[depth].keys():
                    continue
                sims[depth][i,classes_idx[depth][x_sp[1]]] = float(x_sp[2])
            if depth == len(classes) - 1:
                if sparse_format:
                    word_tuples = list(map(lambda x: x.split(':', 1), line[1].split()))
                else:
                    word_tuples = list(Counter(line[1].split()).items())
                word_tuples = list(map(lambda word_tuple: (word_tuple[0], float(word_tuple[1])), word_tuples))
                doc.append(word_tuples)

    vocab_info = get_vocab_info(doc, labels, train_idx, input_dir, sparse_format=sparse_format)
    vocab_dict = vocab_info['vocab_dict']
    labeled_data_manager = build_data_manager('labeled', label_idx, doc, labels, deltas, sims, vocab_dict)
    unlabeled_data_manager = build_data_manager('unlabeled', unlabel_idx, doc, labels, deltas, sims, vocab_dict)
    test_data_manager = build_data_manager('test', test_idx, doc, labels, deltas, sims, vocab_dict)
    tools.save(os.path.join(output_dir, settings.labeled_data_manager_file), labeled_data_manager)
    tools.save(os.path.join(output_dir, settings.unlabeled_data_manager_file), unlabeled_data_manager)
    tools.save(os.path.join(output_dir, settings.test_data_manager_file), test_data_manager)
    return [labeled_data_manager, unlabeled_data_manager, test_data_manager], vocab_info

def build_data_manager(name, idx, doc, labels, deltas, sims, vocab_dict, sparse_format=False):
    data_labels = []
    data_deltas = []
    data_sims = []
    if len(idx) > 0:
        for depth in range(len(labels)):
            data_labels.append(labels[depth][idx])
            data_deltas.append(deltas[depth][idx])
            data_sims.append(sims[depth][idx])
        data_doc = [doc[i] for i in idx]
        # get doc word freq
        data_xit = sparse.lil_matrix((len(data_doc), len(vocab_dict)), dtype=np.float32)
        
        for i, x in enumerate(data_doc):
            for word_tuple in x:
                word, frequency = word_tuple
                if word in vocab_dict:
                    data_xit[i, vocab_dict[word]] = frequency
    else:
        for depth in range(len(labels)):
            data_labels.append(np.zeros((0,), dtype=np.int32))
            data_deltas.append(sparse.lil_matrix((0, deltas[depth].shape[1]), dtype=np.float32))
            data_sims.append(sparse.lil_matrix((0, sims[depth].shape[1]), dtype=np.float32))
            data_xit = sparse.lil_matrix((0, len(vocab_dict)), dtype=np.float32)
    data_xit = data_xit.tocsc()
    data_manager = DataManager(name, xit=data_xit, labels=data_labels, deltas=data_deltas, sims=data_sims)
    return data_manager

def main(input_dir=settings.data_dir_20ng, label_ratio=0.1, time=0, sparse_format=False):
    logger = logging.getLogger(__name__)
    logger.info(logconfig.key_log(logconfig.DATA_NAME, input_dir))

    depth_files = []
    for file_name in os.listdir(input_dir):
        if file_name.startswith('depth'):
            depth_files.append(file_name)
    depth_files.sort()

    data = tools.load(os.path.join(input_dir, depth_files[-1]))
    classes = tools.load(os.path.join(input_dir, settings.classes_file))
    train_test_idx = tools.load(os.path.join(input_dir, settings.train_test_idx_file))
    train_idx = train_test_idx['train_idx']

    output_dir = os.path.join(input_dir, str(label_ratio), str(time))

    logger.info(logconfig.key_log(logconfig.FUNCTION_NAME, 'split_label_unlabel'))
    label_idx, unlabel_idx = split_label_unlabel(data, train_idx, classes[-1], label_ratio, output_dir, seed=time)

    logger.info(logconfig.key_log(logconfig.FUNCTION_NAME, 'process_dataset'))
    [labeled_data_manager, unlabeled_data_manager, test_data_manager], vocab_info = \
        process_dataset(input_dir, output_dir, sparse_format=sparse_format)
    logger.info(logconfig.key_log('VocabularySize', str(len(vocab_info['vocab_dict']))))
            
if __name__ == "__main__":
    log_filename = os.path.join(settings.log_dir, 'build_data_managers.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))
    
    pool = Pool()
    for input_dir in settings.data_dirs:
        sparse_format = False
        for label_ratio in settings.label_ratios:
            for seed in range(settings.times):
                # pool.apply_async(main, args=(input_dir, label_ratio, seed, sparse_format))
                main(input_dir, label_ratio, seed, sparse_format)
                if label_ratio == 1.0:
                    break
    pool.close()
    pool.join()
