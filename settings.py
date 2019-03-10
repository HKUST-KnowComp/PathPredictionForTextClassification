import os
import numpy as np


INF = np.finfo(np.float32).min / 2
EPS = 1e-12

log_dir = 'logs'

input_dir_20ng = 'datasets/20newsgroups'
input_dir_rcv1 = 'datasets/rcv1org'
input_dir_clef = 'datasets/CLEF'
input_dir_lshtc_small = 'datasets/LSHTC_small'
input_dir_lshtc_large = 'datasets/LSHTC_large2010'
input_dir_ipc = 'datasets/IPC'
input_dirs = [input_dir_20ng, input_dir_rcv1]

data_dir_20ng = 'data/20ng'
data_dir_rcv1 = 'data/rcv1'
data_dir_clef = 'data/clef'
data_dir_lshtc_small = 'data/lshtc_small'
data_dir_lshtc_large = 'data/lshtc_large'
data_dir_ipc = 'data/ipc'
data_dirs = [data_dir_20ng, data_dir_rcv1]

max_vocab_size = 200000
train_ratio = 0.8
label_ratios = [0.1]
times = 1


label_unlabel_idx_file = 'label_unlabel_idx.npz'
train_test_idx_file = 'train_test_idx.npz'
deltas_file = 'deltas.npy'
classes_file = 'classes.json'
vocab_file = 'vocab.json'
cat_hier_file = 'cat_hier.txt'
labeled_data_manager_file = 'labeled_data_manager.pkl'
unlabeled_data_manager_file = 'unlabeled_data_manager.pkl'
test_data_manager_file = 'test_data_manager.pkl'
labeled_svmlight_file = 'labeled_svmlight.txt'
dataless_svmlight_file = 'dataless_svmlight.txt'
test_svmlight_file = 'test_svmlight.txt'

# 0: macro
# 1: micro
main_metric=1

path_weight = 1.0
soft_sim=False