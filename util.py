import tools
import settings
import os
import numpy as np
from sklearn.preprocessing import normalize as sklearn_normalize
from sklearn.metrics import precision_recall_fscore_support


def generate_hier_info(deltas, classes, output_dir):
    max_depth = len(deltas)
    hierarchy_file = os.path.join(output_dir, settings.cat_hier_file)

    hier_tree = dict()
    class2no = dict()
    class_set = set()
    hier_tree[0] = 0
    class_set.add("Root")
    class2no["Root"] = 0
    class_cnt = 1

    with open(hierarchy_file, "w") as f:
        for depth in range(max_depth):
            for c in classes[depth]:
                if c not in class_set:
                    hier_tree[class_cnt] = 0
                    class_set.add(c)
                    class2no[c] = class_cnt
                    class_cnt += 1
        for c in classes[0]:
            f.write('0 ' + str(class2no[c]) + '\n')
        for depth in range(max_depth-1):
            delta = deltas[depth]
            shape = delta.shape
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if delta[i][j] == 1 and classes[depth][i] != classes[depth+1][j]:
                        hier_tree[class2no[classes[depth+1][j]]] = class2no[classes[depth][i]]
                        f.write(str(class2no[classes[depth][i]]) + ' ' + str(class2no[classes[depth+1][j]]) + '\n')
    nos = []
    for depth in range(max_depth):
        nos.append([])
        for i in range(len(classes[depth])):
            nos[depth].append(class2no[classes[depth][i]])
    return nos, hier_tree

def get_hier_info(input_dir):
    classes = tools.load(os.path.join(input_dir, settings.classes_file))
    hierarchy_file = os.path.join(input_dir, settings.cat_hier_file)
    max_depth = len(classes)
    
    hier_tree = dict()
    class2no = dict()
    class_set = set()
    hier_tree[0] = 0
    class_set.add("Root")
    class2no["Root"] = 0
    class_cnt = 1
    for depth in range(max_depth):
        for c in classes[depth]:
            if c not in class_set:
                hier_tree[class_cnt] = 0
                class_set.add(c)
                class2no[c] = class_cnt
                class_cnt += 1
    with open(hierarchy_file, "r") as f:
        line = f.readline()
        while line:
            line = line.split()
            hier_tree[int(line[1])] = int(line[0])
            line = f.readline()
    
    nos = []
    for depth in range(max_depth):
        nos.append([])
        for i in range(len(classes[depth])):
            nos[depth].append(class2no[classes[depth][i]])
    return nos, hier_tree

def load_data_managers(input_dir, filter_words=True):
    from build_data_managers import DataManager
    labeled_data = tools.load(os.path.join(input_dir, settings.labeled_data_manager_file))
    unlabeled_data = tools.load(os.path.join(input_dir, settings.unlabeled_data_manager_file))
    test_data = tools.load(os.path.join(input_dir, settings.test_data_manager_file))
    if filter_words:
        non_zero_indices = np.nonzero(labeled_data.xit)
        non_zero_columns = sorted(set(non_zero_indices[1]))
        labeled_data.xit = labeled_data.xit[:,non_zero_columns]
        unlabeled_data.xit = unlabeled_data.xit[:,non_zero_columns]
        test_data.xit = test_data.xit[:,non_zero_columns]
    return [labeled_data, unlabeled_data, test_data]

def build_reverse_data_manager(data_manager, name):
    from build_data_managers import DataManager
    return DataManager(name, xit=data_manager.xit, labels=data_manager.labels[::-1], 
        deltas=data_manager.deltas[::-1], sims=data_manager.sims[::-1], true_idx=data_manager.true_idx)

def build_subdata_manager(data_manager, name, true_idx=None):
    from build_data_managers import DataManager
    if true_idx is None:
        return DataManager(name, xit=data_manager.xit, labels=data_manager.labels, 
            deltas=data_manager.deltas, sims=data_manager.sims, true_idx=data_manager.true_idx)
    elif len(true_idx) > 0:
        data_xit = data_manager.xit[true_idx, :]
        data_labels = []
        data_deltas = []
        data_sims = []
        for depth in range(len(data_manager.deltas)):
            data_labels.append(data_manager.labels[depth][true_idx])
            data_deltas.append(data_manager.deltas[depth][true_idx])
            data_sims.append(data_manager.sims[depth][true_idx, :])
        return DataManager(name, xit=data_xit, labels=data_labels, 
            deltas=data_deltas, sims=data_sims, true_idx=true_idx)
    else:
        data_xit = np.zeros([0] + list(data_manager.xit.shape[1:]))
        data_labels = []
        data_deltas = []
        data_sims = []
        for depth in range(len(data_manager.deltas)):
            data_labels.append(np.zeros([0] + list(data_manager.labels[depth].shape[1:])))
            data_deltas.append(np.zeros([0] + list(data_manager.deltas[depth].shape[1:])))
            data_sims.append(np.zeros([0] + list(data_manager.sims[depth].shape[1:])))
        return DataManager(name, xit=data_xit, labels=data_labels, 
            deltas=data_deltas, sims=data_sims, true_idx=true_idx)

def normalize_theta(x, axis=0, beta=1):
    M = x.shape[0]
    norm = np.sum(x, axis=axis, keepdims=True)
    return (x + beta) / (norm + M * beta)

def normalize(x, axis=1, norm='l1'):
    return sklearn_normalize(x, norm=norm, axis=axis)

def hardmax(x, axis=1):
    one = np.argmax(x, axis=axis)
    hard_x = np.zeros(x.shape)
    hard_x[np.arange(hard_x.shape[0]), one] = 1.0
    return hard_x

def softmax(loga, axis=-1):
    """
    Compute the sotfmax function (normalized exponentials) without underflow.
    return:exp^a_i / \sum_j exp^a_j
    """
    m = np.max(loga, axis=axis, keepdims=True)
    logam = loga - m
    out = np.exp(logam)
    out /= np.sum(out, axis=axis, keepdims=True)
    return out

def logsum(loga, axis=-1):
    """
    Compute a sum of logs without underflow.
    \log \sum_c e^{b_c} = log [(\sum_c e^{b_c}) e^{-B}e^B]
                        = log [(\sum_c e^{b_c-B}) e^B]
                        = [log(\sum_c e^{b_c-B}) + B
    where B = max_c b_c
    return:log(sum_i a_i)
    """
    B = np.max(loga, axis=axis, keepdims=True)
    logaB = aB = loga - B
    aB = np.exp(logaB)
    return np.log(np.sum(aB, axis=axis)) + np.squeeze(B)

def check_labels(labeled_data_delta):
    labels = sorted(list(set(labeled_data_delta)))
    return labels

def compute_path_score(labeled_data_deltas, deltas, path_weights=None):
    if path_weights is None:
        path_weights = [1] * len(deltas)
    delta = np.identity(labeled_data_deltas[-1].shape[1]) # Cn x Cn
    path_score = path_weights[-1] * labeled_data_deltas[-1]
    for depth in range(len(deltas)-2, -1, -1):
        delta = np.dot(deltas[depth], delta) # Ci x Cn
        path_score += path_weights[depth] * np.dot(labeled_data_deltas[depth], delta)
    return path_score

def compute_p_r_f1(y_true, y_pre):
    p_M, r_M, f1_M, _ = precision_recall_fscore_support(y_true, y_pre, average="macro")
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pre, average="micro")
    return [(p_M, p_m), (r_M, r_m), (f1_M, f1_m)]

def compute_hier_p_r_f1(y_true_last, y_pre_last, nos, hier_tree):
    y_true = []
    y_pre = []
    need_modified = False
    for x in y_true_last:
        if x < nos[-1][0]:
            need_modified = True
            break
    if need_modified:
        no_last = dict(zip(range(len(nos[-1])), nos[-1]))
    else:
        no_last = dict()
        for k in hier_tree:
            no_last[k] = k
    assert len(y_true_last) == len(y_pre_last)
    for i in range(len(y_true_last)):
        x = no_last[y_true_last[i]]
        true_path = []
        while x != 0:
            true_path.append(x)
            x = hier_tree[x]
        true_path.reverse()
        y_true.extend(true_path)

        x = no_last[y_pre_last[i]]
        pre_path = []
        while x != 0:
            pre_path.append(x)
            x = hier_tree[x]
        pre_path.reverse()
        if len(pre_path) < len(true_path):
            for j in range(len(pre_path), len(true_path)):
                pre_path.append(-1)
        elif len(pre_path) > len(true_path):
            pre_path = pre_path[:len(true_path)]
        y_pre.extend(pre_path)
    return compute_p_r_f1(y_true, y_pre)

def compute_overall_p_r_f1(y_trues, y_pres, nos):
    """
    compute all leaf nodes and internal nodes, including the dummy internal nodes
    """
    y_true = []
    y_pre = []
    assert len(nos) == len(y_trues)
    max_depth = len(nos)
    assert len(y_trues[0]) == len(y_pres[0])
    N = len(y_trues[0])
    for i in range(N):
        true_path = []
        pre_path = []
        for depth in range(max_depth):
            x = nos[depth][y_trues[depth][i]]
            if depth == 0 or x != true_path[-1]:
                true_path.append(x)
            x = nos[depth][y_pres[depth][i]]
            if depth == 0 or x != pre_path[-1]:
                pre_path.append(x)
        if len(pre_path) < len(true_path):
            for j in range(len(pre_path), len(true_path)):
                pre_path.append(-1)
        elif len(pre_path) > len(true_path):
            pre_path = pre_path[:len(true_path)]
        y_true.extend(true_path)
        y_pre.extend(pre_path)
    return compute_p_r_f1(y_true, y_pre)

def compute_multi_p_r_f1(y_true_multi, y_pre_multi):
    assert len(y_true_multi) == len(y_pre_multi)
    begin_class_no = 1
    y_true = []
    y_pre = []
    if len(y_true_multi) > 0:
        class_no = len(y_true_multi[0])
        for i in range(len(y_true_multi)):
            true_path = []
            pre_path = []
            for j in range(begin_class_no, class_no):
                if y_true_multi[i][j] == 1.0:
                    true_path.append(j)
                if y_pre_multi[i][j] == 1.0:
                    pre_path.append(j)
            if len(pre_path) < len(true_path):
                for j in range(len(pre_path), len(true_path)):
                    pre_path.append(-1)
            else:
                pre_path = pre_path[:len(true_path)]
            y_true.extend(true_path)
            y_pre.extend(pre_path)
    return compute_p_r_f1(y_true, y_pre)

def compute_statistics(labels, pres, deltas):
    statistics = []
    for depth in range(len(deltas) - 1):
        n11 = 0
        n12 = 0
        n2 = 0
        n31 = 0
        n32 = 0
        n4 = 0
        n = len(labels[depth])
        for i in range(n):
            eq_1 = labels[depth][i] == pres[depth][i]
            eq_2 = labels[depth+1][i] == pres[depth+1][i]
            delta_12 = deltas[depth][pres[depth][i], pres[depth+1][i]]
            if not eq_1 and not eq_2:
                if delta_12:
                    n11 += 1
                else:
                    n12 += 1
            elif eq_1 and eq_2:
                n2 += 1
            elif eq_1 and not eq_2:
                if delta_12:
                    n31 += 1
                else:
                    n32 += 1
            elif not eq_1 and eq_2:
                n4 += 1
        n11 = n11 * 1.0 / n
        n12 = n12 * 1.0 / n
        n2 = n2 * 1.0 / n
        n31 = n31 * 1.0 / n
        n32 = n32 * 1.0 / n
        n4 = n32 * 1.0 / n
        statistics.append([n11, n12, n2, n31, n32, n4])
    return statistics

