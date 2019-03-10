import logging
import logging.config
import logconfig
import numpy as np
import settings
import time
import tools
import os
import csv
import math
from multiprocessing import Process, Pool
from build_data_managers import DataManager
from util import *
import time


def train_NB(data_xit, data_delta):
    '''
    :param data_delta: [document_number, M(class number)], type:np.ndarray
    :param data_xit: [document_number, V(Vocabulary number)], type: scipy.sparse.lil.lil_matrix
    return the unnormalized thetac=p(c)(an M array), thetawc=p(w|c)(a V*M matrix)
    '''
    return train_NBMC(data_xit, [data_delta])


def train_NBMC(data_xit, data_deltas):
    thetas = []
    thetas.append(np.sum(data_deltas[0], axis=0, keepdims=False))
    for depth in range(1, len(data_deltas)):
        thetas.append(np.dot(data_deltas[depth].T, data_deltas[depth - 1]))
    thetas.append(data_xit.transpose().dot(data_deltas[-1]))
    return thetas


def train_hierNB(data_xit, data_deltas):
    thetas_list = []
    for depth in range(len(data_deltas)):
        thetas = train_NB(data_xit, data_deltas[depth])
        thetas_list.append(thetas)
    return thetas_list


def train_WDNB(data_xit, data_deltas):
    return train_hierNB(data_xit, data_deltas)


def train_EM(labeled_data_xit, labeled_data_delta, unlabeled_data_xit, max_iter=50,
             eps=1e-3):
    # Initialize parameters
    labeled_thetas = train_NB(labeled_data_xit, labeled_data_delta)
    thetas = list(map(lambda theta: normalize_theta(theta, axis=0), labeled_thetas))
    l = settings.INF
    for i in range(1, max_iter + 1):
        # E-step: find the probabilistic labels for unlabeled data
        P = predict(thetas, unlabeled_data_xit)
        # M-step: train classifier with the union of labeled and unlabeled data
        unlabeled_thetas = train_NB(unlabeled_data_xit, P)
        thetas_new = []
        for j in range(len(thetas)):
            thetas_new.append(
                normalize_theta(labeled_thetas[j] + unlabeled_thetas[j], axis=0))
        l_new = log_prob(thetas_new, labeled_data_xit, labeled_data_delta, unlabeled_data_xit=unlabeled_data_xit)
        l_diff = (l_new - l) / (abs(l) + settings.EPS)

        if l_diff < eps:
            break
        l = l_new
        thetas = thetas_new
    return thetas


def train_EMMC(labeled_data_xit, labeled_data_deltas, unlabeled_data_xit,
                       deltas, max_iter=50, eps=1e-3):
    # Initialize parameters
    labeled_thetas = train_NBMC(labeled_data_xit, labeled_data_deltas)
    thetas = list(map(lambda theta: normalize_theta(theta, axis=0), labeled_thetas))
    l = settings.INF
    for i in range(1, max_iter + 1):
        # E-step: find the probabilistic labels for unlabeled data
        Ps = predict_multicomp(thetas, unlabeled_data_xit, deltas)
        # M-step: train classifier with the union of labeled and unlabeled data
        unlabeled_thetas = train_NBMC(unlabeled_data_xit, Ps)
        thetas_new = []
        for j in range(len(thetas)):
            thetas_new.append(
                normalize_theta(labeled_thetas[j] + unlabeled_thetas[j], axis=0))
        l_new = log_prob_multicomp(thetas_new, labeled_data_xit, labeled_data_deltas[-1],
                                   unlabeled_data_xit=unlabeled_data_xit)
        l_diff = (l_new - l) / (abs(l) + settings.EPS)

        if l_diff < eps:
            break
        l = l_new
        thetas = thetas_new
    return thetas


def train_hierEM(labeled_data_xit, labeled_data_deltas, unlabeled_data_xit,
                  deltas=None, max_iter=50, eps=1e-3):
    # Initialize parameters
    labeled_thetas_list = train_hierNB(labeled_data_xit, labeled_data_deltas)
    thetas_list = list(
        map(lambda thetas: list(map(lambda theta: normalize_theta(theta, axis=0), thetas)), labeled_thetas_list))
    ls = [settings.INF] * len(thetas_list)
    for i in range(1, max_iter + 1):
        # E-step: find the probabilistic labels for unlabeled data
        Ps = predict_hier(thetas_list, unlabeled_data_xit, deltas)
        # M-step: train classifier with the union of labeled and unlabeled data
        unlabeled_thetas_list = train_hierNB(unlabeled_data_xit, Ps)
        thetas_list_new = []
        for j in range(len(thetas_list)):
            thetas_list_new.append([])
            for k in range(len(thetas_list[0])):
                thetas_list_new[j].append(
                    normalize_theta(labeled_thetas_list[j][k] + unlabeled_thetas_list[j][k],
                                    axis=0))
        ls_new = log_prob_hier(thetas_list_new, labeled_data_xit, labeled_data_deltas,
                               unlabeled_data_xit=unlabeled_data_xit)
        l_diff = 0.0
        l_diff_abs = 0.0
        for j in range(len(ls)):
            l_diff += (ls_new[j] - ls[j]) / abs(ls[j])
            l_diff_abs += abs(ls_new[j] - ls[j]) / abs(ls[j])
            ls[j] = ls_new[j]

        if l_diff < eps or l_diff_abs < eps:
            break
        thetas_list = thetas_list_new
    return thetas_list


def train_WDEM(labeled_data_xit, labeled_data_deltas, unlabeled_data_xit,
                           deltas=None, path_weights=None, max_iter=50, eps=1e-3):
    # Initialize parameters
    labeled_thetas_list = train_WDNB(labeled_data_xit, labeled_data_deltas)
    thetas_list = list(
        map(lambda thetas: list(map(lambda theta: normalize_theta(theta, axis=0), thetas)), labeled_thetas_list))
    l = settings.INF
    for i in range(1, max_iter + 1):
        # E-step: find the probabilistic labels for unlabeled data
        Ps = predict_xin_pathscore(thetas_list, unlabeled_data_xit, deltas=deltas, path_weights=path_weights)
        # M-step: train classifier with the union of labeled and unlabeled data
        unlabeled_thetas_list = train_WDNB(unlabeled_data_xit, Ps)
        thetas_list_new = []
        for j in range(len(thetas_list)):
            thetas_list_new.append([])
            for k in range(len(thetas_list[0])):
                thetas_list_new[j].append(
                    normalize_theta(labeled_thetas_list[j][k] + unlabeled_thetas_list[j][k],
                                    axis=0))
        l_new = log_prob_xin_pathscore(thetas_list_new, labeled_data_xit, labeled_data_deltas,
                                       unlabeled_data_xit=unlabeled_data_xit, deltas=deltas, path_weights=path_weights)
        l_diff = (l_new - l) / (abs(l) + settings.EPS)

        if l_diff < eps:
            break
        l = l_new
        thetas_list = thetas_list_new
    return thetas_list


def train_PCEM(labeled_data_xit, labeled_data_deltas, unlabeled_data_xit, deltas, path_weights,
                             max_iter=50, eps=1e-3):
    # Initialize parameters
    labeled_path_score = compute_path_score(labeled_data_deltas, deltas, path_weights)
    labeled_path_label = hardmax(labeled_path_score, axis=1)
    labeled_thetas = train_NB(labeled_data_xit, labeled_path_score)
    thetas = list(map(lambda theta: normalize_theta(theta, axis=0), labeled_thetas))
    l = settings.INF
    sum_weights = np.sum(path_weights)
    for i in range(1, max_iter + 1):
        # E-step: find the probabilistic labels for unlabeled data
        Ps = predict_huiru_pathscore(thetas, unlabeled_data_xit, deltas)
        # path_score = compute_path_score([hardmax(p, axis=1) for p in Ps], deltas, path_weights)
        path_score = compute_path_score(Ps, deltas, path_weights)
        # M-step: train classifier with the union of labeled and unlabeled data
        unlabeled_thetas = train_NB(unlabeled_data_xit, Ps[-1])   # path_score
        thetas_new = []
        for j in range(len(thetas)):
            thetas_new.append(
                normalize_theta(labeled_thetas[j] + unlabeled_thetas[j], axis=0))
        l_new = log_prob(thetas, labeled_data_xit, labeled_path_label, unlabeled_data_xit=unlabeled_data_xit)
        l_diff = (l_new - l) / (abs(l) + settings.EPS)

        if l_diff < eps:
            break
        l = l_new
        thetas = thetas_new
    return thetas


def train_one_depth(data_managers_list, depth, deltas, method):
    thetas_list = []
    unlabeled_cnt = 0
    test_cnt = 0
    for i, data_managers in enumerate(data_managers_list):
        unlabeled_cnt += data_managers[1].xit.shape[0] if data_managers[1] else 0
        test_cnt += data_managers[2].xit.shape[0]
    unlabeled_pre = np.zeros((unlabeled_cnt,), dtype=np.int32)
    test_label = np.zeros((test_cnt,), dtype=np.int32)
    test_pre = np.zeros((test_cnt,), dtype=np.int32)
    for i, data_managers in enumerate(data_managers_list):
        if depth == 0:
            next_labels = np.array(range(deltas[0].shape[0]))
        else:
            next_labels = np.nonzero(deltas[depth - 1][i])[0]
        if 'labeled' in method:
            sim = data_managers[0].deltas[depth][:, next_labels]
        elif 'dataless' in method:
            if settings.soft_sim:
                if depth == 0:
                    sim = normalize(data_managers[0].sims[depth], axis=1)
                else:
                    sim = normalize(data_managers[0].sims[depth][:, next_labels], axis=1)
            else:
                if depth == 0:
                    sim = hardmax(data_managers[0].sims[depth], axis=1)
                else:
                    sim = hardmax(data_managers[0].sims[depth][:, next_labels], axis=1)
        else:
            raise NotImplementedError
        if 'NB' in method:
            thetas = train_NB(data_managers[0].xit, sim)
            thetas = list(map(lambda theta: normalize_theta(theta, axis=0), thetas))
            thetas_list.append(thetas)
            unlabeled_pre_part = []
            test_pre_part = predict_label(thetas, data_managers[2].xit)
            if depth != 0:
                test_pre_part = np.array([next_labels[x] for x in test_pre_part])
        elif 'EM' in method:
            thetas = train_EM(data_managers[0].xit, sim, data_managers[1].xit)
            thetas_list.append(thetas)
            unlabeled_pre_part = predict_label(thetas, data_managers[1].xit)
            test_pre_part = predict_label(thetas, data_managers[2].xit)
            if depth != 0:
                unlabeled_pre_part = np.array([next_labels[x] for x in unlabeled_pre_part])
                test_pre_part = np.array([next_labels[x] for x in test_pre_part])
            if data_managers[1].true_idx is None:
                unlabeled_pre = unlabeled_pre_part
            elif len(data_managers[1].true_idx) > 0:
                unlabeled_pre[data_managers[1].true_idx] = unlabeled_pre_part
        else:
            raise NotImplementedError
        if data_managers[2].true_idx is None:
            test_label = data_managers[2].labels[depth]
            test_pre = test_pre_part
        elif len(data_managers[2].true_idx) > 0:
            test_label[data_managers[2].true_idx] = data_managers[2].labels[depth]
            test_pre[data_managers[2].true_idx] = test_pre_part
    return thetas_list, unlabeled_pre, test_pre


def log_prob(thetas, labeled_data_xit, labeled_data_delta, unlabeled_data_xit=None):
    return log_prob_multicomp(thetas, labeled_data_xit, labeled_data_delta, unlabeled_data_xit=unlabeled_data_xit)


def log_prob_multicomp(thetas, labeled_data_xit, labeled_data_delta, unlabeled_data_xit=None):
    log_theta_wc = np.log(thetas[-1])  # V * Cn
    theta_c = np.expand_dims(thetas[0], axis=1)
    for depth in range(1, len(thetas) - 1):
        theta_c = np.dot(thetas[depth], theta_c)
    log_theta_c = np.log(theta_c)  # Cn * 1
    l = 0.0
    if labeled_data_xit is not None:
        log_theta_xc = labeled_data_xit.dot(log_theta_wc)  # N * Cn
        l += np.sum(logsum(np.where(labeled_data_delta, (log_theta_xc + log_theta_c.T), settings.INF)))
    if unlabeled_data_xit is not None and unlabeled_data_xit.shape[0] > 0:
        log_theta_xc = unlabeled_data_xit.dot(log_theta_wc)  # N * Cn
        l += np.sum(logsum((log_theta_xc + log_theta_c.T)))
    return l


def log_prob_hier(thetas_list, labeled_data_xit, labeled_data_deltas, unlabeled_data_xit=None):
    ls = []
    for i in range(len(thetas_list)):
        ls.append(log_prob(thetas_list[i], labeled_data_xit,
                           labeled_data_deltas[i], unlabeled_data_xit=unlabeled_data_xit))
    return ls


def log_prob_xin_pathscore(thetas_list, labeled_data_xit, labeled_data_deltas, unlabeled_data_xit=None, 
                           deltas=None, path_weights=None):
    l = 0.0
    if labeled_data_xit is not None:
        P_xcs_matrix = predict_prob_xin_pathscore(thetas_list, labeled_data_xit, deltas=deltas,
                                                  path_weights=path_weights)
        N = labeled_data_xit.shape[0]
        P_xcs_matrix = P_xcs_matrix.reshape(N, -1)
        labeled_delta_matrix = labeled_data_deltas[0]
        for depth in range(1, len(labeled_data_deltas)):
            labeled_delta_matrix = np.multiply(np.expand_dims(labeled_delta_matrix, axis=2),
                                               np.expand_dims(labeled_data_deltas[depth], axis=1))
            labeled_delta_matrix = labeled_delta_matrix.reshape((N, -1))
        l += np.sum(np.log(np.sum(P_xcs_matrix * labeled_delta_matrix, axis=1)))
    if unlabeled_data_xit is not None and unlabeled_data_xit.shape[0] > 0:
        P_xcs_matrix = predict_prob_xin_pathscore(thetas_list, unlabeled_data_xit, deltas=deltas,
                                                  path_weights=path_weights)
        N = unlabeled_data_xit[0].shape[0]
        P_xcs_matrix = P_xcs_matrix.reshape(N, -1)
        l += np.sum(np.log(np.sum(P_xcs_matrix, axis=1)))
    return l


def predict(thetas, data_xit):
    log_theta_wc = np.log(thetas[-1])  # V * Cn
    theta_c = np.expand_dims(thetas[0], axis=1)
    for depth in range(1, len(thetas) - 1):
        theta_c = np.dot(thetas[depth], theta_c)
    log_theta_c = np.log(theta_c)  # Cn * 1
    log_theta_xc = data_xit.dot(log_theta_wc)  # N * Cn
    return softmax(log_theta_xc + log_theta_c.T)


def predict_label(thetas, data_xit, P=None):
    if P is None:
        P = predict(thetas, data_xit)
    return np.argmax(P, axis=1)


def predict_multicomp(thetas, data_xit, deltas):
    log_theta_wc = np.log(thetas[-1])  # V * Cn
    theta_c = np.expand_dims(thetas[0], axis=1)
    for depth in range(1, len(thetas) - 1):
        theta_c = np.dot(thetas[depth] * deltas[depth - 1].T, theta_c)
    log_theta_c = np.log(theta_c)  # Cn * 1
    log_theta_xc = data_xit.dot(log_theta_wc)  # N * Cn
    P_xc = softmax(log_theta_xc + log_theta_c.T)  # N * Cn
    Ps = [P_xc]
    for depth in range(len(deltas) - 2, -1, -1):
        Ps.append(Ps[-1].dot(deltas[depth].T))
    Ps.reverse()
    return Ps


def predict_label_multicomp(thetas, data_xit, deltas):
    Ps = predict_multicomp(thetas, data_xit, deltas)
    y_pres = []
    for P in Ps:
        y_pres.append(np.argmax(P, axis=1))
    return y_pres


def predict_hier(thetas_list, data_xit, deltas=None):
    P_xcs = []
    Ps = []
    classes_size = []
    N = data_xit.shape[0]
    for depth in range(len(thetas_list)):
        thetas = thetas_list[depth]
        log_theta_wc = np.log(thetas[-1])  # V * Cn
        theta_c = np.expand_dims(thetas[0], axis=1)
        for inner_depth in range(1, len(thetas) - 1):
            theta_c = np.dot(thetas[inner_depth], theta_c)
        log_theta_c = np.log(theta_c)  # Cn * 1
        log_theta_xc = data_xit.dot(log_theta_wc)  # N * Cn
        P_xc = softmax(log_theta_xc + log_theta_c.T)  # N * Cn
        P_xcs.append(P_xc)
        Ps.append(np.zeros(P_xc.shape))
        classes_size.append(P_xc.shape[1])
    if N > 0:
        P_xcs_matrix = P_xcs[0]
        for depth in range(1, len(P_xcs)):
            P_xcs_matrix = np.multiply(np.expand_dims(P_xcs_matrix, axis=2), np.expand_dims(P_xcs[depth], axis=1))
            P_xcs_matrix = P_xcs_matrix.reshape((N, -1))
        if deltas is not None:
            delta_matrix = deltas[0]
            for depth in range(1, len(deltas) - 1):
                delta_matrix = np.multiply(np.expand_dims(delta_matrix, axis=2), np.expand_dims(deltas[depth], axis=0))
                delta_matrix = delta_matrix.reshape((-1, deltas[depth].shape[-1]))
            delta_matrix = delta_matrix.reshape((1, -1))
            P_xcs_matrix = P_xcs_matrix * delta_matrix
        P_xcs_matrix = normalize(P_xcs_matrix, axis=1)
        P_xcs_matrix = P_xcs_matrix.reshape([N] + classes_size)  # N * C1 * C2 ... * Cn
        for i in range(N):
            max_index_list = np.unravel_index(np.argmax(P_xcs_matrix[i]), classes_size)
            for j in range(len(Ps)):
                Ps[j][i, max_index_list[j]] = 1
    return Ps


def predict_prob_xin_pathscore(thetas_list, data_xit, deltas=None, path_weights=None):
    P_xcs = []
    classes_size = []
    N = data_xit.shape[0]
    for depth in range(len(thetas_list)):
        thetas = thetas_list[depth]
        log_theta_wc = np.log(thetas[-1])  # V * Cn
        theta_c = np.expand_dims(thetas[0], axis=1)
        for inner_depth in range(1, len(thetas) - 1):
            theta_c = np.dot(thetas[inner_depth], theta_c)
        log_theta_c = np.log(theta_c)  # Cn * 1
        log_theta_xc = data_xit.dot(log_theta_wc)  # N * Cn
        P_xc = softmax(log_theta_xc + log_theta_c.T)  # N * Cn
        P_xcs.append(P_xc * path_weights[depth])
        classes_size.append(P_xc.shape[1])
    if N > 0:
        P_xcs_matrix = P_xcs[0]
        for depth in range(1, len(P_xcs)):
            P_xcs_matrix = np.add(np.expand_dims(P_xcs_matrix, axis=2), np.expand_dims(P_xcs[depth], axis=1))
            P_xcs_matrix = P_xcs_matrix.reshape((N, -1))
        if deltas is not None:
            delta_matrix = deltas[0]
            for depth in range(1, len(deltas) - 1):
                delta_matrix = np.multiply(np.expand_dims(delta_matrix, axis=2), np.expand_dims(deltas[depth], axis=0))
                delta_matrix = delta_matrix.reshape((-1, deltas[depth].shape[-1]))
            delta_matrix = delta_matrix.reshape((1, -1))
            P_xcs_matrix = P_xcs_matrix * delta_matrix
        P_xcs_matrix = normalize(P_xcs_matrix, axis=1)
        P_xcs_matrix = P_xcs_matrix.reshape([N] + classes_size)  # N * C1 * C2 ... * Cn
    else:
        P_xcs_matrix = np.zeros([0] + classes_size)
    return P_xcs_matrix


def predict_xin_pathscore(thetas_list, data_xit, deltas=None, path_weights=None):
    P_xcs_matrix = predict_prob_xin_pathscore(thetas_list, data_xit, deltas=deltas, path_weights=path_weights)
    N = P_xcs_matrix.shape[0]
    classes_size = P_xcs_matrix.shape[1:]
    Ps = []
    for class_size in classes_size:
        Ps.append(np.zeros((N, class_size)))
    for i in range(N):
        max_index_list = np.unravel_index(np.argmax(P_xcs_matrix[i]), classes_size)
        for j in range(len(Ps)):
            Ps[j][i, max_index_list[j]] = 1
    return Ps


def predict_huiru_pathscore(thetas, data_xit, deltas):
    Ps = []
    P_bottom = predict(thetas, data_xit)
    Ps.append(P_bottom)
    for depth in range(len(deltas) - 2, -1, -1):
        P_high = normalize(np.dot(P_bottom, deltas[depth].T), axis=1)
        Ps.append(P_high)
        P_bottom = P_high
    Ps.reverse()
    return Ps


def predict_label_hier(thetas_list, data_xit, deltas=None):
    Ps = predict_hier(thetas_list, data_xit, deltas)
    y_pres = []
    for i in range(len(Ps)):
        y_pres.append(np.argmax(Ps[i], axis=1))
    return y_pres


def predict_label_xin_pathscore(thetas_list, data_xit, deltas=None, path_weights=None):
    data_xit_csr = data_xit.tocsr()
    y_pres = [None for k in range(len(thetas_list))]
    batch_size = 512
    for i in range(0, data_xit.shape[0], batch_size):
        j = min(i + batch_size, data_xit.shape[0])
        Ps = predict_xin_pathscore(thetas_list, data_xit_csr[i:j, :], deltas=deltas, path_weights=path_weights)
        for k in range(len(Ps)):
            if y_pres[k] is None:
                y_pres[k] = np.argmax(Ps[k], axis=1)
            else:
                y_pres[k] = np.concatenate([y_pres[k], np.argmax(Ps[k], axis=1)])
    return y_pres


def predict_label_huiru_pathscore(thetas, data_xit, deltas):
    Ps = predict_huiru_pathscore(thetas, data_xit, deltas)
    y_pres = []
    y_bottom = np.argmax(Ps[-1], axis=1)
    y_pres.append(y_bottom)
    for depth in range(len(deltas) - 2, -1, -1):
        y_high = [np.argmax(deltas[depth][:, j]) for j in y_bottom]
        y_bottom = y_high
        y_pres.append(y_bottom)
    y_pres.reverse()
    return y_pres


def run_check_similarity(data_managers):
    logger = logging.getLogger(__name__)
    model_name = "check_similarity"
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))
    test_pres = []
    for depth in range(len(data_managers[2].labels)):
        test_pres.append(np.argmax(data_managers[2].sims[depth], axis=1))
    return [], test_pres


def run_flatNB(data_managers, method='labeled'):
    logger = logging.getLogger(__name__)
    model_name = 'flatNB_' + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sim = data_managers[0].deltas[-1]
    elif 'dataless' in method:
        if settings.soft_sim:
            sim = normalize(data_managers[0].sims[-1], axis=1)
        else:
            sim = hardmax(data_managers[0].sims[-1], axis=1)
    else:
        raise NotImplementedError

    start = time.time()
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    # thetas = train_NB(data_managers[0].xit[:,non_zero_columns], sim)
    thetas = train_NB(data_managers[0].xit, sim)
    thetas = list(map(lambda theta: normalize_theta(theta, axis=0), thetas))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    # y_pre = predict_label(thetas, data_managers[2].xit[:,non_zero_columns])
    y_pre = predict_label(thetas, data_managers[2].xit)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas, y_pre


def run_flatEM(data_managers, method='labeled'):
    logger = logging.getLogger(__name__)
    model_name = 'flatEM_' + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sim = data_managers[0].deltas[-1]
    elif 'dataless' in method:
        if settings.soft_sim:
            sim = normalize(data_managers[0].sims[-1], axis=1)
        else:
            sim = hardmax(data_managers[0].sims[-1], axis=1)
    else:
        raise NotImplementedError

    start = time.time()
    thetas = train_EM(data_managers[0].xit, sim, data_managers[1].xit)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    y_pre = predict_label(thetas, data_managers[2].xit)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas, y_pre


def run_levelNB(data_managers, method='labeled'):
    logger = logging.getLogger(__name__)
    model_name = 'levelNB' + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    thetas_list = []
    y_pres = []
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    start = time.time()
    max_depth = len(sims)
    for depth in range(max_depth):
        # non_zero_indices = np.nonzero(data_managers[0].xit)
        # non_zero_columns = sorted(set(non_zero_indices[1]))
        # thetas = train_NB(data_managers[0].xit[:,non_zero_columns], sims[depth])
        thetas = train_NB(data_managers[0].xit, sims[depth])
        thetas = list(map(lambda theta: normalize_theta(theta, axis=0), thetas))
        thetas_list.append(thetas)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    for depth in range(max_depth):
        # y_pre = predict_label(thetas_list[depth], data_managers[2].xit[:,non_zero_columns])
        y_pre = predict_label(thetas_list[depth], data_managers[2].xit)
        y_pres.append(y_pre)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas_list, y_pres


def run_levelEM(data_managers, method='labeled'):
    logger = logging.getLogger(__name__)
    model_name = 'levelEM' + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    thetas_list = []
    y_pres = []
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    start = time.time()
    max_depth = len(sims)
    for depth in range(max_depth):
        thetas = train_EM(data_managers[0].xit, sims[depth], data_managers[1].xit)
        thetas_list.append(thetas)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    for depth in range(max_depth):
        y_pre = predict_label(thetas_list[depth], data_managers[2].xit)
        y_pres.append(y_pre)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas_list, y_pres


def run_hierNB(data_managers, deltas, method='labeled', soft_hier=True):
    logger = logging.getLogger(__name__)
    model_name = "hierNB_" + ("soft" if soft_hier else "hard") + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    start = time.time()
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    # thetas_list = train_hierNB(data_managers[0].xit[:,non_zero_columns], sims)
    thetas_list = train_hierNB(data_managers[0].xit, sims)
    thetas_list = list(map(lambda thetas: list(map(lambda theta: normalize_theta(theta, axis=0), thetas)), thetas_list))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    # test_pres = predict_label_hier(thetas_list, data_managers[2].xit[:,non_zero_columns], deltas=(None if soft_hier else deltas))
    test_pres = predict_label_hier(thetas_list, data_managers[2].xit, deltas=(None if soft_hier else deltas))
    logger.info("predicting time: " + str(time.time() - start))
    return thetas_list, test_pres


def run_hierEM(data_managers, deltas, method='labeled', soft_hier=True):
    logger = logging.getLogger(__name__)
    model_name = "hierEM_" + ("soft" if soft_hier else "hard") + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    start = time.time()
    thetas_list = train_hierEM(data_managers[0].xit, sims, data_managers[1].xit, 
                               deltas=(None if soft_hier else deltas))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    test_pres = predict_label_hier(thetas_list, data_managers[2].xit, deltas=(None if soft_hier else deltas))
    logger.info("predicting time: " + str(time.time() - start))
    return thetas_list, test_pres


def run_NBMC(data_managers, deltas, method='labeled'):
    logger = logging.getLogger(__name__)
    model_name = "NBMC_" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    # thetas = train_NBMC(data_managers[0].xit[:,non_zero_columns], sims)
    thetas = train_NBMC(data_managers[0].xit, sims)
    thetas = list(map(lambda theta: normalize_theta(theta, axis=0), thetas))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    # y_pres = predict_label_multicomp(thetas, data_managers[2].xit[:,non_zero_columns], deltas)
    y_pres = predict_label_multicomp(thetas, data_managers[2].xit, deltas)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas, y_pres


def run_EMMC(data_managers, deltas, method='labeled'):
    logger = logging.getLogger(__name__)
    model_name = "EMMC_" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))
    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    thetas = train_EMMC(data_managers[0].xit, sims, data_managers[1].xit, deltas)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    y_pres = predict_label_multicomp(thetas, data_managers[2].xit, deltas)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas, y_pres


def run_TDNB(data_managers, deltas, method='labeled'):
    logger = logging.getLogger(__name__)
    if 'BU' in method:
        model_name = 'BUNB_' + method
    else:
        model_name = 'TDNB_' + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        labels = data_managers[0].labels
    elif 'dataless' in method:
        labels = list(map(lambda sim: np.argmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    thetas_lists = []
    unlabeled_pres = []
    test_pres = []
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    data_managers_d0 = [
        # DataManager(data_managers[0].name + '_d0', xit=data_managers[0].xit[:,non_zero_columns], labels=data_managers[0].labels, 
        DataManager(data_managers[0].name + '_d0', xit=data_managers[0].xit, labels=data_managers[0].labels,
                    deltas=data_managers[0].deltas, sims=data_managers[0].sims, true_idx=None),
        None,
        # DataManager(data_managers[2].name + '_d0', xit=data_managers[2].xit[:,non_zero_columns], labels=data_managers[2].labels, 
        DataManager(data_managers[2].name + '_d0', xit=data_managers[2].xit, labels=data_managers[2].labels,
                    deltas=data_managers[2].deltas, sims=data_managers[2].sims, true_idx=None)]
    data_managers_list = [data_managers_d0]
    start = time.time()
    for depth in range(len(deltas)):
        thetas_list, unlabeled_pre, test_pre = train_one_depth(data_managers_list, depth, deltas, 'NB_' + method)
        thetas_lists.append(thetas_list)
        unlabeled_pres.append(unlabeled_pre)
        test_pres.append(test_pre)
        # prepare for the next depth
        if depth == len(deltas) - 1:
            break
        class_depth_no = deltas[depth].shape[0]
        labeled_true_idx_list = [[] for i in range(class_depth_no)]
        unlabeled_true_idx_list = [[] for i in range(class_depth_no)]
        test_true_idx_list = [[] for i in range(class_depth_no)]

        for i, l in enumerate(labels[depth]):
            labeled_true_idx_list[l].append(i)
        for i, u in enumerate(unlabeled_pre):
            unlabeled_true_idx_list[u].append(i)
        for i, t in enumerate(test_pre):
            test_true_idx_list[t].append(i)
        data_managers_list.clear()
        for i in range(class_depth_no):
            data_managers_list.append([
                build_subdata_manager(data_managers_d0[0], data_managers[0].name + '_d%d_c%d' % (depth, i),
                                      labeled_true_idx_list[i]),
                None,
                build_subdata_manager(data_managers_d0[2], data_managers[2].name + '_d%d_c%d' % (depth, i),
                                      test_true_idx_list[i])])
    logger.info("training and predicting time: " + str(time.time() - start))
    return thetas_lists, test_pres


def run_TDEM(data_managers, deltas, method='labeled'):
    logger = logging.getLogger(__name__)
    if 'BU' in method:
        model_name = 'BUEM_' + method
    else:
        model_name = 'TDEM_' + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        labels = data_managers[0].labels
    elif 'dataless' in method:
        labels = list(map(lambda sim: np.argmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError

    thetas_lists = []
    unlabeled_pres = []
    test_pres = []
    data_managers_list = [[
        DataManager(data_managers[0].name + '_d0', xit=data_managers[0].xit, labels=data_managers[0].labels,
                    deltas=data_managers[0].deltas, sims=data_managers[0].sims, true_idx=None),
        DataManager(data_managers[1].name + '_d0', xit=data_managers[1].xit, labels=data_managers[1].labels,
                    deltas=data_managers[1].deltas, sims=data_managers[1].sims, true_idx=None),
        DataManager(data_managers[2].name + '_d0', xit=data_managers[2].xit, labels=data_managers[2].labels,
                    deltas=data_managers[2].deltas, sims=data_managers[2].sims, true_idx=None)]]
    start = time.time()
    for depth in range(len(deltas)):
        thetas_list, unlabeled_pre, test_pre = train_one_depth(data_managers_list, depth, deltas, 'EM_' + method)
        thetas_lists.append(thetas_list)
        unlabeled_pres.append(unlabeled_pre)
        test_pres.append(test_pre)
        # prepare for the next depth
        if depth == len(deltas) - 1:
            break
        class_depth_no = deltas[depth].shape[0]
        labeled_true_idx_list = [[] for i in range(class_depth_no)]
        unlabeled_true_idx_list = [[] for i in range(class_depth_no)]
        test_true_idx_list = [[] for i in range(class_depth_no)]

        for i, l in enumerate(labels[depth]):
            labeled_true_idx_list[l].append(i)
        for i, u in enumerate(unlabeled_pre):
            unlabeled_true_idx_list[u].append(i)
        for i, t in enumerate(test_pre):
            test_true_idx_list[t].append(i)
        data_managers_list.clear()
        for i in range(class_depth_no):
            data_managers_list.append([
                build_subdata_manager(data_managers[0], data_managers[0].name + '_d%d_c%d' % (depth, i),
                                      labeled_true_idx_list[i]),
                build_subdata_manager(data_managers[1], data_managers[1].name + '_d%d_c%d' % (depth, i),
                                      unlabeled_true_idx_list[i]),
                build_subdata_manager(data_managers[2], data_managers[2].name + '_d%d_c%d' % (depth, i),
                                      test_true_idx_list[i])])
    logger.info("training and predicting time: " + str(time.time() - start))
    return thetas_lists, test_pres


def run_BUNB(data_managers, deltas, method='labeled'):
    reverse_data_managers = [
        build_reverse_data_manager(data_managers[0], data_managers[0].name + '_reversed'),
        None,
        build_reverse_data_manager(data_managers[2], data_managers[2].name + '_reversed')]
    reverse_deltas = list(map(lambda delta: delta.T, deltas[0:-1]))
    reverse_deltas.reverse()
    reverse_deltas.append(np.zeros((0, 0), dtype=np.int32))
    thetas_lists, test_pres = run_TDNB(
        reverse_data_managers, reverse_deltas, method='BUNB_' + method)
    return thetas_lists[::-1], test_pres[::-1]


def run_EM_bottomup(data_managers, deltas, method='labeled'):
    reverse_data_managers = list(map(lambda data_manager:
                                     build_reverse_data_manager(data_manager, data_manager.name + '_reversed'),
                                     data_managers))
    reverse_deltas = list(map(lambda delta: delta.T, deltas[0:-1]))
    reverse_deltas.reverse()
    reverse_deltas.append(np.zeros((0, 0), dtype=np.int32))
    thetas_lists, test_pres = run_TDEM(
        reverse_data_managers, reverse_deltas, method='BUNB_' + method)
    return thetas_lists[::-1], test_pres[::-1]


def run_WDNB(data_managers, deltas, method='labeled', soft_pathscore=True, path_weights=None):
    logger = logging.getLogger(__name__)
    model_name = 'WDNB_' + ("soft_" if soft_pathscore else "hard_") + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    # thetas_list = train_WDNB(data_managers[0].xit[:,non_zero_columns], sims)
    thetas_list = train_WDNB(data_managers[0].xit, sims)
    thetas_list = list(map(lambda thetas: list(map(lambda theta: normalize_theta(theta, axis=0), thetas)), thetas_list))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    # test_pres = predict_label_xin_pathscore(thetas_list, data_managers[2].xit[:,non_zero_columns], deltas=(None if soft_pathscore else deltas), path_weights=path_weights)
    test_pres = predict_label_xin_pathscore(thetas_list, data_managers[2].xit,
                                            deltas=(None if soft_pathscore else deltas), path_weights=path_weights)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas_list, test_pres


def run_WDEM(data_managers, deltas, method='labeled', soft_pathscore=True, path_weights=None):
    logger = logging.getLogger(__name__)
    model_name = 'WDEM_' + ("soft_" if soft_pathscore else "hard_") + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    thetas_list = train_WDEM(data_managers[0].xit, sims,
                                         data_managers[1].xit, deltas=(None if soft_pathscore else deltas),
                                         path_weights=path_weights)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    test_pres = predict_label_xin_pathscore(thetas_list, data_managers[2].xit,
                                            deltas=(None if soft_pathscore else deltas), path_weights=path_weights)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas_list, test_pres


def run_PCNB(data_managers, deltas, method='labeled', path_weights=None):
    logger = logging.getLogger(__name__)
    model_name = "PCNB_" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    path_score = compute_path_score(sims, deltas, path_weights=path_weights)
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    # thetas = train_NB(data_managers[0].xit[:,non_zero_columns], path_score)
    thetas = train_NB(data_managers[0].xit, path_score)
    thetas = list(map(lambda theta: normalize_theta(theta, axis=0), thetas))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    # test_pres = predict_label_huiru_pathscore(thetas, data_managers[2].xit[:,non_zero_columns], deltas)
    test_pres = predict_label_huiru_pathscore(thetas, data_managers[2].xit, deltas)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas, test_pres


def run_PCEM(data_managers, deltas, method='labeled', path_weights=None):
    logger = logging.getLogger(__name__)
    model_name = "PCEM_" + method
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    if 'labeled' in method:
        sims = data_managers[0].deltas
    elif 'dataless' in method:
        if settings.soft_sim:
            sims = list(map(lambda sim: normalize(sim, axis=1), data_managers[0].sims))
        else:
            sims = list(map(lambda sim: hardmax(sim, axis=1), data_managers[0].sims))
    else:
        raise NotImplementedError
    start = time.time()
    path_score = compute_path_score(sims, deltas, path_weights=path_weights)
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    thetas = train_PCEM(data_managers[0].xit, sims, data_managers[1].xit, deltas,
                                                   path_weights)
    # logger.info("Test results(iter, depth, l_diff, macro_f1, micro_f1):")
    # for i in range(0, len(iterdetails)):
    #     logger.info(str(iterdetails[i]))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    test_pres = predict_label_huiru_pathscore(thetas, data_managers[2].xit, deltas)
    logger.info("predicting time: " + str(time.time() - start))
    return thetas, test_pres


def run_classifiers(classifier_name, data_managers, method, **kw):
    if classifier_name == 'check_similarity':
        return run_check_similarity(data_managers)
    elif 'NB' in classifier_name:
        if 'labeled' in method:
            labels = check_labels(data_managers[0].labels[-1])
            if len(labels) == 1:
                return None, [labels[0]] * data_managers[2].xit.shape[0]
        elif 'dataless' in method and 'leaf' in classifier_name:
            labels = check_labels(np.argmax(data_managers[0].sims[-1], axis=1))
            if len(labels) == 1:
                return None, [labels[0]] * data_managers[2].xit.shape[0]
        if classifier_name == 'flatNB':
            return run_flatNB(data_managers, method=method)
        elif classifier_name == 'levelNB':
            return run_levelNB(data_managers, method=method)
        elif classifier_name == 'NBMC':
            return run_NBMC(data_managers, deltas=kw['deltas'], method=method)
        elif classifier_name == 'TDNB':
            return run_TDNB(data_managers, deltas=kw['deltas'], method=method)
        elif classifier_name == 'BUNB':
            return run_BUNB(data_managers, deltas=kw['deltas'], method=method)
        elif classifier_name == 'hierNB_soft':
            return run_hierNB(data_managers, deltas=kw['deltas'], soft_hier=True, method=method)
        elif classifier_name == 'hierNB_hard':
            return run_hierNB(data_managers, deltas=kw['deltas'], soft_hier=False, method=method)
        elif classifier_name == 'WDNB_soft':
            return run_WDNB(data_managers, deltas=kw['deltas'], soft_pathscore=True,
                                        path_weights=kw['path_weights'], method=method)
        elif classifier_name == 'WDNB_hard':
            return run_WDNB(data_managers, deltas=kw['deltas'], soft_pathscore=False,
                                        path_weights=kw['path_weights'], method=method)
        elif classifier_name == 'PCNB':
            return run_PCNB(data_managers, deltas=kw['deltas'], path_weights=kw['path_weights'],
                                          method=method)
    elif 'EM' in classifier_name:
        if classifier_name == 'flatEM':
            return run_flatEM(data_managers, method=method)
        elif classifier_name == 'levelEM':
            return run_levelEM(data_managers, method=method)
        elif classifier_name == 'EMMC':
            return run_EMMC(data_managers, deltas=kw['deltas'], method=method)
        elif classifier_name == 'TDEM':
            return run_TDEM(data_managers, deltas=kw['deltas'], method=method)
        elif classifier_name == 'BUEM':
            return run_EM_bottomup(data_managers, deltas=kw['deltas'], method=method)
        elif classifier_name == 'hierEM_soft':
            return run_hierEM(data_managers, deltas=kw['deltas'], soft_hier=True, method=method)
        elif classifier_name == 'hierEM_hard':
            return run_hierEM(data_managers, deltas=kw['deltas'], soft_hier=False, method=method)
        elif classifier_name == 'WDEM_soft':
            return run_WDEM(data_managers, deltas=kw['deltas'], soft_pathscore=True,
                                        path_weights=kw['path_weights'], method=method)
        elif classifier_name == 'WDEM_hard':
            return run_WDEM(data_managers, deltas=kw['deltas'], soft_pathscore=False,
                                        path_weights=kw['path_weights'], method=method)
        elif classifier_name == 'PCEM':
            return run_PCEM(data_managers, deltas=kw['deltas'], path_weights=kw['path_weights'],
                                          method=method)
    raise NotImplementedError('%s is not supported!' % (classifier_name))


def main(input_dir=settings.data_dir_20ng, label_ratio=0.1, times=1, classifier_names=None):
    logger = logging.getLogger(__name__)

    if label_ratio == 1.0:
        times = 1
    classes = tools.load(os.path.join(input_dir, settings.classes_file))
    deltas = tools.load(os.path.join(input_dir, settings.deltas_file))

    if not classifier_names:
        classifier_names = ['check_similarity',
            'flatNB', 'levelNB', 'NBMC', 'TDNB', 'BUNB', 'hierNB_soft', 'hierNB_hard', 'WDNB_hard', 'PCNB',
            'flatEM', 'levelEM', 'EMMC', 'TDEM', 'BUEM', 'hierEM_soft', 'hierEM_hard', 'WDEM_hard', 'PCEM']
    path_weights = [1.0]
    for i in range(1, len(classes)):
        path_weights.append(path_weights[-1] * settings.path_weight)
    nos, hier_tree = get_hier_info(input_dir)
    kw = {'deltas': deltas, 'path_weights': path_weights}
    if label_ratio == 1.0:
        times = 1
    for mode in ["labeled", "dataless"]:
        # times, methods, depth+1, [[(M_precision,m_precision), (M_recall,m_recall),  (M_f1, m_f1)], ...]
        metrics_result = np.zeros((times, 2, len(classifier_names), 3, 2))  

        for i in range(times):
            method_index = 0
            sub_dir = os.path.join(input_dir, str(label_ratio), str(i))
            logger.info(logconfig.key_log(logconfig.START_PROGRAM, sub_dir))

            data_managers = load_data_managers(sub_dir)

            if mode == "dataless" and np.max(data_managers[2].sims[0][0]) == 0.0:
                continue

            for j, classifier_name in enumerate(classifier_names):
                if label_ratio == 1.0 and classifier_name.startswith('EM'):
                    continue
                result = run_classifiers(classifier_name, data_managers, mode, **kw)
                if len(data_managers[2].labels) == len(result[1]):
                    metrics_result[i, 0, j] = compute_p_r_f1(data_managers[2].labels[-1], result[1][-1])
                    metrics_result[i, 1, j] = compute_overall_p_r_f1(data_managers[2].labels, result[1], nos)
                else:
                    metrics_result[i, 0, j] = compute_p_r_f1(data_managers[2].labels[-1], result[1])
                    metrics_result[i, 1, j] = compute_hier_p_r_f1(data_managers[2].labels[-1], result[1], nos,
                                                                  hier_tree)

        avg_M_metrics_result = np.mean(metrics_result[:, :, :, :, 0], axis=0)
        std_M_metrics_result = np.std(metrics_result[:, :, :, :, 0], axis=0)
        avg_m_metrics_result = np.mean(metrics_result[:, :, :, :, 1], axis=0)
        std_m_metrics_result = np.std(metrics_result[:, :, :, :, 1], axis=0)

        with open(os.path.join(input_dir, str(label_ratio), 'NB_EM_%s.csv' % (mode)), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Leaf'] + classifier_names)
            csv_writer.writerow(['Macro precision avg'] + list(avg_M_metrics_result[0, :, 0]))
            csv_writer.writerow(['Macro precision std'] + list(std_M_metrics_result[0, :, 0]))
            csv_writer.writerow(['Micro precision avg'] + list(avg_m_metrics_result[0, :, 0]))
            csv_writer.writerow(['Micro precision std'] + list(std_m_metrics_result[0, :, 0]))
            csv_writer.writerow(['Macro recall avg'] + list(avg_M_metrics_result[0, :, 1]))
            csv_writer.writerow(['Macro recall std'] + list(std_M_metrics_result[0, :, 1]))
            csv_writer.writerow(['Micro recall avg'] + list(avg_m_metrics_result[0, :, 1]))
            csv_writer.writerow(['Micro recall std'] + list(std_m_metrics_result[0, :, 1]))
            csv_writer.writerow(['Macro f1 avg'] + list(avg_M_metrics_result[0, :, 2]))
            csv_writer.writerow(['Macro f1 std'] + list(std_M_metrics_result[0, :, 2]))
            csv_writer.writerow(['Micro f1 avg'] + list(avg_m_metrics_result[0, :, 2]))
            csv_writer.writerow(['Micro f1 std'] + list(std_m_metrics_result[0, :, 2]))
            csv_writer.writerow([])
            csv_writer.writerow(['Overall'] + classifier_names)
            csv_writer.writerow(['Macro precision avg'] + list(avg_M_metrics_result[1, :, 0]))
            csv_writer.writerow(['Macro precision std'] + list(std_M_metrics_result[1, :, 0]))
            csv_writer.writerow(['Micro precision avg'] + list(avg_m_metrics_result[1, :, 0]))
            csv_writer.writerow(['Micro precision std'] + list(std_m_metrics_result[1, :, 0]))
            csv_writer.writerow(['Macro recall avg'] + list(avg_M_metrics_result[1, :, 1]))
            csv_writer.writerow(['Macro recall std'] + list(std_M_metrics_result[1, :, 1]))
            csv_writer.writerow(['Micro recall avg'] + list(avg_m_metrics_result[1, :, 1]))
            csv_writer.writerow(['Micro recall std'] + list(std_m_metrics_result[1, :, 1]))
            csv_writer.writerow(['Macro f1 avg'] + list(avg_M_metrics_result[1, :, 2]))
            csv_writer.writerow(['Macro f1 std'] + list(std_M_metrics_result[1, :, 2]))
            csv_writer.writerow(['Micro f1 avg'] + list(avg_m_metrics_result[1, :, 2]))
            csv_writer.writerow(['Micro f1 std'] + list(std_m_metrics_result[1, :, 2]))
            csv_writer.writerow([])
    logger.info(logconfig.key_log(logconfig.END_PROGRAM, input_dir))


if __name__ == "__main__":
    log_filename = os.path.join(settings.log_dir, 'NB_EM.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))

    classifier_names = [
        'flatNB', 'levelNB', 'NBMC', 'TDNB', 'WDNB_hard', 'PCNB',
        'flatEM', 'levelEM', 'EMMC', 'TDEM', 'WDEM_hard', 'PCEM']

    pool = Pool()
    for input_dir in settings.data_dirs:
        for label_ratio in settings.label_ratios:
            # pool.apply_async(main, args=(input_dir, label_ratio, settings.times, classifier_names))
            main(input_dir, label_ratio, settings.times, classifier_names)
    pool.close()
    pool.join()