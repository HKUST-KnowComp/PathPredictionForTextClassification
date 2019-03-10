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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer

def run_leaf(data_managers, deltas, method='LR_labeled', dual=True):
    logger = logging.getLogger(__name__)
    model_name = method + "_leaf"
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    model_list = []
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
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    if 'LR' in method:
        model = LogisticRegression(dual=dual, solver='liblinear')
    elif 'SVM' in method:
        model = LinearSVC(dual=dual)
    if 'tf-idf' in method:
        tf_idf = TfidfTransformer()
        # tf_idf.fit(data_managers[0].xit[:,non_zero_columns])
        tf_idf.fit(data_managers[0].xit)
        # model.fit(tf_idf.transform(data_managers[0].xit[:,non_zero_columns]), np.argmax(sims[-1], axis=1))
        model.fit(tf_idf.transform(data_managers[0].xit), np.argmax(sims[-1], axis=1))
    else:
        # model.fit(data_managers[0].xit[:,non_zero_columns], np.argmax(sims[-1], axis=1))
        model.fit(data_managers[0].xit, np.argmax(sims[-1], axis=1))
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    if 'tf-idf' in method:
        # y_pre = model.predict(tf_idf.transform(data_managers[2].xit[:, non_zero_columns]))
        y_pre = model.predict(tf_idf.transform(data_managers[2].xit))
    else:
        # y_pre = model.predict(data_managers[2].xit[:, non_zero_columns])
        y_pre = model.predict(data_managers[2].xit)
    logger.info("predicting time: " + str(time.time() - start))
    return model, y_pre

def run_level(data_managers, deltas, method='LR_labeled', dual=True):
    logger = logging.getLogger(__name__)
    model_name = method + "_level"
    logger.info(logconfig.key_log(logconfig.MODEL_NAME, model_name))

    model_list = []
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
    # non_zero_indices = np.nonzero(data_managers[0].xit)
    # non_zero_columns = sorted(set(non_zero_indices[1]))
    if 'tf-idf' in method:
        tf_idf = TfidfTransformer()
        # tf_idf.fit(data_managers[0].xit[:,non_zero_columns])
        tf_idf.fit(data_managers[0].xit)
    else:
        tf_idf = None
    for depth in range(max_depth):
        if 'LR' in method:
            model = LogisticRegression(dual=True, solver='liblinear')
        elif 'SVM' in method:
            model = LinearSVC(dual=dual)
        if 'tf-idf' in method:
            # model.fit(tf_idf.transform(data_managers[0].xit[:,non_zero_columns]), np.argmax(sims[depth], axis=1))
            model.fit(tf_idf.transform(data_managers[0].xit), np.argmax(sims[depth], axis=1))
        else:
            # model.fit(data_managers[0].xit[:,non_zero_columns], np.argmax(sims[depth], axis=1))
            model.fit(data_managers[0].xit, np.argmax(sims[depth], axis=1))
        model_list.append(model)
    logger.info("training time: " + str(time.time() - start))
    start = time.time()
    for depth in range(max_depth):
        if 'tf-idf' in method:
            # y_pre = model_list[depth].predict(tf_idf.transform(data_managers[2].xit[:, non_zero_columns]))
            y_pre = model_list[depth].predict(tf_idf.transform(data_managers[2].xit))
        else:
            # y_pre = model_list[depth].predict(data_managers[2].xit[:, non_zero_columns])
            y_pre = model_list[depth].predict(data_managers[2].xit)
        y_pres.append(y_pre)
    logger.info("predicting time: " + str(time.time() - start))
    return model_list, y_pres

def run_classifiers(classifier_name, data_managers, method, **kw):    
    if 'labeled' in method:
        labels = check_labels(data_managers[0].labels[-1])
        if len(labels) == 1:
            return None, [labels[0]] * data_managers[2].xit.shape[0]
    elif 'dataless' in method and 'leaf' in classifier_name:
        labels = check_labels(np.argmax(data_managers[0].sims[-1], axis=1))
        if len(labels) == 1:
            return None, [labels[0]] * data_managers[2].xit.shape[0]
    
    if 'LR' in classifier_name:
        if classifier_name == 'flatLR':
            return run_leaf(data_managers, kw['deltas'], method='LR_'+method, dual=kw['dual'])
        elif classifier_name == 'levelLR':
            return run_level(data_managers, kw['deltas'], method='LR_'+method, dual=kw['dual'])
    elif 'SVM' in classifier_name:
        if classifier_name == 'flatSVM':
            return run_leaf(data_managers, kw['deltas'], method='SVM_'+method, dual=kw['dual'])
        elif classifier_name == 'levelSVM':
            return run_level(data_managers, kw['deltas'], method='SVM_'+method, dual=kw['dual'])
    raise NotImplementedError('%s is not supported!' % (classifier_name))
        
def main(input_dir=settings.data_dir_20ng, label_ratio=0.1, times=1, classifier_names=None, dual=True):
    logger = logging.getLogger(__name__)

    if label_ratio == 1.0:
        times = 1
    classes = tools.load(os.path.join(input_dir, settings.classes_file))
    deltas = tools.load(os.path.join(input_dir, settings.deltas_file))
    kw = {'deltas': deltas, 'dual': dual}
    if not classifier_names:
        classifier_names = ['flatLR', 'levelLR',
                            'flatSVM', 'levelSVM']
    path_weights = [1.0]
    for i in range(1, len(classes)):
        path_weights.append(path_weights[-1] * settings.path_weight)
    if label_ratio == 1.0:
        times = 1
        
    nos, hier_tree = get_hier_info(input_dir)
    for mode in ["labeled", "dataless"]:
        metrics_result = np.zeros((times, 2, len(classifier_names), 3, 2)) # times, methods, depth+1, [[(M_precision,m_precision), (M_recall,m_recall),  (M_f1, m_f1)], ...]

        for i in range(times):
            method_index = 0
            sub_dir = os.path.join(input_dir, str(label_ratio), str(i))
            logger.info(logconfig.key_log(logconfig.START_PROGRAM, sub_dir))

            data_managers = load_data_managers(sub_dir)

            if mode == "dataless" and np.max(data_managers[2].sims[0][0]) == 0.0:
                continue

            for j, classifier_name in enumerate(classifier_names):
                result = run_classifiers(classifier_name, data_managers, mode, **kw)
                if len(data_managers[2].labels) == len(result[1]):
                    metrics_result[i,0,j] = compute_p_r_f1(data_managers[2].labels[-1], result[1][-1])
                    metrics_result[i,1,j] = compute_overall_p_r_f1(data_managers[2].labels, result[1], nos)
                else:
                    metrics_result[i,0,j] = compute_p_r_f1(data_managers[2].labels[-1], result[1])
                    metrics_result[i,1,j] = compute_hier_p_r_f1(data_managers[2].labels[-1], result[1], nos, hier_tree)

        avg_M_metrics_result = np.mean(metrics_result[:,:,:,:,0], axis=0)
        std_M_metrics_result = np.std(metrics_result[:,:,:,:,0], axis=0)
        avg_m_metrics_result = np.mean(metrics_result[:,:,:,:,1], axis=0)
        std_m_metrics_result = np.std(metrics_result[:,:,:,:,1], axis=0)
        
        with open(os.path.join(input_dir, str(label_ratio), 'LR_SVM_%s.csv' % (mode)), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Leaf'] + classifier_names)
            csv_writer.writerow(['Macro precision avg'] + list(avg_M_metrics_result[0,:,0]))
            csv_writer.writerow(['Macro precision std'] + list(std_M_metrics_result[0,:,0]))
            csv_writer.writerow(['Micro precision avg'] + list(avg_m_metrics_result[0,:,0]))
            csv_writer.writerow(['Micro precision std'] + list(std_m_metrics_result[0,:,0]))
            csv_writer.writerow(['Macro recall avg'] + list(avg_M_metrics_result[0,:,1]))
            csv_writer.writerow(['Macro recall std'] + list(std_M_metrics_result[0,:,1]))
            csv_writer.writerow(['Micro recall avg'] + list(avg_m_metrics_result[0,:,1]))
            csv_writer.writerow(['Micro recall std'] + list(std_m_metrics_result[0,:,1]))
            csv_writer.writerow(['Macro f1 avg'] + list(avg_M_metrics_result[0,:,2]))
            csv_writer.writerow(['Macro f1 std'] + list(std_M_metrics_result[0,:,2]))
            csv_writer.writerow(['Micro f1 avg'] + list(avg_m_metrics_result[0,:,2]))
            csv_writer.writerow(['Micro f1 std'] + list(std_m_metrics_result[0,:,2]))
            csv_writer.writerow([])
            csv_writer.writerow(['Overall'] + classifier_names)
            csv_writer.writerow(['Macro precision avg'] + list(avg_M_metrics_result[1,:,0]))
            csv_writer.writerow(['Macro precision std'] + list(std_M_metrics_result[1,:,0]))
            csv_writer.writerow(['Micro precision avg'] + list(avg_m_metrics_result[1,:,0]))
            csv_writer.writerow(['Micro precision std'] + list(std_m_metrics_result[1,:,0]))
            csv_writer.writerow(['Macro recall avg'] + list(avg_M_metrics_result[1,:,1]))
            csv_writer.writerow(['Macro recall std'] + list(std_M_metrics_result[1,:,1]))
            csv_writer.writerow(['Micro recall avg'] + list(avg_m_metrics_result[1,:,1]))
            csv_writer.writerow(['Micro recall std'] + list(std_m_metrics_result[1,:,1]))
            csv_writer.writerow(['Macro f1 avg'] + list(avg_M_metrics_result[1,:,2]))
            csv_writer.writerow(['Macro f1 std'] + list(std_M_metrics_result[1,:,2]))
            csv_writer.writerow(['Micro f1 avg'] + list(avg_m_metrics_result[1,:,2]))
            csv_writer.writerow(['Micro f1 std'] + list(std_m_metrics_result[1,:,2]))
            csv_writer.writerow([])
    logger.info(logconfig.key_log(logconfig.END_PROGRAM, input_dir))
    
if __name__ == "__main__":
    log_filename = os.path.join(settings.log_dir, 'LR_SVM.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))

    pool = Pool(10)
    for input_dir in settings.data_dirs:
        classifier_names = ['flatLR', 'levelLR',
                            'flatSVM', 'levelSVM']
        for label_ratio in settings.label_ratios:
            # pool.apply_async(main, args=(input_dir, label_ratio, settings.times, classifier_names))
            main(input_dir, label_ratio, settings.times, classifier_names)
    pool.close()
    pool.join()
