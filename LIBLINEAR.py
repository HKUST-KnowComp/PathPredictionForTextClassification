import logging
import logging.config
import logconfig
import settings
import time
import tools
import os
import numpy as np
import csv
from multiprocessing import Pool, Process
from build_data_managers import DataManager
from util import *
import time

def train_HierCost(input_dir, num_features, classifier_names, mode, output_dir=None):
    logger = logging.getLogger(__name__)

    if output_dir is None:
        output_dir = input_dir
    hierarchy_file = os.path.join(input_dir, settings.cat_hier_file)
    data_files = {
        'labeled': os.path.join(input_dir, settings.labeled_svmlight_file),
        'dataless': os.path.join(input_dir, settings.dataless_svmlight_file)}
    model_dirs = []
    for classifier_name in classifier_names:
        assert mode in data_files
        data_file = data_files[mode]
        model_dir = os.path.join(output_dir, "%s_%s.model" % (classifier_name, mode))
        log_file = os.path.join(settings.log_dir, '%s_%s.log' % (classifier_name, mode))
        model_dirs.append(model_dir)

        if os.path.exists(data_file) and not os.path.exists(model_dir):
            if "LIBLINEAR_LR_primal" in classifier_name:
                # command = "liblinear/train -s 0 %s %s >> %s" % (data_file, model_dir, log_file) # single-core
                command = "liblinear/train -n 10 -s 0 %s %s > %s" % (data_file, model_dir, log_file) # multi-core
            elif "LIBLINEAR_LR_dual" in classifier_name:
                # command = "liblinear/train -s 7 %s %s >> %s" % (data_file, model_dir, log_file) # single-core
                command = "liblinear/train -n 10 -s 7 %s %s > %s" % (data_file, model_dir, log_file) # multi-core
            elif "LIBLINEAR_SVC_primal" in classifier_name:
                # command = "liblinear/train -s 2 %s %s >> %s" % (data_file, model_dir, log_file) # single-core
                command = "liblinear/train -n 10 -s 2 %s %s > %s" % (data_file, model_dir, log_file) # multi-core
            elif "LIBLINEAR_SVC_dual" in classifier_name:
                # command = "liblinear/train -s 1 %s %s >> %s" % (data_file, model_dir, log_file) # single-core
                command = "liblinear/train -n 10 -s 1 %s %s > %s" % (data_file, model_dir, log_file) # multi-core
            else:
                raise NotImplementedError
            logger.info(command)
            os.system(command)
    return model_dirs

def predict_HierCost(input_dir, model_dirs, num_features, nos, hier_tree, output_dir=None):
    logger = logging.getLogger(__name__)

    if output_dir is None:
        output_dir = input_dir
    hierarchy_file = os.path.join(input_dir, settings.cat_hier_file)
    test_file = os.path.join(input_dir, settings.test_svmlight_file)
    pred_files = list(map(lambda model_dir: model_dir.replace("model", "pred"), model_dirs))
    metrics_files = list(map(lambda model_dir: model_dir.replace("model", "metrics"), model_dirs))
    for i in range(len(model_dirs)):
        if os.path.exists(model_dirs[i]):
            if not os.path.exists(pred_files[i]):
                command = "liblinear/predict %s %s %s > %s" % (
                    test_file, model_dirs[i], pred_files[i], metrics_files[i])
                logger.info(command)
                os.system(command)

    leaf_metrics_list = []
    overall_metrics_list = []
    for pred_file in pred_files:
        leaf_metrics = np.zeros((3,2), dtype=np.float32)
        overall_metrics = np.zeros((3,2), dtype=np.float32)
        if os.path.exists(pred_file):
            try:
                with open(pred_file, 'r') as f:
                    y_pre = list(map(lambda x: int(x), f.readlines()))
                with open(test_file, 'r') as f:
                    y_true = []
                    line = f.readline()
                    while line:
                        line = line.split(' ', 1)[0]
                        y_true.append(int(line))
                        line = f.readline()
                leaf_metrics = compute_p_r_f1(y_true, y_pre)
                overall_metrics = compute_hier_p_r_f1(y_true, y_pre, nos, hier_tree)
            except Exception as e:
                print(e)
                raise e
        leaf_metrics_list.append(leaf_metrics)
        overall_metrics_list.append(overall_metrics)
    return np.array([leaf_metrics_list, overall_metrics_list])

def main(input_dir=settings.data_dir_20ng, label_ratio=0.1, times=1, classifier_names=None):
    logger = logging.getLogger(__name__)
    
    vocab_info = tools.load(os.path.join(input_dir, settings.vocab_file))
    vocab_size = len(vocab_info["vocab_dict"])
    del vocab_info
    nos, hier_tree = get_hier_info(input_dir)
    if not classifier_names:
        classifier_names = ['LIBLINEAR_LR_primal', 'LIBLINEAR_LR_dual', 'LIBLINEAR_SVC_primal', 'LIBLINEAR_SVC_dual']
    if label_ratio == 1.0:
        times = 1
    for mode in ["labeled", "dataless"]:
        metrics_result = np.zeros((times, 2, len(classifier_names), 3, 2)) # times, methods, [[(M_precision,m_precision), (M_recall,m_recall),  (M_f1, m_f1)], ...]
        
        for i in range(times):
            sub_dir = os.path.join(input_dir, str(label_ratio), str(i))
            logger.info(logconfig.key_log(logconfig.START_PROGRAM, sub_dir))
            model_dirs = train_HierCost(sub_dir, vocab_size, classifier_names, mode)
            metrics_list = predict_HierCost(sub_dir, model_dirs, vocab_size, nos, hier_tree)
            metrics_result[i] = metrics_list
        avg_M_metrics_result = np.mean(metrics_result[:,:,:,:,0], axis=0)
        std_M_metrics_result = np.std(metrics_result[:,:,:,:,0], axis=0)
        avg_m_metrics_result = np.mean(metrics_result[:,:,:,:,1], axis=0)
        std_m_metrics_result = np.std(metrics_result[:,:,:,:,1], axis=0)
        with open(os.path.join(input_dir, str(label_ratio), 'LIBLINEAR_%s.csv' % (mode)), 'w') as f:
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
    if not os.path.exists('liblinear'):
        raise FileNotFoundError('Please download the liblinar tool and put the folder here or create a soft link!')
    log_filename = os.path.join(settings.log_dir, 'LIBLINEAR.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))

    pool = Pool(10)
    classifier_names = ['LIBLINEAR_LR_primal', 'LIBLINEAR_LR_dual', 'LIBLINEAR_SVC_primal', 'LIBLINEAR_SVC_dual']
    for input_dir in settings.data_dirs:
        for label_ratio in settings.label_ratios:
            # pool.apply_async(main, args=(input_dir, label_ratio, settings.times, classifier_names))
            main(input_dir, label_ratio, settings.times, classifier_names)
    pool.close()
    pool.join()
     
