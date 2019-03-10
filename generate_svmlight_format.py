import logging
import logging.config
import logconfig
import numpy as np
import settings
import time
import tools
import os
import csv
import shutil
from multiprocessing import Pool
from build_data_managers import DataManager
from sklearn.datasets import dump_svmlight_file
from util import *


def generate_svmlight_format(data_managers, no_last, output_dir):
    labeled_file = os.path.join(output_dir, settings.labeled_svmlight_file)
    dataless_file = os.path.join(output_dir, settings.dataless_svmlight_file)
    test_file = os.path.join(output_dir, settings.test_svmlight_file)

    dump_svmlight_file(data_managers[0].xit, [no_last[x] for x in data_managers[0].labels[-1]], labeled_file, zero_based=False)
    if np.max(data_managers[0].sims[-1][0]) > 0:
        f_labeled = open(labeled_file, "r")
        f_dataless = open(dataless_file, "w")
        line = f_labeled.readline()
        i = 0
        while line:
            line = line.split(' ', 1)
            f_dataless.write(str(no_last[np.argmax(data_managers[0].sims[-1][i])]) + ' ' + line[1])
            line = f_labeled.readline()
            i += 1
        f_labeled.close()
        f_dataless.close()
    dump_svmlight_file(data_managers[2].xit, [no_last[x] for x in data_managers[2].labels[-1]], test_file, zero_based=False)

def main(input_dir=settings.data_dir_20ng, label_ratio=0.1, time=0, output_dir=None):
    logger = logging.getLogger(__name__)

    if output_dir is None:
        output_dir = input_dir
    tools.make_sure_path_exists(output_dir)
    deltas = tools.load(os.path.join(input_dir, settings.deltas_file))
    classes = tools.load(os.path.join(input_dir, settings.classes_file))
    nos, hier_tree = generate_hier_info(deltas, classes, input_dir)
    sub_dir = os.path.join(input_dir, str(label_ratio), str(time))
    output_dir = os.path.join(output_dir, str(label_ratio), str(time))
    tools.make_sure_path_exists(output_dir)

    logger.info(logconfig.key_log('Input dir', sub_dir))
    logger.info(logconfig.key_log('Output dir', output_dir))
    
    cat_hier_file = os.path.join(output_dir, settings.cat_hier_file)
    if not os.path.exists(cat_hier_file):
        shutil.copyfile(os.path.join(input_dir, settings.cat_hier_file), os.path.join(output_dir, settings.cat_hier_file))
    if os.path.exists(os.path.join(output_dir, settings.labeled_svmlight_file)) and \
        os.path.exists(os.path.join(output_dir, settings.dataless_svmlight_file)) and \
            os.path.exists(os.path.join(output_dir, settings.test_svmlight_file)):
        return
    data_managers = load_data_managers(sub_dir)
    generate_svmlight_format(data_managers, nos[-1], output_dir)

if __name__ == "__main__":
    log_filename = os.path.join(settings.log_dir, 'generate_svmlight_format.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))

    pool = Pool()
    for input_dir in settings.data_dirs:
        for label_ratio in settings.label_ratios:
            for time in range(settings.times):
                pool.apply_async(main, args=(input_dir, label_ratio, time))
                # main(input_dir, label_ratio, time)
                if label_ratio == 1.0:
                    break
    pool.close()
    pool.join()
