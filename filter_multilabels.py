import logging
import logging.config
import logconfig
import os
import settings
from collections import defaultdict, Counter


def filter_multilabels(input_dir):
    logger = logging.getLogger(__name__)
    logger.info(logconfig.key_log(logconfig.DATA_NAME, input_dir))
    
    paths = []
    for file_name in os.listdir(input_dir):
        if os.path.splitext(file_name)[-1].startswith('.depth'):
            paths.append(os.path.join(input_dir, file_name))
    paths.sort()

    valid_id_counter = Counter()
    for depth in range(len(paths)):
        doc_topic_id = defaultdict(lambda: defaultdict(lambda: set())) # doc_topic[i][j] means a set about a document with [doc_text i] and [topic j] 
        with open(paths[depth], 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.strip()
                if line:
                    line_sp = line.split('\t')
                    topics = line_sp[2].split(';')
                    if len(topics) == 2: # an empty str will be at the last
                        doc_topic_id[line_sp[1]][topics[0]].add(line)
                line = f.readline()
        with open(paths[depth] + '.filtered', 'w', encoding='utf-8') as f:
            for doc, y in doc_topic_id.items():
                # multi-label
                if len(y) > 1:
                    continue
                for xx, yy in y.items():
                    # just keep one document
                    lines = sorted(list(yy))
                    line = lines[0]
                    doc_id = line.split('\t', 1)[0]
                    if depth == 0 or (valid_id_counter[doc_id] & (1 << (depth-1))):
                        valid_id_counter[doc_id] += (1 << depth)
                        f.write(line)
                        f.write('\n')
                    break
        logger.info(logconfig.key_log(logconfig.DEPTH, str(depth)))

if __name__ == '__main__':
    log_filename = os.path.join(settings.log_dir, 'filter_multilabels.log')
    logconfig.logging.config.dictConfig(logconfig.logging_config_dict('INFO', log_filename))
    for input_dir in settings.input_dirs:
        filter_multilabels(input_dir)
    