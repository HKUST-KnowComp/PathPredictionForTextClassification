import logging
import logging.config

def logging_config_dict(level='INFO', output_filename='./example.log'):
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'default': {
                'format': '[%(levelname)s - %(asctime)s - %(module)s] %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'default'
            },
            'file': {
                'level': level,
                'class': 'logging.FileHandler',
                'filename': output_filename,
                'mode': 'a',
                'formatter': 'default',

            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': level,
            },
        }
    }
    return LOGGING


START_PROGRAM = "StartProgram"
START = "Start"
MODEL_NAME = "ModelName"
FUNCTION_NAME = "FunctionName"
DATA_NAME = "DataName"
ITERATION = "Iteration"
DEPTH = "Depth"
EVAL_NAME = "EvalName"
EVAL_RESULT = "EvalResult"
END = "End"
END_PROGRAM = "EndProgram"


def key_log(key, value):
    return "${:s} {:s}".format(key, value)

