import os
import logging

def _ensure_dir(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception:
            pass

def _set_logger(label=None, outdir=None, level='INFO', silence=True):

    if label == None:
        label = 'zflows'

    if level.upper() == 'DEBUG':
        datefmt = '%m-%d-%Y %H:%M:%S'
        fmt     = '[{}] [%(asctime)s.%(msecs)04d] %(message)s'.format(label)
    else:
        datefmt = '%m-%d-%Y %H:%M'
        fmt     = '[{}] [%(asctime)s] %(message)s'.format(label)

    # initialize logger
    logger = logging.getLogger(label)
    logger.propagate = False
    logger.setLevel(('{}'.format(level)).upper())

    # set streamhandler
    if not silence:
        if any([type(h) == logging.StreamHandler for h in logger.handlers]) is False:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            stream_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(stream_handler)

    # set filehandler
    if outdir != None:
        if any([type(h) == logging.FileHandler for h in logger.handlers]) is False:
            log_file = '{}/{}.log'.format(outdir, label)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            file_handler.setLevel(('{}'.format(level)).upper())
            logger.addHandler(file_handler)

    return logger
