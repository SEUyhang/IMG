import logging
import os

logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

fh = logging.FileHandler('test.log' )
fh.setLevel(logging.WARNING)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

formatter = logging.Formatter(fmt='%(asctime)s : %(levelname)s : %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.debug('this is debug')
logger.info('this is info')
logger.warning('this is warning')
logger.error('this is error')
