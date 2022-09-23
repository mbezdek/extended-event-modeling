import logging
import os
import coloredlogs
logger = logging.getLogger(__name__)
# logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
LOGFORMAT = '%(name)s - %(levelname)s - %(message)s'
# c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
# c_handler.setFormatter(coloredlogs.ColoredFormatter(LOGFORMAT))
c_handler.setFormatter(logging.Formatter(LOGFORMAT))
logger.addHandler(c_handler)
