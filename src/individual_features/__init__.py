# if appear_feature is imported as a module, then logger will be effective.
# Otherwise, if appear_feature is ran as a script, logger will not be recognized
import logging
import os
import coloredlogs
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get('LOGLEVEL', logging.DEBUG))
# must have a handler, otherwise logging will use lastresort
c_handler = logging.StreamHandler()
LOGFORMAT = 'aa %(name)s - %(levelname)s - %(message)s'
# c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
# c_handler.setFormatter(coloredlogs.ColoredFormatter(LOGFORMAT))
c_handler.setFormatter(logging.Formatter(LOGFORMAT))
logger.addHandler(c_handler)
