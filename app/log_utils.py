import sys
import os
import logging
import logging.handlers

logger = None

def init_logging(appname:str, filename: str):

    global logger
    logger = logging.getLogger(appname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(module)-16s - %(levelname)-4s %(message)s', '%d-%m-%Y %H:%M:%S')

    logfilename = os.path.join(os.path.dirname(sys.modules[__name__].__file__), filename)
    handlers = [
        logging.handlers.RotatingFileHandler(logfilename, maxBytes=20485760, backupCount=2, encoding='utf-8'),
        logging.StreamHandler(),
    ]

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    werkzeug = logging.getLogger('werkzeug')
    if werkzeug:
        logger.warning('Redirecting werkzeug too')
        werkzeug.handlers = handlers
