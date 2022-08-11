import mlflow

import contextlib
import inspect
import logging
import os

@contextlib.contextmanager
def logfile_artifact(fmt="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO, header=(30 * "=")):
    file_logger = logging.FileHandler("experiment.log")
    file_logger.setLevel(level)
    file_logger.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(file_logger)
    logging.info(header + "Start Run" + header)
    try:
        yield file_logger
        logging.info(header + " End Run " + header)
    finally:
        logging.getLogger().removeHandler(file_logger)
        mlflow.log_artifact(file_logger.baseFilename)
        os.remove(file_logger.baseFilename)

def log_arguments(func):
    """ Decorator to log all arguments of a function with mlflow. """
    func_signature = inspect.signature(func)
    def decorated_func(*args, **kwargs):
        signature_binded = func_signature.bind(*args, **kwargs)
        signature_binded.apply_defaults()
        mlflow.log_params(signature_binded.arguments)
        return func(*args, **kwargs)
    return decorated_func
    
