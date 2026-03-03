import logging
import sys
import os

def setup_logger(name="logger", log_file="outputs/train.log", level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)        
    else:
        handler = logging.StreamHandler(sys.stdout)
        
    handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(handler)
    if log_file:
         logger.addHandler(console_handler)

    return logger
