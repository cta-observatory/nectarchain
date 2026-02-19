# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module handle the logging messages for the Bokeh webpage for the RTA of NectarCAM.
"""

# imports
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ============================================================
# Setup logging
# ============================================================
def setup_logging(log_dir, log_file_name="bokeh_log.log", level=logging.INFO):
    """
    Configure logging for console and file output.

    Parameters
    ----------
    log_dir : str
        Directory where log files are stored.
    log_file_name : str, optional
        Name of the file where to save logs.
    level : int, optional
        Logging level (logging.INFO, logging.DEBUG, etc.)
    """

    os.makedirs(log_dir, exist_ok=True)

    log_file = Path(log_dir) / log_file_name

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Prevent duplicate handlers if re-run
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S"
    )
    console_handler.setFormatter(console_format)

    # Rotating file handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
