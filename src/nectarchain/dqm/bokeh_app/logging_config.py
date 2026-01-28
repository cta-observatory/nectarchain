import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(logger_name="nectarchain.dqm.bokeh_app", log_level=logging.INFO):
    """Set up a rotating file logger for the Bokeh app

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger, by default 'nectarchain.dqm.bokeh_app'
    log_level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG), by default logging.INFO

    Returns
    -------
    logging.Logger
        Configured logger
    """

    # FIXME: temporary dir for the Bokeh app logs is
    # src/nectarchain/dqm/bokeh_app/logs. We may need something better and cleaner.
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "app.log")

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if the function is called multiple times
    if logger.handlers:
        return logger

    # Rotating file handler for when the app will be running continously
    handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB per log file
        backupCount=3,  # Keep up to 3 backup log files
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(funcName)s (line %(lineno)d) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
