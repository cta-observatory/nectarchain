# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module handle argument parsing of the main() of the Bokeh webpage for the RTA of NectarCAM.
"""


# imports
import sys
import json
import logging
from pathlib import Path

import numpy as np


# Bokeh imports
from bokeh.io import curdoc


# Bokeh RTA imports
from .high_level_builders import build_ui
from .data_fetch_helpers import _get_latest_file
from .update_helpers import (
    periodic_update_display,
    start_periodic_updates
)
from .logging_config import setup_logging


# ============================================================
# Logging configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================
# Main function (with CLI argument parsing)
# ============================================================
def create_app(doc):
    """
    Create the Bokeh document to be sent to the Bokeh server
    and add it to the existing Bokeh document ``doc``.
    
    Parameters
    ----------
    doc: Bokeh Document
        Document to send to the server for the interface.
    """

    # Root of the project
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Log directory initialization
    logger = setup_logging(
        log_dir=PROJECT_ROOT / "log_dir",
        log_file_name="bokeh_log.log",
        level=logging.INFO
    )

    test_interface = "test-interface" in set(sys.argv[1:])

    # JSON constants
    with open(PROJECT_ROOT / "utils/static/constants.json") as constants_file:
        json_dict = json.load(constants_file)
    REAL_TIME_TAG = json_dict["REAL_TIME_TAG"]
    DEFAULT_UPDATE_MS = json_dict["DEFAULT_UPDATE_MS"]
    DEFAULT_EXTENSION = json_dict["DEFAULT_EXTENSION"]
    time_parentkeys = json_dict["time_parentkeys"]
    time_childkeys = json_dict["time_childkeys"]

    # Path of data
    if test_interface:
        logger.info("Test interface - displaying example runs")
        RESSOURCE_PATH = PROJECT_ROOT / json_dict["EXAMPLE_RESSOURCE_PATH"]
    else:
        logger.info("Real interface - fetching data currently produced by RTA")
        RESSOURCE_PATH = PROJECT_ROOT / json_dict["RESSOURCE_PATH"]

    # Bokeh item storages
    display_registry = []
    widgets = {"PERIODIC_CB_ID": None}

    # Retrieve latest file to simulate real time data
    # Will change when we add the stream listening part
    file = _get_latest_file(RESSOURCE_PATH)

    # make_body() default kwargs
    # Keep statistic functions for timelines only if in the Numpy module
    with open(PROJECT_ROOT / "utils/static/make_body_default_kwargs.json") as make_body_default_kwargs_file:
        make_body_kwargs = json.load(make_body_default_kwargs_file)
        numpy_funcs = {}
        numpy_func_names = {}
        for func_key in make_body_kwargs["func_timeline"].keys():
            try:
                np_func = getattr(np, make_body_kwargs["func_timeline"][func_key])
                numpy_funcs[func_key] = np_func
                numpy_func_names[func_key] = make_body_kwargs["label_2d_timeline"][func_key]
            except Exception as e:
                logger.warning(f"Fail to get function from Numpy module: {e}")
        make_body_kwargs["func_timeline"] = numpy_funcs
        make_body_kwargs["label_2d_timeline"] = numpy_func_names

    # Build UI
    root_layout, header_ret = build_ui(
        ressource_path = RESSOURCE_PATH,
        file = file,
        filepath = getattr(file, "filename", None),
        display_registry = display_registry,
        widgets = widgets,
        real_time_tag = REAL_TIME_TAG,
        default_update_ms = DEFAULT_UPDATE_MS,
        extension = DEFAULT_EXTENSION,
        time_parentkeys=time_parentkeys,
        time_childkeys=time_childkeys,
        **make_body_kwargs
    )
    doc.add_root(root_layout)
    header_ret = root_layout.children[0].children

    # Start real-time at launch
    try:
        periodic_update_display(file, display_registry, widgets, header_ret[1])
        widgets["PERIODIC_CB_ID"] = start_periodic_updates(
            file=file,
            display_registry=display_registry,
            widgets=widgets,
            status_col=header_ret[1],
            interval_ms=DEFAULT_UPDATE_MS
        )
    except Exception:
        pass

def main():
    """Generate the Bokeh interface from an empty Bokeh document."""
    create_app(curdoc())