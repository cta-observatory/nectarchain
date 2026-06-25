"""
This module handle argument parsing of the main()
of the Bokeh webpage for the RTA of NectarCAM.
"""

import argparse
import json
import logging

# imports
import os
import shutil
import sys
from collections import deque
from pathlib import Path

import ctapipe
import numpy as np

# Bokeh imports
from bokeh.io import curdoc
from watchdog.observers import Observer

from .data_fetch_helpers import LatestFilesHandler, _get_latest_file
from .high_level_builders import build_ui
from .logging_config import setup_logging
from .update_helpers import periodic_update_display, start_periodic_updates

# Bokeh RTA imports
from .utils_helpers import hdf5Proxy

# ============================================================
# Logging configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ============================================================
# CLI argument parser
# ============================================================
def parse_server_cli():
    parser = argparse.ArgumentParser(
        prog="bokeh_app", description="RTA NectarCAM Bokeh interface"
    )

    parser.add_argument(
        "--test-interface",
        action="store_true",
        help="Run the app using example resource path instead of real-time data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where to load DL1 of the RTA.",
    )

    # parser.add_argument(
    #     "--debug",
    #     action="store_true",
    #     help="Enable debug logging"
    # )

    return parser.parse_args(sys.argv[1:])


# ============================================================
# Main function (with CLI argument parsing call)
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
        level=logging.INFO,
    )
    logger.info("Bokeh app started")

    # Check CLI
    SERVER_CONFIG = parse_server_cli()

    # JSON constants
    with open(PROJECT_ROOT / "utils/static/constants.json") as constants_file:
        json_dict = json.load(constants_file)
    REAL_TIME_TAG = json_dict["REAL_TIME_TAG"]
    DEFAULT_UPDATE_MS = json_dict["DEFAULT_UPDATE_MS"]
    DEFAULT_EXTENSION = json_dict["DEFAULT_EXTENSION"]
    MAX_READ_FILES = json_dict["MAX_READ_FILES"]
    time_parentkey = json_dict["time_parentkey"]
    time_childkey = json_dict["time_childkey"]
    group_parentkeys = json_dict["group_parentkeys"]

    # Path of data
    if SERVER_CONFIG.test_interface:
        # Test data import
        logger.info("Test interface - displaying example runs")
        RESOURCE_PATH = PROJECT_ROOT / json_dict["EXAMPLE_RESOURCE_PATH"]
        if os.path.exists(RESOURCE_PATH):
            shutil.rmtree(RESOURCE_PATH)
        os.makedirs(RESOURCE_PATH)
        for filename in json_dict["EXAMPLE_DATA_FILES"]:
            filepath = ctapipe.utils.get_dataset_path(
                filename,
                url="http://cccta-dataserver.in2p3.fr/data/ctapipe-test-data/v1.1.0",
            )
            shutil.move(filepath, os.path.join(RESOURCE_PATH, filename))

    else:
        # Real time start
        logger.info("Real interface - fetching data currently produced by RTA")
        RESOURCE_PATH = PROJECT_ROOT / json_dict["RESOURCE_PATH"]
        if SERVER_CONFIG.output_dir != "":
            RESOURCE_PATH = Path(SERVER_CONFIG.output_dir)
    # Scan of resource repository once
    latest_files = deque(
        sorted(Path(RESOURCE_PATH).glob("*.h5"), key=lambda p: p.stat().st_mtime)[
            -MAX_READ_FILES:
        ],
        maxlen=MAX_READ_FILES,
    )

    observer = Observer()
    observer.schedule(
        LatestFilesHandler(latest_files, MAX_READ_FILES), RESOURCE_PATH, recursive=False
    )
    observer.start()

    # Bokeh item storages
    display_registry = []
    widgets = {"PERIODIC_CB_ID": None}

    # Retrieve latest file to simulate real time data
    # Will change when we add the stream listening part
    with _get_latest_file(list(latest_files)) as fileHDF5:
        fileproxy = hdf5Proxy(fileHDF5)
    if time_parentkey is not None and time_childkey is not None:
        try:
            sort_indexes = fileproxy[time_parentkey].sort_from_key(time_childkey)
            for group_parentkey in group_parentkeys:
                fileproxy[group_parentkey].mask(sort_indexes)
        except Exception as e:
            logger.warning(f"create_app: failed data time sorting: {e}")

    # make_body() default kwargs
    # Keep statistic functions for timelines only if in the Numpy module
    with open(
        PROJECT_ROOT / "utils/static/make_body_default_kwargs.json"
    ) as make_body_default_kwargs_file:
        make_body_kwargs = json.load(make_body_default_kwargs_file)
        numpy_funcs = {}
        numpy_func_names = {}
        for func_key in make_body_kwargs["func_timeline"].keys():
            try:
                np_func = getattr(np, make_body_kwargs["func_timeline"][func_key])
                numpy_funcs[func_key] = np_func
                numpy_func_names[func_key] = make_body_kwargs["label_2d_timeline"][
                    func_key
                ]
            except Exception as e:
                logger.warning(f"Fail to get function from Numpy module: {e}")
        make_body_kwargs["func_timeline"] = numpy_funcs
        make_body_kwargs["label_2d_timeline"] = numpy_func_names

    # Build UI
    root_layout, header_ret = build_ui(
        file_list=list(latest_files),
        resource_path=RESOURCE_PATH,
        file=fileproxy,
        filepath=getattr(fileproxy, "filename", None),
        display_registry=display_registry,
        widgets=widgets,
        real_time_tag=REAL_TIME_TAG,
        default_update_ms=DEFAULT_UPDATE_MS,
        extension=DEFAULT_EXTENSION,
        sort_time_parentkey=time_parentkey,
        sort_time_childkey=time_childkey,
        group_parentkeys=group_parentkeys,
        **make_body_kwargs,
    )
    doc.add_root(root_layout)
    header_ret = root_layout.children[0].children

    # Start real-time at launch
    periodic_update_display(
        list(latest_files),
        display_registry,
        widgets,
        header_ret[1],
        time_parentkey=time_parentkey,
        time_childkey=time_childkey,
    )
    start_periodic_updates(
        file_list=list(latest_files),
        resource_path=RESOURCE_PATH,
        display_registry=display_registry,
        widgets=widgets,
        status_col=header_ret[1],
        interval_ms=DEFAULT_UPDATE_MS,
        time_parentkey=time_parentkey,
        time_childkey=time_childkey,
    )


def main():
    """Generate the Bokeh interface from an empty Bokeh document."""
    create_app(curdoc())
