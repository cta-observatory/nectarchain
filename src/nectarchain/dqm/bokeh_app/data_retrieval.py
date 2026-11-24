import os
import re

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

from nectarchain.dqm.bokeh_app.logging_config import setup_logger

base_dir = os.path.abspath(os.path.dirname(__file__))
labels_path = os.path.join(base_dir, "data", "labels.json")


NOTINCAMERADISPLAY = [
    "CAMERA-PING-PONG-.*",
    "TRIGGER-.*",
    "PED-INTEGRATION-.*",
    "START-TIMES",
    "WF-.*",
    ".*PIXTIMELINE-.*",
    "CAMERA-TEMPERATURE-TREND",
]
TEST_PATTERN = "(?:% s)" % "|".join(NOTINCAMERADISPLAY)

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

logger = setup_logger()


def get_available_cameras_from_db_keys(src):
    """Get available cameras from database keys

    Parameters
    ----------
    src : DQMDB
        Object-oriented database defined in nectarchain.dqm.db_utils
        from ZODB and ZEO ClientStorage

    Returns
    -------
    set
        Set containing the names of available cameras
    """

    all_database_keys = list(src.keys())
    available_cameras = set()
    for key in all_database_keys:
        if not re.match(TEST_PATTERN, key):
            camera_name = key.split("NectarCAM")[1].split("_")[0]
            available_cameras.add(camera_name)

    logger.info(
        "Successfully extracted available cameras"
        + f"from database keys: {available_cameras}"
    )

    return available_cameras


def get_run_ids_for_camera(src, camera_code):
    """Get run ids for a given camera from database keys

    Parameters
    ----------
    src : DQMDB
        Object-oriented database defined in nectarchain.dqm.db_utils
        from ZODB and ZEO ClientStorage
    camera_code : str
        Code of the camera to filter the run ids for

    Returns
    -------
    list
        List containing the run ids for the given camera
    """

    all_database_keys = list(src.keys())
    run_ids_for_camera = []
    for key in all_database_keys:
        if f"NectarCAM{camera_code}" in key:
            run_ids_for_camera.append(key)

    logger.info(
        f"Successfully extracted run ids for camera {camera_code} from database keys"
    )

    run_ids_for_camera = sorted(run_ids_for_camera, reverse=True)
    return run_ids_for_camera


def get_rundata(src, runid):
    """Get run data to populate plots on the Bokeh displays

    Parameters
    ----------
    src : DQMDB
        Object-oriented database defined in nectarchain.dqm.db_utils
        from ZODB and ZEO ClientStorage
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAMQM_Run6310'

    Returns
    -------
    dict
        Dictionary containing quantities extracted
        with nectarchain.dqm.start_dqm and stored into the database
    """

    run_data = src[runid]

    logger.info(f"Successfully extracted data for run {runid}")

    return run_data


def refresh_run_options(
    db, available_cameras, run_selects, refresh_statuses=None, prev_run_counts=None
):
    """Refresh the run options for all camera Select widgets by polling the database.

    This function is called periodically to update the available run list
    without refreshing the entire application, allowing users to see newly
    added runs without losing their current work.

    Parameters
    ----------
    db : dict-like
        The database root object (from ZODB) containing all run keys.
    available_cameras : iterable
        Iterable of camera codes (strings) to refresh run options for.
    run_selects : dict
        Dictionary mapping camera codes to their corresponding Select widgets.
    refresh_statuses : dict, optional
        Dictionary mapping camera codes to their corresponding Div widgets
        for displaying refresh status. If provided, these will be updated with
        the refresh timestamp. By default None.
    prev_run_counts : dict, optional
        Dictionary to cache previous run counts per camera for calculating
        how many new runs were added. If provided, it will be updated with new counts.
        By default None.

    Returns
    -------
    None
        Updates the Select widgets' options and refresh status Divs in place.
    """
    from datetime import datetime

    all_keys = list(db.keys())
    refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for cam in available_cameras:
        # Filter runs for this specific camera
        runs = [k for k in all_keys if f"NectarCAM{cam}" in k]
        runs = sorted(runs, reverse=True)
        run_selects[cam].options = runs
        new_count = len(runs)

        # Calculate delta from previous count if cache is provided
        new_runs_since_last = 0
        if prev_run_counts is not None:
            prev_count = prev_run_counts.get(cam, new_count)
            new_runs_since_last = new_count - prev_count
            prev_run_counts[cam] = new_count  # Update cache for next refresh

        # Update refresh status display if provided
        if refresh_statuses is not None and cam in refresh_statuses:
            refresh_statuses[
                cam
            ].text = f"""
            <div style="
                background-color: moccasin;
                border-radius: 10px;
                padding: 10px;
                width: fit-content;
                font-size: 14px;
            ">
                <p>Runs last refreshed: {refresh_time}</p>
                <p>Available runs: {new_count}</p>
                <p>New since last refresh: {new_runs_since_last}</p>
            </div>
            """
