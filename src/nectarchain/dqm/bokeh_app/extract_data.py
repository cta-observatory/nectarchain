import re
from datetime import datetime, timezone

from nectarchain.dqm.bokeh_app.logging_config import setup_logger

logger = setup_logger()


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


def categorize_source_data(source):
    """Categorize source data into plot types in a single pass.

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`

    Returns
    -------
    dict
        Dictionary with categorized data:
        - 'trigger_events': dict of trigger event data
        - 'waveforms': dict with 'average' and 'all' keys
        - 'timelines': dict of timeline data
        - 'camera_displays': dict of camera display data
    """
    categorized = {
        "trigger_events": {},
        "waveforms": {"average": {}, "all": {}},
        "timelines": {},
        "camera_displays": {},
    }

    logger.info("Extracting data...")
    for key in source.keys():
        # Trigger events (excluding WRONG)
        if re.match(r"TRIGGER-EVENTS-(?!WRONG).*", key):
            categorized["trigger_events"][key] = source[key]
            logger.info("- got trigger events")

        # Waveforms - average
        elif re.search(r"WF.*AVERAGE", key):
            categorized["waveforms"]["average"][key] = source[key]
            logger.info("- got average waveforms")

        # Waveforms - all
        elif re.match(r"WF-(?!.*AVERAGE).*", key):
            categorized["waveforms"]["all"][key] = source[key]
            logger.info("- got all other waveforms")

        # Timelines
        elif re.match(r".*PIXTIMELINE-.*", key):
            categorized["timelines"][key] = source[key]
            logger.info("- got timelines")

        # Camera displays (everything not excluded by TEST_PATTERN)
        elif not re.match(TEST_PATTERN, key):
            categorized["camera_displays"][key] = source[key]
            logger.info("- got camera displays")

    # Extract times if START-TIMES exists
    if "START-TIMES" in source.keys():
        run_start_time = int(source["START-TIMES"]["Run start time"].flatten()[0])
        run_start_time_dt = datetime.fromtimestamp(
            run_start_time, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        first_event_time = int(source["START-TIMES"]["First event"].flatten()[0])
        first_event_time_dt = datetime.fromtimestamp(
            first_event_time, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        last_event_time = int(source["START-TIMES"]["Last event"].flatten()[0])
        last_event_time_dt = datetime.fromtimestamp(
            last_event_time, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        categorized["times"] = [
            run_start_time_dt,
            first_event_time_dt,
            last_event_time_dt,
        ]
        logger.info("- got run timestamps")
    else:
        categorized["times"] = [None, None, None]

    return categorized
