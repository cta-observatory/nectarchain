import collections
import re

import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

# ctapipe imports
from ctapipe.visualization.bokeh import CameraDisplay
from ctapipe_io_nectarcam import constants

NOTINDISPLAY = [
    "TRIGGER-.*",
    "PED-INTEGRATION-.*",
    "START-TIMES",
    "WF-.*",
    ".*PixTimeline-.*",
]
TEST_PATTERN = "(?:% s)" % "|".join(NOTINDISPLAY)

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())


def get_rundata(src, runid):
    run_data = src[runid]
    return run_data


def make_camera_displays(db, source, runid):
    displays = collections.defaultdict(dict)
    for key in db[runid].keys():
        if not re.match(TEST_PATTERN, key):
            print(f"Run id {runid} Preparing plot for {key}")
            displays[key] = make_camera_display(source, key=key)
    return dict(displays)


def make_camera_display(source, key):
    # Example camera display
    image = source[key]
    image = np.nan_to_num(image, nan=0.0)
    display = CameraDisplay(geometry=geom)
    try:
        display.image = image
    except ValueError:
        image = np.zeros(shape=display.image.shape)
        display.image = image
    except KeyError:
        image = np.zeros(shape=constants.N_PIXELS)
        display.image = image
    display.add_colorbar()
    display.figure.title = key
    return display
