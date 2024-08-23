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
    ".*PIXTIMELINE-.*",
]
TEST_PATTERN = "(?:% s)" % "|".join(NOTINDISPLAY)

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())


def get_rundata(src, runid):
    run_data = src[runid]
    return run_data


def make_camera_displays(db, source, runid):
    displays = collections.defaultdict(dict)
    for parentkey in db[runid].keys():
        if not re.match(TEST_PATTERN, parentkey):
            for childkey in db[runid][parentkey].keys():
                print(f"Run id {runid} Preparing plot for {parentkey}, {childkey}")
                displays[parentkey][childkey] = make_camera_display(
                    source, parent_key=parentkey, child_key=childkey
                )
    return dict(displays)


def make_camera_display(source, parent_key, child_key):
    # Example camera display
    image = source[parent_key][child_key]
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
    display.figure.title = child_key
    return display
