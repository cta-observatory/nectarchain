import collections
import re

import numpy as np

# bokeh imports
from bokeh.layouts import gridplot
from bokeh.models import TabPanel
from bokeh.plotting import figure

# ctapipe imports
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


def make_timelines(source, runid=None):
    timelines = collections.defaultdict(dict)
    for parentkey in source.keys():
        if re.match("(?:.*PIXTIMELINE-.*)", parentkey):
            for childkey in source[parentkey].keys():
                print(f"Run id {runid} Preparing plot for {parentkey}, {childkey}")
                timelines[parentkey][childkey] = figure(title=childkey)
                evts = np.arange(len(source[parentkey][childkey]))
                timelines[parentkey][childkey].line(
                    x=evts,
                    y=source[parentkey][childkey],
                    line_width=3,
                )
    return dict(timelines)


def update_timelines(data, timelines, runid=None):
    # Reset each timeline
    for k in timelines.keys():
        for kk in timelines[k].keys():
            timelines[k][kk].line(x=0, y=0)

    timelines = make_timelines(data, runid)

    list_timelines = [
        timelines[parentkey][childkey]
        for parentkey in timelines.keys()
        for childkey in timelines[parentkey].keys()
    ]

    layout_timelines = gridplot(
        list_timelines,
        ncols=2,
    )

    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    return tab_timelines


def make_camera_displays(source, runid):
    displays = collections.defaultdict(dict)
    for parentkey in source.keys():
        if not re.match(TEST_PATTERN, parentkey):
            for childkey in source[parentkey].keys():
                print(f"Run id {runid} Preparing plot for {parentkey}, {childkey}")
                displays[parentkey][childkey] = make_camera_display(
                    source, parent_key=parentkey, child_key=childkey
                )
    return dict(displays)


def update_camera_displays(data, displays, runid=None):
    ncols = 3

    # Reset each display
    for k in displays.keys():
        for kk in displays[k].keys():
            displays[k][kk].image = np.zeros(shape=constants.N_PIXELS)

    displays = make_camera_displays(data, runid)

    camera_displays = [
        displays[parentkey][childkey].figure
        for parentkey in displays.keys()
        for childkey in displays[parentkey].keys()
    ]

    layout_camera_displays = gridplot(
        camera_displays,
        sizing_mode="scale_width",
        ncols=ncols,
    )

    tab_camera_displays = TabPanel(
        child=layout_camera_displays, title="Camera displays"
    )

    return tab_camera_displays


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
