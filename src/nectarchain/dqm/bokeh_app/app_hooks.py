import collections
import json
import os
import re

import numpy as np
from astropy.coordinates import SkyCoord

# bokeh imports
from bokeh.layouts import gridplot
from bokeh.models import ColorBar, TabPanel
from bokeh.plotting import figure

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

# ctapipe imports
from ctapipe.visualization.bokeh import CameraDisplay
from ctapipe_io_nectarcam import constants

base_dir = os.path.abspath(os.path.dirname(__file__))
labels_path = os.path.join(base_dir, "data", "labels.json")


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
    """Get run data to populate plots on the Bokeh displays

    Parameters
    ----------
    src : DQMDB
        Object-oriented database defined in nectarchain.dqm.db_utils
        from ZODB and ZEO ClientStorage
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'

    Returns
    -------
    dict
        Dictionary containing quantities extracted
        with nectarchain.dqm.start_dqm and stored into the database
    """

    run_data = src[runid]
    return run_data


def make_timelines(source, runid=None):
    """Make timeline plots for pixel quantities evolving with time

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.
        By default None

    Returns
    -------
    dict
        Nested dictionary containing line plots for the timelines
    """

    with open(labels_path, "r", encoding="utf-8") as file:
        y_axis_labels = json.load(file)["y_axis_labels_timelines"]

    timelines = collections.defaultdict(dict)
    for parentkey in source.keys():
        # Prepare timeline line plots only for pixel quantities evolving with time
        if re.match("(?:.*PIXTIMELINE-.*)", parentkey):
            for childkey in source[parentkey].keys():
                print(f"Run id {runid} Preparing plot for {parentkey}, {childkey}")
                evts = np.arange(len(source[parentkey][childkey]))
                timelines[parentkey][childkey] = figure(
                    title=childkey,
                    x_range=(0, np.max(evts) + 100),
                    y_range=(-1, np.max(source[parentkey][childkey]) + 5),
                )
                timelines[parentkey][childkey].line(
                    x=evts,
                    y=source[parentkey][childkey],
                    line_width=3,
                )
    for parentkey in timelines.keys():
        for childkey in timelines[parentkey].keys():
            timelines[parentkey][childkey].xaxis.axis_label = "Event number"
            try:
                timelines[parentkey][childkey].yaxis.axis_label = y_axis_labels[
                    parentkey
                ]
            except ValueError:
                timelines[parentkey][childkey].yaxis.axis_label = ""
            except KeyError:
                timelines[parentkey][childkey].yaxis.axis_label = ""

            timelines[parentkey][childkey].xaxis.axis_label_text_font_size = "12pt"
            timelines[parentkey][childkey].yaxis.axis_label_text_font_size = "12pt"
            timelines[parentkey][childkey].xaxis.major_label_text_font_size = "10pt"
            timelines[parentkey][childkey].yaxis.major_label_text_font_size = "10pt"
            timelines[parentkey][childkey].xaxis.axis_label_text_font_style = "normal"
            timelines[parentkey][childkey].yaxis.axis_label_text_font_style = "normal"

    return dict(timelines)


def update_timelines(data, timelines, runid=None):
    """Reset each timeline previously created by `make_timelines`

    Parameters
    ----------
    data : dict
        Dictionary returned by `get_rundata`
    timelines : dict
        Nested dictionary containing line plots created by `make_timelines`
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.
        By default None

    Returns
    -------
    bokeh.models.TabPanel
        Updated TabPanel containing the bokeh layout for the timeline plots
    """

    # Reset timeline line plots
    for k in timelines.keys():
        for kk in timelines[k].keys():
            timelines[k][kk].line(x=0, y=0)

    # Make new timeline plots
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

    # Recreate TabPanel layout
    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    return tab_timelines


def make_camera_displays(source, runid):
    """Make camera display plots using `make_camera_display`

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.

    Returns
    -------
    dict
        Nested dictionary containing display plots created by `make_camera_display`
    """

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
    """Reset each display previously created by `make_camera_displays`

    Parameters
    ----------
    data : dict
        Dictionary returned by `get_rundata`
    displays : dict
        Nested dictionary containing display plots
        created by `make_camera_displays`
    runid : str, optional
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.
        By default None

    Returns
    -------
    bokeh.models.TabPanel
        Updated TabPanel containing the bokeh layout for the display plots
    """

    ncols = 3

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


# TODO: some more explanation about the parent and child keys
# may help the user, if needed
def make_camera_display(source, parent_key, child_key):
    """Make camera display plot to fill the nested dict
       created by `make_camera_displays`

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    parent_key : str
        Parent key to extract quantity from the dict
    child_key : str
        Child key to extract quantity from the dict

    Returns
    -------
    ctapipe.visualization.bokeh.CameraDisplay
        CameraDisplay filled with values for the selected quantity,
        and displayed with the geometry from ctapipe.instrument.CameraGeometry
    """

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

    fig = display.figure
    # add axis labels
    pix_x = geom.pix_x
    pix_y = geom.pix_y
    cam_coords = SkyCoord(x=pix_x, y=pix_y, frame=geom.frame)
    fig.xaxis.axis_label = f"x / {cam_coords.x.unit}"
    fig.yaxis.axis_label = f"y / {cam_coords.y.unit}"
    fig.xaxis.axis_label_text_font_size = "12pt"
    fig.xaxis.axis_label_text_font_style = "normal"
    fig.yaxis.axis_label_text_font_size = "12pt"
    fig.yaxis.axis_label_text_font_style = "normal"

    # add colorbar
    color_bar = ColorBar(
        color_mapper=display._color_mapper,
        padding=5,
    )
    fig.add_layout(color_bar, "right")
    color_bar.title_text_font_size = "14pt"
    color_bar.title_text_font_style = "normal"

    with open(labels_path, "r", encoding="utf-8") as file:
        colorbar_labels = json.load(file)["colorbar_labels_camera_display"]

    try:
        color_bar.title = colorbar_labels[parent_key]
    except ValueError:
        color_bar.title = ""
    except KeyError:
        color_bar.title = ""

    display.figure.title = child_key

    return display
