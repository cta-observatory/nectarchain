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

    timelines = collections.defaultdict(dict)
    for parentkey in source.keys():
        # Prepare timeline line plots only for pixel quantities evolving with time
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
        Nested dictionary containing camera display plots
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
        Nested dictionary containing display plots created by `make_camera_displays`
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
    display.add_colorbar()
    display.figure.title = child_key
    return display
