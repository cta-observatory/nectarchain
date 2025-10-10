import re

import numpy as np
from app_hooks import TEST_PATTERN, get_rundata, make_camera_displays, make_timelines

# bokeh imports
from bokeh.layouts import gridplot, layout, row
from bokeh.models import Select, TabPanel, Tabs  # , NumericInput
from bokeh.plotting import curdoc

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe_io_nectarcam import constants

from nectarchain.dqm.db_utils import DQMDB

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())


def update_camera_displays(attr, old, new):
    runid = run_select.value
    new_rundata = get_rundata(db, runid)

    # Reset each display
    for k in displays.keys():
        for kk in displays[k].keys():
            displays[k][kk].image = np.zeros(shape=constants.N_PIXELS)

    for parentkey in db[runid].keys():
        if not re.match(TEST_PATTERN, parentkey):
            for childkey in db[runid][parentkey].keys():
                print(f"Run id {runid} Updating plot for {parentkey}, {childkey}")

                image = new_rundata[parentkey][childkey]
                image = np.nan_to_num(image, nan=0.0)
                try:
                    displays[parentkey][childkey].image = image
                except ValueError as e:
                    print(
                        f"Caught {type(e).__name__} for {childkey}, filling display"
                        f"with zeros. Details: {e}"
                    )
                    image = np.zeros(shape=displays[parentkey][childkey].image.shape)
                    displays[parentkey][childkey].image = image
                except KeyError as e:
                    print(
                        f"Caught {type(e).__name__} for {childkey}, filling display"
                        f"with zeros. Details: {e}"
                    )
                    image = np.zeros(shape=constants.N_PIXELS)
                    displays[parentkey][childkey].image = image
                # TODO: TRY TO USE `stream` INSTEAD, ON UPDATES:
                # display.datasource.stream(new_data)
                # displays[parentkey][childkey].datasource.stream(image)


def update_timelines(attr, old, new):
    runid = run_select.value
    new_rundata = get_rundata(db, runid)
    make_timelines(db, new_rundata, runid)


print("Opening connection to ZODB")
db = DQMDB(read_only=True).root
print("Getting list of run numbers")
runids = sorted(list(db.keys()), reverse=True)

# First, get the run id with the most populated result dictionary
# On the full DB, this takes an awful lot of time, and saturates the RAM on the host
# VM (gets OoM killed)
# run_dict_lengths = [len(db[r]) for r in runids]
# runid = runids[np.argmax(run_dict_lengths)]
runid = "NectarCAM_Run6310"
print(f"We will start with run {runid}")

print("Defining Select")
# runid_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
run_select = Select(value=runid, title="NectarCAM run number", options=runids)

print(f"Getting data for run {run_select.value}")
source = get_rundata(db, run_select.value)
displays = make_camera_displays(db, source, runid)
timelines = make_timelines(db, source, runid)

run_select.on_change("value", update_camera_displays)
run_select.on_change("value", update_camera_displays, update_timelines)

controls = row(run_select)

# # TEST:
# attr = 'value'
# old = runid
# new = runids[1]
# update_camera_displays(attr, old, new)

ncols = 3
camera_displays = [
    displays[parentkey][childkey].figure
    for parentkey in displays.keys()
    for childkey in displays[parentkey].keys()
]
list_timelines = [
    timelines[parentkey][childkey]
    for parentkey in timelines.keys()
    for childkey in timelines[parentkey].keys()
]

layout_camera_displays = gridplot(
    camera_displays,
    sizing_mode="scale_width",
    ncols=ncols,
)

layout_timelines = gridplot(list_timelines, sizing_mode="scale_width", ncols=2)

# Create different tabs
tab_camera_displays = TabPanel(child=layout_camera_displays, title="Camera displays")
tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

# Combine panels into tabs
tabs = Tabs(tabs=[tab_camera_displays, tab_timelines])

# Add to the Bokeh document
curdoc().add_root(layout([controls, tabs]))
curdoc().title = "NectarCAM Data Quality Monitoring web app"
