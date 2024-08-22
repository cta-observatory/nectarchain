import re

import numpy as np
from app_hooks import TEST_PATTERN, get_rundata, make_camera_displays

# bokeh imports
from bokeh.layouts import layout, row
from bokeh.models import Select  # , NumericInput
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

    for parentkey in db[runid].keys():
        if not re.match(TEST_PATTERN, parentkey):
            for childkey in db[runid][parentkey].keys():
                print(f"Run id {runid} Updating plot for {parentkey}, {childkey}")
                # try:
                image = new_rundata[parentkey][childkey]
                image = np.nan_to_num(image, nan=0.0)
                try:
                    displays[parentkey][childkey].image = image
                except ValueError:
                    image = np.zeros(shape=displays[parentkey][childkey].image.shape)
                    displays[parentkey][childkey].image = image
                except KeyError:
                    image = np.zeros(shape=constants.N_PIXELS)
                    displays[parentkey][childkey].image = image
                # TODO: TRY TO USE `stream` INSTEAD, ON UPDATES:
                # display.datasource.stream(new_data)
                # displays[parentkey][childkey].datasource.stream(image)


db = DQMDB(read_only=True).root
runids = sorted(list(db.keys()))
runid = runids[-1]

# runid_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
run_select = Select(value=runid, title="NectarCAM run number", options=runids)

source = get_rundata(db, run_select.value)
displays = make_camera_displays(db, source, runid)

run_select.on_change("value", update_camera_displays)

controls = row(run_select)

# # TEST:
# attr = 'value'
# old = runid
# new = runids[1]
# update_camera_displays(attr, old, new)

ncols = 3
plots = [
    displays[parentkey][childkey].figure
    for parentkey in displays.keys()
    for childkey in displays[parentkey].keys()
]
curdoc().add_root(
    layout(
        [[controls], [[plots[x : x + ncols] for x in range(0, len(plots), ncols)]]],
        sizing_mode="scale_width",
    )
)
curdoc().title = "NectarCAM Data Quality Monitoring web app"
