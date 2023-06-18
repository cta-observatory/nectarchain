import collections
import numpy as np

# bokeh imports
from bokeh.layouts import layout, row
from bokeh.models import Select  # , NumericInput
from bokeh.plotting import curdoc

# ctapipe imports
from ctapipe.visualization.bokeh import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.coordinates import EngineeringCameraFrame

from db_utils import DQMDB

NOTINDISPLAY = ['Results_TriggerStatistics', 'Results_MeanWaveForms_HighGain', 'Results_MeanWaveForms_LowGain']

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

def get_rundata(src, runid):
    run_data = src[runid]
    return run_data

def make_camera_displays(db, source, runid):
    displays = collections.defaultdict(dict)
    for parentkey in db[runid].keys():
        if parentkey not in NOTINDISPLAY:
            for childkey in db[runid][parentkey].keys():
                print(f'Run id {runid} Preparing plot for {parentkey}, {childkey}')
                displays[parentkey][childkey] = make_camera_display(source,
                                                                    parent_key=parentkey,
                                                                    child_key=childkey)
    return dict(displays)

def make_camera_display(source, parent_key, child_key):
    # Example camera display
    image = source[parent_key][child_key]
    display = CameraDisplay(geometry=geom)
    try:
        display.image = image
    except ValueError:
        image = np.zeros(shape=display.image.shape)
        display.image = image
    display.add_colorbar()
    display.figure.title = child_key
    return display
   
def update_camera_displays(attr, old, new):
    runid = run_select.value
    new_rundata = get_rundata(db, runid)
    
    for parentkey in db[runid].keys():
        if parentkey not in NOTINDISPLAY:
            for childkey in db[runid][parentkey].keys():
                print(f'Run id {runid} Updating plot for {parentkey}, {childkey}')
                # try:
                image = new_rundata[parentkey][childkey]
                try:
                    displays[parentkey][childkey].image = image
                except ValueError:
                    image = np.zeros(shape=displays[parentkey][childkey].image.shape)
                    displays[parentkey][childkey].image = image
                # TODO: TRY TO USE `stream` INSTEAD, ON UPDATES:
                # display.datasource.stream(new_data)
                # displays[parentkey][childkey].datasource.stream(image)


db = DQMDB(read_only=True).root
runids = sorted(list(db.keys()))
runid = runids[0]

# runid_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
run_select = Select(value=runid, title='NectarCAM run number', options=runids)

source = get_rundata(db, run_select.value)
displays = make_camera_displays(db, source, runid)

run_select.on_change('value', update_camera_displays)

controls = row(run_select)

# # TEST:
# attr = 'value'
# old = runid
# new = runids[1]
# update_camera_displays(attr, old, new)

ncols = 3
plots = [displays[parentkey][childkey].figure for parentkey in displays.keys() for childkey in displays[parentkey].keys()]
curdoc().add_root(layout([[controls],
                          [[plots[x:x+ncols] for x in range(0, len(plots), ncols)]]],
                         sizing_mode='scale_width'
                         )
                  )
curdoc().title = 'NectarCAM Data Quality Monitoring web app'
