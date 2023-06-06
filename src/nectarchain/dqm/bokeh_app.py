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
    run_data = src[runid.value]
    return run_data

def make_camera_displays(db, source, runid):
    plots = []
    for parentkey in db[runid].keys():
        if parentkey not in NOTINDISPLAY:
            for childkey in db[runid][parentkey].keys():
                print(f'Preparing plot for {parentkey}, {childkey}')
                # try:
                plots.append(make_camera_display(source,
                                                parent_key=parentkey,
                                                child_key=childkey))
                # except BokehUserWarning:
                #     pass
    return plots

def make_camera_display(source, parent_key, child_key):
    # Example camera display
    image = source[parent_key][child_key]
    display = CameraDisplay(geometry=geom, image=image)
    display.add_colorbar()
    display.figure.title = child_key
    return display.figure

def update_camera_displays(attr, old, new):
    runid = run_select.value
    for parentkey in db[runid].keys():
        if parentkey not in NOTINDISPLAY:
            for childkey in db[runid][parentkey].keys():
                update_camera_display()
    
def update_camera_display(attr, old, new):
    runid = run_select.value
    
    src = get_rundata(db, runid)
    source.data.update(src.data)

db = DQMDB().root
runid = sorted(list(db.keys()))[-1]
print(sorted(list(db.keys())))

# runid_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
run_select = Select(value=runid, title='NectarCAM run number', options=sorted(list(db.keys())))

source = get_rundata(db, run_select)
plots = make_camera_displays(db, source, runid)

run_select.on_change('value', update_camera_displays)

controls = row(run_select)

ncols = 3
curdoc().add_root(layout(controls, [plots[x:x+ncols-1] for x in range(0, len(plots), ncols-1)]))
curdoc().title = 'NectarCAM Data Quality Monitoring web app'
