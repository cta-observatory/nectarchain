import numpy as np

from markupsafe import Markup

# bokeh imports
from bokeh.layouts import layout, row, gridplot
from bokeh.io import show
from bokeh.models import Select  # , NumericInput
from bokeh.plotting import curdoc

# ctapipe imports
from ctapipe.visualization.bokeh import CameraDisplay
from ctapipe.instrument import CameraGeometry

from db_utils import SaveDB

NOTDISPLAY = ['Results_TriggerStatistics', 'Results_MeanWaveForms_HighGain', 'Results_MeanWaveForms_LowGain']

geom = CameraGeometry.from_name("NectarCam-003")

def get_rundata(src, runnb):
    run_data = src[runnb.value]
    return run_data

def make_camera_display(source, parent_key, child_key):
    # Example camera display
    image = source[parent_key][child_key]
    display = CameraDisplay(geometry=geom, image=image)
    display.add_colorbar()
    display.figure.title = child_key
    return display.figure
    
def update_camera_display(attr, old, new):
    runnb = run_select.value
    
    src = get_rundata(db, runnb)
    source.data.update(src.data)

db = SaveDB().root
runnb = sorted(list(db.keys()))[-1]
print(sorted(list(db.keys())))

# runnb_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
run_select = Select(value=runnb, title='NectarCAM run number', options=sorted(list(db.keys())))

source = get_rundata(db, run_select)
plots = []
for parentkey in db[runnb].keys():
    if parentkey not in NOTDISPLAY:
        for childkey in db[runnb][parentkey].keys():
            print(f'Preparing plot for {parentkey}, {childkey}')
            try:
                plots.append(make_camera_display(source,
                                                 parent_key=parentkey,
                                                 child_key=childkey))
            except BokehUserWarning:
                pass

run_select.on_change('value', update_camera_display)

controls = row(run_select)

curdoc().add_root(layout(controls, [plots[x:x+6] for x in range(0, len(plots), 6)]))
curdoc().title = 'NectarCAM Data Quality Monitoring web app'
