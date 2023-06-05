import numpy as np

from markupsafe import Markup

# bokeh imports
from bokeh.layouts import layout, column, row
from bokeh.io import show
from bokeh.models import ColumnDataSource, Select  # , NumericInput
from bokeh.plotting import figure, curdoc

# ctapipe imports
from ctapipe.visualization.bokeh import CameraDisplay
from ctapipe.instrument import CameraGeometry

from db_utils import SaveDB

def get_rundata(src, runnb):
    run_data = src[runnb.value]
    return run_data

def make_camera_display(source):
    # Example camera display
    geom = CameraGeometry.from_name("NectarCam-003")
    image = source['Results_MeanCameraDisplay_HighGain']['CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-HIGH-GAIN']
    display = CameraDisplay(geometry=geom, image=image)
    return display.figure
    
def update_camera_display(attr, old, new):
    runnb = run_select.value
    
    src = get_rundata(db, runnb)
    source.data.update(src.data)
    
    

db = SaveDB().root
runnb = sorted(list(db.keys()))[-1]
print(sorted(list(db.keys())))

# runnb_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
run_select = Select(value=runnb, title='Run', options=sorted(list(db.keys())))

source = get_rundata(db, run_select)
plot = make_camera_display(source)

run_select.on_change('value', update_camera_display)

controls = row(run_select)

doc = curdoc()
doc.add_root(layout(controls, plot))
# doc.title('NectarCAM Data Quality Monitoring web app')
