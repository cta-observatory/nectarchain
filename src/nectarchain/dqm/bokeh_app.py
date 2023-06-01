import numpy as np

# bokeh imports
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, NumericInput
from bokeh.plotting import figure, curdoc

# ctapipe imports
from ctapipe.visualization.bokeh import CameraDisplay
from ctapipe.instrument import CameraGeometry

from db_utils import SaveDB

db = SaveDB()
runnb_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")

# Example camera display
geom = CameraGeometry.from_name("NectarCam-003")
image = db.root[runnb_input]['Results_MeanCameraDisplay_HighGain']['CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-HIGH-GAIN']
display = CameraDisplay(geom, image)


def callback(attr, new, old):
    # TODO to be implemented
    print(attr, new, old)


display.enable_pixel_picker(callback)
