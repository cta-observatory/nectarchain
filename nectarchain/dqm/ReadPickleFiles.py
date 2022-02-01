import pickle
import sys

from matplotlib import pyplot as plt


from ctapipe import utils
from ctapipe.visualization import CameraDisplay
#from ctapipe.plotting.camera import CameraPlotter
from ctapipe.image.extractor import *
from ctapipe.io import EventSeeker 
from ctapipe.instrument import CameraGeometry
from ctapipe.io import EventSource, EventSeeker

import mpld3

filename = sys.argv[1]

infile = open(filename,'rb')
new_dict = pickle.load(infile)
infile.close()
print(new_dict)

'''
figx = pickle.load(open('NectarCAM_Run2720_Results.pickle',  'rb')  )
fig1 = figx["Figures"]
fig2 = fig1["Results_MeanCameraDisplay_HighGain_Figs"]
fig = fig2["CAMERA-AVERAGE-DISPLAY-HIGH-GAIN"]
'''


