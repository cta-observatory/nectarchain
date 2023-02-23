import pickle

import os
import sys


#from ctapipe.io import event_source
import sys
 
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from traitlets.config.loader import Config 
from astropy.io import fits
from astropy.table import Table

# ctapipe modules
from ctapipe import utils
from ctapipe.visualization import CameraDisplay
#from ctapipe.plotting.camera import CameraPlotter
from ctapipe.image.extractor import *
from ctapipe.io import EventSeeker 
from ctapipe.instrument import CameraGeometry

from ctapipe.io.hdf5tableio import HDF5TableWriter, HDF5TableReader

from ctapipe.io import EventSource
import ctapipe.instrument.camera.readout

from astropy import time as astropytime

class dqm_summary:
    def __init__(self):
        print("Processor 0")

    def DefineForRun(self, reader1):
        for i, evt1 in enumerate(reader1):
            self.FirstReader = reader1
            self.Samp = len(evt1.r0.tel[0].waveform[0][0])
            self.Chan = len(evt1.r0.tel[0].waveform[0])

        return self.Chan, self.Samp

    def ConfigureForRun(self):
        print("Processor 1")

    def ProcessEvent(self, evt):
        print("Processor 2")

    def FinishRun(self, M, M_ped, counter_evt, counter_ped):
        print("Processor 3")

    def GetResults(self):
        print("Processor 4")

    def PlotResults(
        self, name, FigPath, k, M, M_ped, Mean_M_overChan, Mean_M_ped_overChan
    ):
        print("Processor 5")


    def WriteAllResults(self,path, DICT):
        data2 = Table()
        data1 = Table()
        data = Table()
        hdulist = fits.HDUList()
        for i, j in DICT.items():
            if i == "Results_TriggerStatistics":
                for n2, m2 in j.items():
                    data2[n2] = m2
                hdu2 = fits.BinTableHDU(data2)
                hdu2.name = "Trigger"
            

            elif ((i == "Results_MeanWaveForms_HighGain") or (i == "Results_MeanWaveForms_LowGain")): 
                for n1, m1 in j.items():
                    data1[n1] = m1 
                hdu1 = fits.BinTableHDU(data1)
                hdu1.name = "MWF"


            else:
                for n, m in j.items():
                    data[n] = m
                hdu = fits.BinTableHDU(data)
                hdu.name = "Camera"
                  
        hdulist.append(hdu2)
        hdulist.append(hdu1) 
        hdulist.append(hdu)
        FileName = path + '_Results.fits'
        hdulist.writeto(FileName, overwrite=True)
        return None
