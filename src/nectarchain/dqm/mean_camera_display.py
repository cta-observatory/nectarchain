import os

import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["MeanCameraDisplayHighLowGain"]


class MeanCameraDisplayHighLowGain(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.CameraAverage = None
        self.CameraAverage_ped = None
        self.counter_evt = None
        self.counter_ped = None
        self.camera = None
        self.cmap = None
        self.CameraAverage = []
        self.CameraAverage1 = []
        self.CameraAverage_ped = []
        self.CameraAverage_ped1 = []
        self.CameraAverage_overEvents = None
        self.CameraAverage_overEvents_overSamp = None
        self.CameraAverage_ped_overEvents = None
        self.CameraAverage_ped_overEvents_overSamp = None
        self.MeanCameraDisplay_Results_Dict = {}
        self.MeanCameraDisplay_Figures_Dict = {}
        self.MeanCameraDisplay_Figures_Names_Dict = {}

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp

        self.counter_evt = 0
        self.counter_ped = 0

        self.camera = Reader1.subarray.tel[0].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )

        self.cmap = "gnuplot2"

    def process_event(self, evt, noped):
        self.pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels
        pixel = evt.nectarcam.tel[0].svc.pixel_ids
        if len(pixel) < self.Pix:
            pixel21 = list(np.arange(0, self.Pix - len(pixel), 1, dtype=int))
            pixel = list(pixel)
            pixels = np.concatenate([pixel21, pixel])
        else:
            pixels = pixel

        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
            self.CameraAverage_ped1 = evt.r0.tel[0].waveform[self.k].sum(axis=1)
            self.CameraAverage_ped.append(self.CameraAverage_ped1[pixels])

        else:
            self.counter_evt += 1
            self.CameraAverage1 = evt.r0.tel[0].waveform[self.k].sum(axis=1)
            self.CameraAverage.append(self.CameraAverage1[pixels])

        return None

    def finish_run(self):
        if self.counter_evt > 0:
            self.CameraAverage = np.array(self.CameraAverage)
            self.CameraAverage = self.CameraAverage.sum(axis=0)
            self.CameraAverage_overEvents = self.CameraAverage / self.counter_evt

            self.CameraAverage_overEvents_overSamp = (
                self.CameraAverage_overEvents / self.Samp
            )

        if self.counter_ped > 0:
            self.CameraAverage_ped = np.array(self.CameraAverage_ped)
            self.CameraAverage_ped = self.CameraAverage_ped.sum(axis=0)
            self.CameraAverage_ped_overEvents = (
                self.CameraAverage_ped / self.counter_ped
            )
            self.CameraAverage_ped_overEvents_overSamp = (
                self.CameraAverage_ped_overEvents / self.Samp
            )

    def get_results(self):
        # ASSIGN RESUTLS TO DICT
        if self.k == 0:
            if self.counter_evt > 0:
                # self.MeanCameraDisplay_Results_Dict[
                # "CAMERA-AVERAGE-OverEVENTS-HIGH-GAIN"
                # ]  = self.CameraAverage_overEvents
                self.MeanCameraDisplay_Results_Dict[
                    "CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-HIGH-GAIN"
                ] = self.CameraAverage_overEvents_overSamp

            if self.counter_ped > 0:
                # self.MeanCameraDisplay_Results_Dict[
                # "CAMERA-AVERAGE-PED-OverEVENTS-HIGH-GAIN"
                # ]= self.CameraAverage_ped_overEvents
                self.MeanCameraDisplay_Results_Dict[
                    "CAMERA-AVERAGE-PED-OverEVENTS-OverSamp-HIGH-GAIN"
                ] = self.CameraAverage_ped_overEvents_overSamp

        if self.k == 1:
            if self.counter_evt > 0:
                # self.MeanCameraDisplay_Results_Dict[
                # "CAMERA-AVERAGE-OverEVENTS-LOW-GAIN"
                # ]  = self.CameraAverage_overEvents
                self.MeanCameraDisplay_Results_Dict[
                    "CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-LOW-GAIN"
                ] = self.CameraAverage_overEvents_overSamp

            if self.counter_ped > 0:
                # self.MeanCameraDisplay_Results_Dict[
                # "CAMERA-AVERAGE-PED-OverEVENTS-LOW-GAIN"
                # ]= self.CameraAverage_ped_overEvents
                self.MeanCameraDisplay_Results_Dict[
                    "CAMERA-AVERAGE-PED-OverEVENTS-OverSamp-LOW-GAIN"
                ] = self.CameraAverage_ped_overEvents_overSamp

        return self.MeanCameraDisplay_Results_Dict

    def plot_results(self, name, fig_path):
        # titles = ['All', 'Pedestals']
        if self.k == 0:
            gain_c = "High"
        if self.k == 1:
            gain_c = "Low"

        if self.counter_evt > 0:
            fig1, disp1 = plt.subplots()
            disp1 = CameraDisplay(
                geometry=self.camera[~self.pixelBAD[0]],
                image=self.CameraAverage_overEvents_overSamp[~self.pixelBAD[0]],
                cmap=plt.cm.coolwarm,
            )
            disp1.add_colorbar()
            disp1.axes.text(2.0, 0, "Charge (DC)", rotation=90)
            plt.title("Camera average %s gain (ALL)" % gain_c)

            self.MeanCameraDisplay_Figures_Dict[
                "CAMERA-AVERAGE-PHY-DISPLAY-%s-GAIN" % gain_c
            ] = fig1
            full_name = name + "_Camera_Mean_%sGain.png" % gain_c
            FullPath = os.path.join(fig_path, full_name)
            self.MeanCameraDisplay_Figures_Names_Dict[
                "CAMERA-AVERAGE-PHY-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath
            plt.close()

        if self.counter_ped > 0:
            fig2, disp2 = plt.subplots()
            disp2 = CameraDisplay(
                geometry=self.camera[~self.pixelBAD[0]],
                image=self.CameraAverage_ped_overEvents_overSamp[~self.pixelBAD[0]],
                cmap=plt.cm.coolwarm,
            )
            disp2.add_colorbar()
            disp2.axes.text(2.0, 0, "Charge (DC)", rotation=90)
            plt.title("Camera average %s gain (PED)" % gain_c)

            self.MeanCameraDisplay_Figures_Dict[
                "CAMERA-AVERAGE-PED-DISPLAY-%s-GAIN" % gain_c
            ] = fig2
            full_name = name + "_Pedestal_Mean_%sGain.png" % gain_c
            FullPath = os.path.join(fig_path, full_name)
            self.MeanCameraDisplay_Figures_Names_Dict[
                "CAMERA-AVERAGE-PED-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath
            plt.close()

        return (
            self.MeanCameraDisplay_Figures_Dict,
            self.MeanCameraDisplay_Figures_Names_Dict,
        )
