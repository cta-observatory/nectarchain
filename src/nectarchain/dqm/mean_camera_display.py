from dqm_summary_processor import dqm_summary
from matplotlib import pyplot as plt
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
import numpy as np


class MeanCameraDisplay_HighLowGain(dqm_summary):
    def __init__(self, gaink):
        self.k = gaink
        return None

    def ConfigureForRun(self, path, Chan, Samp, Reader1):
        # define number of channels and samples
        self.Chan = Chan
        self.Samp = Samp

        self.CameraAverage = np.zeros((self.Chan))
        self.CameraAverage_ped = np.zeros((self.Chan))
        self.counter_evt = 0
        self.counter_ped = 0

        self.camera = CameraGeometry.from_name("NectarCam-003")
        self.camera2 = CameraGeometry.from_name("NectarCam-003")

        self.cmap = "gnuplot2"
        self.cmap2 = "gnuplot2"

    def ProcessEvent(self, evt):
        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
        else:
            self.counter_evt += 1

        if evt.trigger.event_type.value == 32:  # only peds now
            self.CameraAverage_ped += (
                evt.r0.tel[0].waveform[self.k].sum(axis=1)
            )  # fill channels one by one and sum them for peds only
        else:
            self.CameraAverage += (
                evt.r0.tel[0].waveform[self.k].sum(axis=1)
            )  # fill channels one by one and sum them
        return None

    def FinishRun(self):
        self.CameraAverage_overEvents = self.CameraAverage / self.counter_evt
        self.CameraAverage_overEvents_overSamp = (
            self.CameraAverage_overEvents / self.Samp
        )

        if self.counter_ped > 0:
            self.CameraAverage_ped_overEvents = (
                self.CameraAverage_ped / self.counter_ped
            )
            self.CameraAverage_ped_overEvents_overSamp = (
                self.CameraAverage_ped_overEvents / self.Samp
            )

    def GetResults(self):
        # INITIATE DICT
        self.MeanCameraDisplay_Results_Dict = {}

        # ASSIGN RESUTLS TO DICT
        if self.k == 0:
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

    def PlotResults(self, name, FigPath):
        self.MeanCameraDisplay_Figures_Dict = {}
        self.MeanCameraDisplay_Figures_Names_Dict = {}

        # titles = ['All', 'Pedestals']
        if self.k == 0:
            gain_c = "High"
        if self.k == 1:
            gain_c = "Low"

        if self.counter_evt > 0:
            fig1, self.disp1 = plt.subplots()
            self.disp1 = CameraDisplay(
                geometry=self.camera,
                image=self.CameraAverage_overEvents_overSamp,
                cmap=self.cmap,
            )
            self.disp1.cmap = self.cmap
            self.disp1.cmap = plt.cm.coolwarm
            self.disp1.add_colorbar()
            self.disp1.axes.text(2.0, 0, "Charge (DC)", rotation=90)
            plt.title("Camera average %s gain (ALL)" % gain_c)

            self.MeanCameraDisplay_Figures_Dict[
                "CAMERA-AVERAGE-PHY-DISPLAY-%s-GAIN" % gain_c
            ] = fig1
            full_name = name + "_Camera_Mean_%sGain.png" % gain_c
            FullPath = FigPath + full_name
            self.MeanCameraDisplay_Figures_Names_Dict[
                "CAMERA-AVERAGE-PHY-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath
            plt.close()

        if self.counter_ped > 0:
            fig2, self.disp2 = plt.subplots()
            self.disp2 = CameraDisplay(
                geometry=self.camera2,
                image=self.CameraAverage_ped_overEvents_overSamp,
                cmap=self.cmap2,
            )
            self.disp2.cmap = self.cmap2
            self.disp2.cmap = plt.cm.coolwarm
            self.disp2.add_colorbar()
            self.disp2.axes.text(2.0, 0, "Charge (DC)", rotation=90)
            plt.title("Camera average %s gain (PED)" % gain_c)

            self.MeanCameraDisplay_Figures_Dict[
                "CAMERA-AVERAGE-PED-DISPLAY-%s-GAIN" % gain_c
            ] = fig2
            full_name = name + "_Pedestal_Mean_%sGain.png" % gain_c
            FullPath = FigPath + full_name
            self.MeanCameraDisplay_Figures_Names_Dict[
                "CAMERA-AVERAGE-PED-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath
            plt.close()

        return (
            self.MeanCameraDisplay_Figures_Dict,
            self.MeanCameraDisplay_Figures_Names_Dict,
        )
