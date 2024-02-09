import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay
from dqm_summary_processor import DQMSummary
from matplotlib import pyplot as plt


class PixelParticipationHighLowGain(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.counter_evt = None
        self.counter_ped = None
        self.BadPixels_ped = None
        self.BadPixels = None
        self.camera = None
        self.camera2 = None
        self.cmap = None
        self.cmap2 = None
        self.PixelParticipation_Results_Dict = {}
        self.PixelParticipation_Figures_Dict = {}
        self.PixelParticipation_Figures_Names_Dict = {}

    def ConfigureForRun(self, path, Pix, Samp, Reader1):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.counter_evt = 0
        self.counter_ped = 0
        self.BadPixels_ped = np.zeros(self.Pix)
        self.BadPixels = np.zeros(self.Pix)

        self.camera = Reader1.subarray.tel[0].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )

        self.camera2 = Reader1.subarray.tel[0].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )

        self.cmap = "gnuplot2"
        self.cmap2 = "gnuplot2"

    def ProcessEvent(self, evt, noped):
        pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels[self.k]
        pixel = evt.nectarcam.tel[0].svc.pixel_ids
        if len(pixel) < self.Pix:
            pixel21 = list(np.arange(0, self.Pix - len(pixel), 1, dtype=int))
            pixel = list(pixel)
            pixels = np.concatenate([pixel21, pixel])
        else:
            pixels = pixel

        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
            BadPixels_ped1 = list(map(int, pixelBAD[pixels]))
            self.BadPixels_ped += BadPixels_ped1

        else:
            self.counter_evt += 1
            BadPixels1 = list(map(int, pixelBAD[pixels]))
            self.BadPixels += BadPixels1
        return None

    def FinishRun(self):
        self.BadPixels_ped = np.array(self.BadPixels_ped)
        self.BadPixels = np.array(self.BadPixels)

    def GetResults(self):
        # ASSIGN RESUTLS TO DICT
        if self.k == 0:
            if self.counter_evt > 0:
                self.PixelParticipation_Results_Dict[
                    "CAMERA-BadPix-PHY-OverEVENTS-HIGH-GAIN"
                ] = self.BadPixels

            if self.counter_ped > 0:
                self.PixelParticipation_Results_Dict[
                    "CAMERA-BadPix-PED-PHY-OverEVENTS-HIGH-GAIN"
                ] = self.BadPixels_ped

        if self.k == 1:
            if self.counter_evt > 0:
                self.PixelParticipation_Results_Dict[
                    "CAMERA-BadPix-PHY-OverEVENTS-LOW-GAIN"
                ] = self.BadPixels

            if self.counter_ped > 0:
                self.PixelParticipation_Results_Dict[
                    "CAMERA-BadPix-PED-PHY-OverEVENTS-LOW-GAIN"
                ] = self.BadPixels_ped

        return self.PixelParticipation_Results_Dict

    def PlotResults(self, name, FigPath):
        # titles = ['All', 'Pedestals']
        if self.k == 0:
            gain_c = "High"
        if self.k == 1:
            gain_c = "Low"

        if self.counter_evt > 0:
            entity = self.BadPixels
            title = "Camera BPX %s gain (ALL)" % gain_c

        if self.counter_ped > 0:
            entity = self.BadPixels_ped
            title = "Camera BPX %s gain (PED)" % gain_c

        fig, disp = plt.subplots()
        disp = CameraDisplay(
            geometry=self.camera,
            image=entity,
            cmap=self.cmap,
        )
        disp.cmap = self.cmap
        disp.cmap = plt.cm.coolwarm
        disp.add_colorbar()
        disp.axes.text(2.0, 0, "Bad Pixels", rotation=90)
        plt.title(title)

        if self.counter_ped > 0:
            self.PixelParticipation_Figures_Dict[
                "CAMERA-BADPIX-PHY-DISPLAY-%s-GAIN" % gain_c
            ] = fig
            full_name = name + "_Camera_BPX_%sGain.png" % gain_c
            FullPath = FigPath + full_name
            self.PixelParticipation_Figures_Names_Dict[
                "CAMERA-BADPIX-PHY-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath
        if self.counter_evt > 0:
            self.PixelParticipation_Figures_Dict[
                "CAMERA-BADPIX-PED-DISPLAY-%s-GAIN" % gain_c
            ] = fig
            full_name = name + "_Pedestal_BPX_%sGain.png" % gain_c
            FullPath = FigPath + full_name
            self.PixelParticipation_Figures_Names_Dict[
                "CAMERA-BADPIX-PED-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        return (
            self.PixelParticipation_Figures_Dict,
            self.PixelParticipation_Figures_Names_Dict,
        )
