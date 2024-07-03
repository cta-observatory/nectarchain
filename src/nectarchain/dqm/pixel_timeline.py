import numpy as np
from matplotlib import pyplot as plt

from nectarchain.dqm.dqm_summary_processor import DQMSummary


class PixelTimelineHighLowGain(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.counter_evt = None
        self.counter_ped = None
        self.SumBadPixels_ped = []
        self.SumBadPixels = []
        self.BadPixelTimeline_ped = None
        self.BadPixelTimeline = None
        self.camera = None
        self.cmap = None
        self.cmap2 = None
        self.PixelTimeline_Results_Dict = {}
        self.PixelTimeline_Figures_Dict = {}
        self.PixelTimeline_Figures_Names_Dict = {}

    def ConfigureForRun(self, path, Pix, Samp, Reader1):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.counter_evt = 0
        self.counter_ped = 0

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
            self.counter_evt += 1
            BadPixels_ped1 = list(map(int, pixelBAD[pixels]))
            SumBadPixelsEvent_ped = sum(BadPixels_ped1)
            self.SumBadPixels_ped.append(SumBadPixelsEvent_ped)
            self.SumBadPixels.append(0)

        else:
            self.counter_evt += 1
            self.counter_ped += 1
            BadPixels1 = list(map(int, pixelBAD[pixels]))
            SumBadPixelsEvent = sum(BadPixels1)
            self.SumBadPixels.append(SumBadPixelsEvent)
            self.SumBadPixels_ped.append(0)

        return None

    def FinishRun(self):
        self.BadPixelTimeline_ped = (
            np.array(self.SumBadPixels_ped, dtype=float) / self.Pix
        )
        self.BadPixelTimeline = np.array(self.SumBadPixels, dtype=float) / self.Pix
        print(self.BadPixelTimeline)
        print(self.BadPixelTimeline_ped)

    def GetResults(self):
        # ASSIGN RESUTLS TO DICT
        if self.k == 0:
            if self.counter_evt > 0:
                self.PixelTimeline_Results_Dict[
                    "CAMERA-BadPixTimeline-PHY-HIGH-GAIN"
                ] = self.BadPixelTimeline

            if self.counter_ped > 0:
                self.PixelTimeline_Results_Dict[
                    "CAMERA-BadPixTimeline-PED-HIGH-GAIN"
                ] = self.BadPixelTimeline_ped

        if self.k == 1:
            if self.counter_evt > 0:
                self.PixelTimeline_Results_Dict[
                    "CAMERA-BadPixTimeline-PHY-LOW-GAIN"
                ] = self.BadPixelTimeline

            if self.counter_ped > 0:
                self.PixelTimeline_Results_Dict[
                    "CAMERA-BadPixTimeline-PED-LOW-GAIN"
                ] = self.BadPixelTimeline_ped

        return self.PixelTimeline_Results_Dict

    def PlotResults(self, name, FigPath):
        # titles = ['All', 'Pedestals']
        if self.k == 0:
            gain_c = "High"
        if self.k == 1:
            gain_c = "Low"

        if self.counter_evt > 0:
            fig1, disp = plt.subplots()
            plt.plot(
                np.arange(self.counter_evt),
                self.BadPixelTimeline * 100,
                label="Physical events",
            )
            plt.legend()
            plt.xlabel("Timeline")
            plt.ylabel("BPX fraction (%)")
            plt.title("BPX Timeline %s gain (ALL)" % gain_c)

            full_name = name + "_BPX_Timeline_%sGain_All.png" % gain_c
            FullPath = FigPath + full_name
            self.PixelTimeline_Figures_Dict["BPX-TIMELINE-ALL-%s-GAIN" % gain_c] = fig1
            self.PixelTimeline_Figures_Names_Dict[
                "BPX-TIMELINE-ALL-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        if self.counter_ped > 0:
            fig2, disp = plt.subplots()
            plt.plot(
                np.arange(self.counter_ped),
                self.BadPixelTimeline_ped * 100,
                label="Pedestal events",
            )
            plt.legend()
            plt.xlabel("Timeline")
            plt.ylabel("BPX fraction (%)")
            plt.title("BPX Timeline %s gain (PED)" % gain_c)

            full_name = name + "_BPX_Timeline_%sGain_Ped.png" % gain_c
            FullPath = FigPath + full_name
            self.PixelTimeline_Figures_Dict["BPX-TIMELINE-PED-%s-GAIN" % gain_c] = fig2
            self.PixelTimeline_Figures_Names_Dict[
                "BPX-TIMELINE-PED-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        return (
            self.PixelTimeline_Figures_Dict,
            self.PixelTimeline_Figures_Names_Dict,
        )
