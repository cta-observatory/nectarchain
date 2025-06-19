import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["PixelParticipationHighLowGain"]


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
        self.PixelParticipation_Results_Dict = {}
        self.PixelParticipation_Figures_Dict = {}
        self.PixelParticipation_Figures_Names_Dict = {}

        gain_c = "High" if gaink == 0 else "Low"
        self.gain_c = gain_c

        self.figure_keys = {
            "ped": f"CAMERA-BADPIX-PED-DISPLAY-{gain_c}-GAIN",
            "phy": f"CAMERA-BADPIX-PHY-DISPLAY-{gain_c}-GAIN",
        }

        self.figure_filenames = {
            "ped": f"_Pedestal_BPX_{gain_c}Gain.png",
            "phy": f"_Camera_BPX_{gain_c}Gain.png",
        }

    def ConfigureForRun(self, path, Pix, Samp, Reader1, **kwargs):
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

    def ProcessEvent(self, evt, noped):
        pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels[self.k]
        pixel = evt.nectarcam.tel[0].svc.pixel_ids

        # Create a full-size zero-initialized array for indexing safety
        status = np.zeros(self.Pix, dtype=int)
        np.put(status, pixel, pixelBAD[pixel])

        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
            self.BadPixels_ped += status

        else:
            self.counter_evt += 1
            self.BadPixels += status

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
        for key, data, count in [
            ("phy", self.BadPixels, self.counter_evt),
            ("ped", self.BadPixels_ped, self.counter_ped),
        ]:
            if count == 0:
                continue

            fig, disp = plt.subplots()

            title = (
                f"Camera BPX {self.gain_c} gain ({'ALL' if key == 'phy' else 'PED'})"
            )

            disp = CameraDisplay(
                geometry=self.camera,
                image=data,
                cmap=plt.cm.coolwarm,
            )
            disp.add_colorbar()
            disp.axes.text(2.0, 0, "Bad Pixels", rotation=90)
            plt.title(title)

            fig_key = self.figure_keys[key]
            fig_name = name + self.figure_filenames[key]
            fig_path = FigPath + fig_name

            self.PixelParticipation_Figures_Dict[fig_key] = fig
            self.PixelParticipation_Figures_Names_Dict[fig_key] = fig_path

            plt.close(fig)

        return (
            self.PixelParticipation_Figures_Dict,
            self.PixelParticipation_Figures_Names_Dict,
        )
