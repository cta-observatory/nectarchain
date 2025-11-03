import os

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay

from .dqm_summary_processor import DQMSummary

__all__ = ["PixelParticipationHighLowGain"]


class PixelParticipationHighLowGain(DQMSummary):
    def __init__(self, gaink, r0=False):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.tel_id = None
        self.counter_evt = 0
        self.counter_ped = 0
        self.BadPixels_ped = None
        self.BadPixels = None
        self.camera = None
        self.cmap = "gnuplot2"
        self.PixelParticipation_Results_Dict = {}
        self.PixelParticipation_Figures_Dict = {}
        self.PixelParticipation_Figures_Names_Dict = {}
        super().__init__(r0)

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.counter_evt = 0
        self.counter_ped = 0
        self.BadPixels_ped = np.zeros(self.Pix)
        self.BadPixels = np.zeros(self.Pix)
        self.tel_id = Reader1.subarray.tel_ids[0]
        self.camera = Reader1.subarray.tel[self.tel_id].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )

    def process_event(self, evt, noped):
        pixelBAD = evt.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels[self.k]
        pixels = evt.nectarcam.tel[self.tel_id].svc.pixel_ids

        # Ensure 'pixels' is fixed length
        if len(pixels) < self.Pix:
            missing = np.arange(start=0, stop=self.Pix - len(pixels), step=1, dtype=int)
            pixels = np.concatenate([missing, pixels])

        bad_pixels = np.array(pixelBAD[pixels]).astype(int)
        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
            self.BadPixels_ped += bad_pixels

        else:
            self.counter_evt += 1
            self.BadPixels += bad_pixels

    def finish_run(self):
        self.BadPixels_ped = np.array(self.BadPixels_ped)
        self.BadPixels = np.array(self.BadPixels)

    def get_results(self):
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

    def plot_results(self, name, fig_path):
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
            FullPath = os.path.join(fig_path, full_name)
            self.PixelParticipation_Figures_Names_Dict[
                "CAMERA-BADPIX-PHY-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath
        if self.counter_evt > 0:
            self.PixelParticipation_Figures_Dict[
                "CAMERA-BADPIX-PED-DISPLAY-%s-GAIN" % gain_c
            ] = fig
            full_name = name + "_Pedestal_BPX_%sGain.png" % gain_c
            FullPath = os.path.join(fig_path, full_name)
            self.PixelParticipation_Figures_Names_Dict[
                "CAMERA-BADPIX-PED-DISPLAY-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        return (
            self.PixelParticipation_Figures_Dict,
            self.PixelParticipation_Figures_Names_Dict,
        )
