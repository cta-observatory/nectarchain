import os

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.containers import EventType
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay

from .dqm_summary_processor import DQMSummary

__all__ = ["PixelParticipationHighLowGain"]


class PixelParticipationHighLowGain(DQMSummary):
    def __init__(self, gaink, r0=False):
        self.k = gaink
        # For results dict keys (all caps)
        self.gain_key = "HIGH" if gaink == 0 else "LOW"
        # For plot titles and filenames (title case)
        self.gain_display = "High" if gaink == 0 else "Low"
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
        # Pre-allocate arrays with the correct dtype to avoid conversions
        self.BadPixels_ped = np.zeros(self.Pix, dtype=np.int64)
        self.BadPixels = np.zeros(self.Pix, dtype=np.int64)
        self.tel_id = Reader1.subarray.tel_ids[0]
        self.camera = Reader1.subarray.tel[self.tel_id].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )

    def process_event(self, evt, noped):
        pixelBAD = evt.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels[self.k]
        pixels = evt.nectarcam.tel[self.tel_id].svc.pixel_ids

        # Use np.put to efficiently place bad pixel values at their indices
        # This avoids creating temporary arrays with concatenate
        bad_pixels = np.zeros(self.Pix, dtype=np.int64)
        np.put(bad_pixels, pixels, pixelBAD[pixels])

        if evt.trigger.event_type == EventType.SKY_PEDESTAL:
            # count sky peds, event id 2
            self.counter_ped += 1
            self.BadPixels_ped += bad_pixels
        elif evt.trigger.event_type == EventType.SUBARRAY:
            # count standard physics stereo events, event id 32
            self.counter_evt += 1
            self.BadPixels += bad_pixels
        # TODO: add ids for other event types, e.g., dark pedestals
        # TODO: this else is wrong, we should have a separate counter
        # for other event types, e.g., dark pedestals. It has to be implemented.
        else:
            self.counter_evt += 1
            self.BadPixels += bad_pixels

    def finish_run(self):
        # Arrays are already numpy arrays from pre-allocation, no conversion needed
        pass

    def get_results(self):
        # ASSIGN RESULTS TO DICT
        # Use the pre-computed gain_key string for cleaner code
        if self.counter_evt > 0:
            self.PixelParticipation_Results_Dict[
                f"CAMERA-BadPix-PHY-OverEVENTS-{self.gain_key}-GAIN"
            ] = self.BadPixels

        if self.counter_ped > 0:
            self.PixelParticipation_Results_Dict[
                f"CAMERA-BadPix-PED-PHY-OverEVENTS-{self.gain_key}-GAIN"
            ] = self.BadPixels_ped

        return self.PixelParticipation_Results_Dict

    def plot_results(self, name, fig_path):
        # Only create plots if we have data
        if self.counter_evt > 0:
            entity = self.BadPixels
            title = f"Camera BPX {self.gain_display} gain (ALL)"
            full_name = f"{name}_Camera_BPX_{self.gain_display}Gain.png"
            key = f"CAMERA-BADPIX-PHY-DISPLAY-{self.gain_display}-GAIN"

            fig = self._create_badpixels_plot(entity, title)
            self.PixelParticipation_Figures_Dict[key] = fig
            self.PixelParticipation_Figures_Names_Dict[key] = os.path.join(
                fig_path, full_name
            )

        if self.counter_ped > 0:
            entity = self.BadPixels_ped
            title = f"Camera BPX {self.gain_display} gain (PED)"
            full_name = f"{name}_Pedestal_BPX_{self.gain_display}Gain.png"
            key = f"CAMERA-BADPIX-PED-DISPLAY-{self.gain_display}-GAIN"

            fig = self._create_badpixels_plot(entity, title)
            self.PixelParticipation_Figures_Dict[key] = fig
            self.PixelParticipation_Figures_Names_Dict[key] = os.path.join(
                fig_path, full_name
            )

        return (
            self.PixelParticipation_Figures_Dict,
            self.PixelParticipation_Figures_Names_Dict,
        )

    def _create_badpixels_plot(self, entity, title):
        """Helper method to create bad pixels plot with consistent styling"""
        fig, disp = plt.subplots()
        disp = CameraDisplay(
            geometry=self.camera,
            image=entity,
            cmap=plt.cm.coolwarm,
        )
        disp.add_colorbar()
        disp.axes.text(2.0, 0, "Bad Pixels", rotation=90)
        plt.title(title)
        plt.close()
        return fig
