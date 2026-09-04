import os

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.containers import EventType
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay

from .dqm_summary_processor import DQMSummary

__all__ = ["PixelParticipationHighLowGain"]


class PixelParticipationHighLowGain(DQMSummary):
    """Compute and track bad-pixel participation per gain channel.

    Accumulates hardware-failing-pixel status over physical and pedestal
    events, and produces camera-display figures of the resulting masks.
    """

    def __init__(self, gaink, r0=False):
        """Initialize the pixel-participation processor.

        Parameters
        ----------
        gaink : int
            Gain channel index (0 = high gain, 1 = low gain).
        r0 : bool, optional
            Whether to use R0 waveforms (skip R1 corrections).
        """
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
        """Configure the processor for a new run.

        Parameters
        ----------
        path : str
            Path to the run file (unused by this processor).
        Pix : int
            Number of pixels in the camera.
        Samp : int
            Number of waveform samples (unused by this processor).
        Reader1 : ctapipe_io_nectarcam.LightNectarCAMEventSource
            Event source used to retrieve subarray and camera geometry.
        **kwargs
            Additional keyword arguments (ignored).
        """
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
        """Process a single event, accumulating bad-pixel masks.

        Parameters
        ----------
        evt : ctapipe EventSource event
            The event container.
        noped : bool
            Whether pedestal subtraction is enabled (unused).
        """
        pixelBAD = evt.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels[self.k]
        pixels = evt.nectarcam.tel[self.tel_id].svc.pixel_ids

        # Ensure 'pixels' is fixed length
        if len(pixels) < self.Pix:
            missing = np.arange(start=0, stop=self.Pix - len(pixels), step=1, dtype=int)
            pixels = np.concatenate([missing, pixels])

        bad_pixels = np.array(pixelBAD[pixels]).astype(int)

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
        """Finalise accumulated bad-pixel arrays by converting to ndarray."""
        self.BadPixels_ped = np.array(self.BadPixels_ped)
        self.BadPixels = np.array(self.BadPixels)

    def get_results(self):
        """Store bad-pixel masks in the results dictionary per gain.

        Returns
        -------
        dict
            Dictionary mapping result keys to bad-pixel arrays.
        """

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
        """Generate camera-display figures of bad-pixel masks.

        Parameters
        ----------
        name : str
            Base name for output figure files.
        fig_path : str
            Directory where figure files will be saved.

        Returns
        -------
        tuple of dict
            (figures_dict, filenames_dict) mapping result keys to
            matplotlib figures and their save paths.
        """

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
