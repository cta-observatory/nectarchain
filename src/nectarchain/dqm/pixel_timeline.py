import logging
import os

import numpy as np
from ctapipe.containers import EventType
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["PixelTimelineHighLowGain"]

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class PixelTimelineHighLowGain(DQMSummary):
    """Track the fraction of bad pixels over time per gain channel.

    Accumulates the per-event count of hardware-failing pixels separately
    for physical and pedestal events, then normalises by total pixel count
    to produce a timeline of bad-pixel fraction.
    """

    def __init__(self, gaink, r0=False):
        """Initialise the pixel-timeline processor.

        Parameters
        ----------
        gaink : int
            Gain channel index (0 = high gain, 1 = low gain).
        r0 : bool, optional
            Whether to use R0 waveforms (skip R1 corrections).
        """
        self.k = gaink
        self.gain_c = "High" if gaink == 0 else "Low"

        self.Pix = None
        self.Samp = None
        self.tel_id = None
        self.counter_evt = None
        self.counter_ped = None

        self.SumBadPixels_ped = []
        self.SumBadPixels = []

        self.BadPixelTimeline_ped = None
        self.BadPixelTimeline = None

        self.PixelTimeline_Results_Dict = {}
        self.PixelTimeline_Figures_Dict = {}
        self.PixelTimeline_Figures_Names_Dict = {}

        self.figure_keys = {
            "all": f"BPX-TIMELINE-ALL-{self.gain_c}-GAIN",
            "ped": f"BPX-TIMELINE-PED-{self.gain_c}-GAIN",
        }

        self.figure_filenames = {
            "all": f"_BPX_Timeline_{self.gain_c}Gain_All.png",
            "ped": f"_BPX_Timeline_{self.gain_c}Gain_Ped.png",
        }

        super().__init__(r0)

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        """Configure the processor for a new run.

        Parameters
        ----------
        path : str
            Path to the run file (unused).
        Pix : int
            Number of pixels in the camera.
        Samp : int
            Number of waveform samples (unused).
        Reader1 : ctapipe_io_nectarcam.LightNectarCAMEventSource
            Event source used to retrieve the telescope ID.
        **kwargs
            Additional keyword arguments (ignored).
        """
        self.Pix = Pix
        self.Samp = Samp
        self.counter_evt = 0
        self.counter_ped = 0
        self.tel_id = Reader1.subarray.tel_ids[0]

    def process_event(self, evt, noped):
        """Process a single event, updating the bad-pixel count timeline.

        Parameters
        ----------
        evt : ctapipe EventSource event
            The event container.
        noped : bool
            Whether pedestal subtraction is enabled (unused).
        """
        pixelBAD = evt.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels[self.k]
        pixels = evt.nectarcam.tel[self.tel_id].svc.pixel_ids

        status = np.zeros(self.Pix, dtype=int)
        np.put(status, pixels, pixelBAD[pixels])
        bad_count = np.sum(status)

        if evt.trigger.event_type == EventType.SKY_PEDESTAL:
            # count sky peds, event id 2
            self.counter_ped += 1
            self.SumBadPixels_ped.append(bad_count)
            self.SumBadPixels.append(0)
        elif evt.trigger.event_type == EventType.SUBARRAY:
            # count standard physics stereo events, event id 32
            self.counter_evt += 1
            self.SumBadPixels.append(bad_count)
            self.SumBadPixels_ped.append(0)
        # TODO: add ids for other event types, e.g., dark pedestals
        # TODO: this else is wrong, we should have a separate counter
        # for other event types, e.g., dark pedestals. It has to be implemented.
        else:
            self.counter_evt += 1
            self.SumBadPixels.append(bad_count)
            self.SumBadPixels_ped.append(0)

        return None

    def finish_run(self):
        """Normalise accumulated bad-pixel counts by the number of pixels."""
        self.BadPixelTimeline_ped = (
            np.array(self.SumBadPixels_ped, dtype=float) / self.Pix
        )
        self.BadPixelTimeline = np.array(self.SumBadPixels, dtype=float) / self.Pix
        log.debug(f"BadPixelTimeline is:\n{self.BadPixelTimeline}")
        log.debug(f"BadPixelTimeline_ped is:\n{self.BadPixelTimeline_ped}")

    def get_results(self):
        """Store results to output dictionary"""

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

    def plot_results(self, name, fig_path):
        """Generate timeline plots of bad-pixel fraction.

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
        for key, data, count, label in [
            ("all", self.BadPixelTimeline, self.counter_evt, "Physical events"),
            ("ped", self.BadPixelTimeline_ped, self.counter_ped, "Pedestal events"),
        ]:
            if count == 0:
                continue

            fig, ax = plt.subplots()
            ax.plot(np.arange(count), data * 100, label=label)
            ax.set_xlabel("Timeline")
            ax.set_ylabel("BPX fraction (%)")
            ax.set_title(f"BPX Timeline {self.gain_c} gain ({key.capitalize()})")
            ax.legend(loc="upper right")

            key_id = self.figure_keys[key]
            filename = name + self.figure_filenames[key]
            full_path = os.path.join(fig_path, filename)

            self.PixelTimeline_Figures_Dict[key_id] = fig
            self.PixelTimeline_Figures_Names_Dict[key_id] = full_path

            plt.close(fig)

        return (
            self.PixelTimeline_Figures_Dict,
            self.PixelTimeline_Figures_Names_Dict,
        )
