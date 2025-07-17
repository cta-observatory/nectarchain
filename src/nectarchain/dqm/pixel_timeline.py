import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["PixelTimelineHighLowGain"]

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class PixelTimelineHighLowGain(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.gain_c = "High" if gaink == 0 else "Low"

        self.Pix = None
        self.Samp = None
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

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.counter_evt = 0
        self.counter_ped = 0

    def process_event(self, evt, noped):
        pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels[self.k]
        pixels = evt.nectarcam.tel[0].svc.pixel_ids

        status = np.zeros(self.Pix, dtype=int)
        np.put(status, pixels, pixelBAD[pixels])
        bad_count = np.sum(status)

        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
            self.SumBadPixels_ped.append(bad_count)
            self.SumBadPixels.append(0)

        else:
            self.counter_evt += 1
            self.SumBadPixels.append(bad_count)
            self.SumBadPixels_ped.append(0)

        return None

    def finish_run(self):
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
            ax.legend()

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
