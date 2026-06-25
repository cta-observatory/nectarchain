import os

import numpy as np
from ctapipe.containers import EventType
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["WaveFormsHighLowGain"]


class WaveFormsHighLowGain(DQMSummary):
    def __init__(self, gaink, r0=False):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.tel_id = None
        self.wf = None
        self.wf_ped = None
        self.counter_evt = None
        self.counter_ped = None
        self.wf_average = None
        self.wf_ped_average = None
        self.wf_mean_over_pix = []
        self.wf_ped_mean_over_pix = []
        self.MeanWaveForms_Results_Dict = {}
        self.MeanWaveForms_Figures_Dict = {}
        self.MeanWaveForms_Figures_Names_Dict = {}

        self.wf_list_plot = None

        gain_c = "High" if self.k == 0 else "Low"
        self.gain_c = gain_c

        self.figure_keys = {
            "physical": f"FIGURE-WF-Physical-{gain_c}-GAIN",
            "pedestals": f"FIGURE-WF-Pedestals-{gain_c}-GAIN",
            "combined": f"FIGURE-WF-COMBINED-{gain_c}-GAIN",
        }

        self.figure_filenames = {
            "physical": f"_MeanWaveforms_Physical_{gain_c}Gain.png",
            "pedestals": f"_MeanWaveforms_Pedestals_{gain_c}Gain.png",
            "combined": f"_MeanWaveforms_CombinedPlot_{gain_c}Gain.png",
        }

        super().__init__(r0)

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        """Initialize waveform buffers and counters for a new run."""
        self.Pix = Pix
        self.Samp = Samp

        self.tel_id = Reader1.subarray.tel_ids[0]

        self.wf = np.zeros((Pix, Samp), dtype=np.float32)
        self.wf_ped = np.zeros((Pix, Samp), dtype=np.float32)

        self.counter_evt = 0
        self.counter_ped = 0

        self.wf_average = np.zeros((Pix, Samp), dtype=np.float32)
        self.wf_ped_average = np.zeros((Pix, Samp), dtype=np.float32)

        self.wf_list_plot = list(np.arange(1, Samp + 1))

    def process_event(self, evt, noped):
        is_ped = evt.trigger.event_type == EventType.SKY_PEDESTAL
        is_phy = evt.trigger.event_type == EventType.SUBARRAY

        # Update counters
        if is_ped:
            # count sky peds, event id 2
            self.counter_ped += 1
        elif is_phy:
            # count standard physics stereo events, event id 32
            self.counter_evt += 1
        # TODO: add ids for other event types, e.g., dark pedestals
        # TODO: this else is wrong, we should have a separate counter
        # for other event types, e.g., dark pedestals. It has to be implemented.
        else:
            self.counter_evt += 1

        # Extract ALL waveforms at once (vectorized)
        if self.r0:
            waveforms = evt.r0.tel[self.tel_id].waveform[self.k]
            # Shape: (Pix, Samp)
        else:
            # Handle both 2D (Pix, Samp) and 3D (Gain, Pix, Samp) cases
            wf = evt.r1.tel[self.tel_id].waveform
            if wf.ndim == 3:
                waveforms = wf[self.k]  # Select gain channel
            else:
                waveforms = wf  # Already 2D

        if is_ped:
            self.wf_ped += waveforms
        elif is_phy:
            self.wf += waveforms
        else:
            self.wf += waveforms

        return None

    def finish_run(self):
        """Compute mean waveforms over events and pixels."""

        if self.counter_evt > 0:
            self.wf_average = self.wf / self.counter_evt  # get average
            # get average over pixels
            self.wf_mean_over_pix = np.nanmean(self.wf_average, axis=0)

        if self.counter_ped > 0:
            # get average pedestals
            self.wf_ped_average = self.wf_ped / self.counter_ped
            self.wf_ped_mean_over_pix = np.nanmean(self.wf_ped_average, axis=0)

        return None

    def get_results(self):
        """Store waveform statistics in results dictionary by gain and type."""

        self.MeanWaveForms_Results_Dict[
            f"WF-PHY-AVERAGE-PIX-{self.gain_c.upper()}-GAIN"
        ] = self.wf_mean_over_pix
        self.MeanWaveForms_Results_Dict[
            f"WF-PHY-{self.gain_c.upper()}-GAIN"
        ] = self.wf_average
        if self.counter_ped > 0:
            self.MeanWaveForms_Results_Dict[
                f"WF-PED-AVERAGE-PIX-{self.gain_c.upper()}-GAIN"
            ] = self.wf_ped_mean_over_pix
            self.MeanWaveForms_Results_Dict[
                f"WF-PED-{self.gain_c.upper()}-GAIN"
            ] = self.wf_ped_average

        return self.MeanWaveForms_Results_Dict

    def plot_results(self, name, fig_path):
        wf_list = np.array(self.wf_list_plot)

        colors = ["black", "red"]
        titles = ["physical", "pedestals"]

        full_fig, full_ax = plt.subplots()
        array_plot = [self.wf_average]

        if self.counter_ped > 0:
            array_plot.append(self.wf_ped_average)

        for i, x in enumerate(array_plot):
            key = titles[i]
            fig_key = self.figure_keys[key]
            full_name = name + self.figure_filenames[key]
            fig_name = os.path.join(fig_path, full_name)

            part_fig, part_ax = plt.subplots()

            # VECTORIZED: Plot all waveforms at once
            part_ax.plot(wf_list, x.T, color=colors[i], alpha=0.08, linewidth=1)
            full_ax.plot(wf_list, x.T, color=colors[i], alpha=0.08, linewidth=1)

            part_ax.set_title(f"Mean Waveforms {key.capitalize()} ({self.gain_c} Gain)")
            part_ax.set_xlabel("Samples")
            part_ax.set_ylabel("Amplitude (DC)")
            part_ax.grid()

            self.MeanWaveForms_Figures_Dict[fig_key] = part_fig
            self.MeanWaveForms_Figures_Names_Dict[fig_key] = fig_name

            plt.close(part_fig)

        # Combined figure setup
        full_ax.set_title(f"Mean Waveforms Combined Plot ({self.gain_c} Gain)")
        full_ax.set_xlabel("Samples")
        full_ax.set_ylabel("Amplitude (DC)")
        full_ax.grid()

        combined_key = self.figure_keys["combined"]
        combined_path = os.path.join(fig_path, name + self.figure_filenames["combined"])

        self.MeanWaveForms_Figures_Dict[combined_key] = full_fig
        self.MeanWaveForms_Figures_Names_Dict[combined_key] = combined_path

        plt.close(full_fig)

        return self.MeanWaveForms_Figures_Dict, self.MeanWaveForms_Figures_Names_Dict
