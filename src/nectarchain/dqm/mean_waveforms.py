import os

import numpy as np
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["MeanWaveFormsHighLowGain"]


class MeanWaveFormsHighLowGain(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.Mwf = None
        self.Mwf_ped = None
        self.counter_evt = None
        self.counter_ped = None
        self.Mwf_average = None
        self.Mwf_ped_average = None
        self.Mwf_Mean_overPix = []
        self.Mwf_ped_Mean_overPix = []
        self.MeanWaveForms_Results_Dict = {}
        self.MeanWaveForms_Figures_Dict = {}
        self.MeanWaveForms_Figures_Names_Dict = {}

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

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        """Initialize waveform buffers and counters for a new run."""
        self.Pix = Pix
        self.Samp = Samp

        self.Mwf = np.zeros((Pix, Samp))
        self.Mwf_ped = np.zeros((Pix, Samp))

        self.counter_evt = 0
        self.counter_ped = 0

        self.Mwf_average = np.zeros((Pix, Samp))
        self.Mwf_ped_average = np.zeros((Pix, Samp))

        self.wf_list_plot = np.arange(1, Samp + 1)

    def process_event(self, evt, noped):
        waveform = evt.r0.tel[0].waveform[self.k]

        if evt.trigger.event_type.value == 32:  # only peds now
            self.counter_ped += 1
            self.Mwf_ped += waveform
        else:
            self.counter_evt += 1
            self.Mwf += waveform
        return None

    def finish_run(self):
        """Compute mean waveforms over events and pixels."""

        if self.counter_evt > 0:
            self.Mwf_average = self.Mwf / self.counter_evt  # get average
            # get average over pixels
            self.Mwf_Mean_overPix = np.mean(self.Mwf_average, axis=0)

        if self.counter_ped > 0:
            # get average pedestals
            self.Mwf_ped_average = self.Mwf_ped / self.counter_ped
            self.Mwf_ped_Mean_overPix = np.mean(self.Mwf_ped_average, axis=0)

        return None

    def get_results(self):
        """Store waveform statistics in results dictionary by gain and type."""

        if self.k == 0:
            self.MeanWaveForms_Results_Dict.update(
                {"WF-PHY-AVERAGE-PIX-HIGH-GAIN": self.Mwf_Mean_overPix}
            )
            if self.counter_ped > 0:
                self.MeanWaveForms_Results_Dict.update(
                    {"WF-AVERAGE-PED-PIX-HIGH-GAIN": self.Mwf_ped_Mean_overPix}
                )

        elif self.k == 1:
            self.MeanWaveForms_Results_Dict.update(
                {"WF-AVERAGE-PIX-LOW-GAIN": self.Mwf_Mean_overPix}
            )
            if self.counter_ped > 0:
                self.MeanWaveForms_Results_Dict.update(
                    {"WF-PHY-AVERAGE-PED-PIX-LOW-GAIN": self.Mwf_ped_Mean_overPix}
                )

        return self.MeanWaveForms_Results_Dict

    def plot_results(self, name, fig_path):
        wf_list = np.array(self.wf_list_plot)

        colors = ["blue", "red"]
        titles = ["physical", "pedestals"]

        full_fig, full_ax = plt.subplots()
        array_plot = [self.Mwf_average]

        if self.counter_ped > 0:
            array_plot.append(self.Mwf_ped_average)

        for i, x in enumerate(array_plot):
            key = titles[i]
            fig_key = self.figure_keys[key]
            full_name = name + self.figure_filenames[key]
            fig_name = os.path.join(fig_path, full_name)

            part_fig, part_ax = plt.subplots()

            for ipix in range(self.Pix):
                part_ax.plot(
                    wf_list, x[ipix, :], color=colors[i], alpha=0.005, linewidth=1
                )
                full_ax.plot(
                    wf_list, x[ipix, :], color=colors[i], alpha=0.005, linewidth=1
                )

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
