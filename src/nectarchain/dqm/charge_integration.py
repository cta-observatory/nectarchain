import os

import ctapipe.instrument.camera.readout
import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.image.extractor import FixedWindowSum  # noqa: F401
from ctapipe.image.extractor import FullWaveformSum  # noqa: F401
from ctapipe.image.extractor import GlobalPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import LocalPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import NeighborPeakWindowSum  # noqa: F401
from ctapipe.image.extractor import SlidingWindowMaxSum  # noqa: F401
from ctapipe.image.extractor import TwoPassWindowSum  # noqa: F401
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import constants
from matplotlib import pyplot as plt
from traitlets.config.loader import Config

from ..makers.component import ChargesComponent
from ..makers.component.core import ArrayDataComponent
from ..makers.extractor.utils import CtapipeExtractor
from .dqm_summary_processor import DQMSummary

__all__ = ["ChargeIntegrationHighLowGain"]


class ChargeIntegrationHighLowGain(DQMSummary):
    def __init__(self, gaink):
        self.k = gaink
        self.gain_c = "High" if gaink == 0 else "Low"

        self.Pix = None
        self.Samp = None
        self.counter_evt = None
        self.counter_ped = None

        self.image_all = []
        self.peakpos_all = []
        self.image_ped = []
        self.peakpos_ped = []
        self.ped_all = []
        self.ped_ped = []
        self.camera = None
        self.integrator = None
        self.pixelBAD = None
        self.image_all_stats = None
        self.image_ped_stats = None
        self.ped_all_stats = None
        self.ped_ped_stats = None

        self.ChargeInt_Results_Dict = {}
        self.ChargeInt_Figures_Dict = {}
        self.ChargeInt_Figures_Names_Dict = {}

    def configure_for_run(self, path, Pix, Samp, Reader1, **charges_kwargs):
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp

        self.counter_evt = 0
        self.counter_ped = 0

        self.camera = Reader1.subarray.tel[0].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )

        self.cmap = "gnuplot2"

        self.subarray = Reader1.subarray
        subarray = Reader1.subarray
        subarray.tel[
            0
        ].camera.readout = ctapipe.instrument.camera.readout.CameraReadout.from_name(
            "NectarCam"
        )

        if charges_kwargs:
            extractor_kwargs = (
                ChargesComponent._get_extractor_kwargs_from_method_and_kwargs(
                    method=charges_kwargs["method"],
                    kwargs=charges_kwargs["extractor_kwargs"],
                )
            )
            self.integrator = eval(charges_kwargs["method"])(
                subarray, **extractor_kwargs
            )
        else:
            config = Config(
                {"GlobalPeakWindowSum": {"window_shift": 4, "window_width": 12}}
            )
            self.integrator = GlobalPeakWindowSum(subarray, config=config)

    def process_event(self, evt, noped):
        self.pixels = evt.nectarcam.tel[0].svc.pixel_ids
        self.pixelBADplot = evt.mon.tel[0].pixel_status.hardware_failing_pixels

        (
            broken_pixels_hg,
            broken_pixels_lg,
        ) = ArrayDataComponent._compute_broken_pixels_event(evt, self.pixels)

        if self.k == 0:
            self.pixelBAD = broken_pixels_hg
            channel = constants.HIGH_GAIN
        if self.k == 1:
            self.pixelBAD = broken_pixels_lg
            channel = constants.LOW_GAIN

        waveform = evt.r0.tel[0].waveform[self.k]
        waveform = waveform[self.pixels]

        ped = np.mean(waveform[:, 20])
        if noped:
            waveform = waveform - ped

        try:
            output = CtapipeExtractor.get_image_peak_time(
                self.integrator(
                    waveforms=waveform,
                    tel_id=0,
                    selected_gain_channel=channel,
                    broken_pixels=self.pixelBAD,
                )
            )
        except IndexError:
            waveform = waveform[np.newaxis, :]
            output = CtapipeExtractor.get_image_peak_time(
                self.integrator(
                    waveforms=waveform,
                    tel_id=0,
                    selected_gain_channel=channel,
                    broken_pixels=self.pixelBAD,
                )
            )

        image = output[0]
        peakpos = output[1]

        if evt.trigger.event_type.value == 32:  # count peds
            self.counter_ped += 1
            self.image_ped.append(image)
            self.peakpos_ped.append(peakpos)
            self.ped_ped.append(ped)
        else:
            self.counter_evt += 1
            self.image_all.append(image)
            self.peakpos_all.append(peakpos)
            self.ped_all.append(ped)

    def finish_run(self):
        self.peakpos_all = np.array(self.peakpos_all, dtype=float)
        if self.counter_ped > 0:
            self.peakpos_ped = np.array(self.peakpos_ped, dtype=float)

        # rms, percentile, mean deviation, median, mean,
        self.image_all = np.array(self.image_all, dtype=float)
        if self.image_all.size:
            self.image_all_stats = {
                "average": np.mean(self.image_all, axis=0),
                "median": np.median(self.image_all, axis=0),
                "std": np.std(self.image_all, axis=0),
                "rms": np.sqrt(np.sum(self.image_all**2, axis=0)),
            }

        self.ped_all = np.array(self.ped_all, dtype=float)
        if self.ped_all.size:
            self.ped_all_stats = {
                "average": np.mean(self.ped_all, axis=0),
                "median": np.median(self.ped_all, axis=0),
                "std": np.std(self.ped_all, axis=0),
                "rms": np.sqrt(np.sum(self.ped_all**2, axis=0)),
            }

        if self.counter_ped > 0:
            self.image_ped = np.array(self.image_ped, dtype=float)
            if self.image_ped.size:
                self.image_ped_stats = {
                    "average": np.mean(self.image_ped, axis=0),
                    "median": np.median(self.image_ped, axis=0),
                    "std": np.std(self.image_ped, axis=0),
                    "rms": np.sqrt(np.sum(self.image_ped**2, axis=0)),
                }

            self.ped_ped = np.array(self.ped_ped, dtype=float)
            if self.ped_ped.size:
                self.ped_ped_stats = {
                    "average": np.mean(self.ped_ped, axis=0),
                    "median": np.median(self.ped_ped, axis=0),
                    "std": np.std(self.ped_ped, axis=0),
                    "rms": np.sqrt(np.sum(self.ped_ped**2, axis=0)),
                }

    def get_results(self):
        for k, v in self.image_all_stats.items():
            self.ChargeInt_Results_Dict[
                (
                    f"CHARGE-INTEGRATION-IMAGE-ALL-{k.upper()}-"
                    f"{self.gain_c.upper()}-GAIN"
                )
            ] = v

        for k, v in self.ped_all_stats.items():
            self.ChargeInt_Results_Dict[
                f"PED-INTEGRATION-IMAGE-ALL-{k.upper()}-{self.gain_c.upper()}-GAIN"
            ] = v

        if self.counter_ped > 0:
            for k, v in self.image_ped_stats.items():
                self.ChargeInt_Results_Dict[
                    (
                        f"CHARGE-INTEGRATION-PED-ALL-{k.upper()}-"
                        f"{self.gain_c.upper()}-GAIN"
                    )
                ] = v

            for k, v in self.ped_ped_stats.items():
                self.ChargeInt_Results_Dict[
                    f"PED-INTEGRATION-PED-ALL-{k.upper()}-{self.gain_c.upper()}-GAIN"
                ] = v

        return self.ChargeInt_Results_Dict

    def _plot_camera_image(self, image, title, text, filename, key, fig_path):
        fig, disp = plt.subplots()
        disp = CameraDisplay(geometry=self.camera[~self.pixelBADplot[0]])
        disp.image = image
        disp.cmap = plt.cm.coolwarm
        disp.axes.text(2, -0.8, text, fontsize=12, rotation=90)
        disp.add_colorbar()
        plt.title(title)
        full_path = os.path.join(fig_path, filename)
        self.ChargeInt_Figures_Dict[key] = fig
        self.ChargeInt_Figures_Names_Dict[key] = full_path
        plt.close()

    def plot_results(self, name, fig_path):
        if self.counter_evt > 0:
            # Charge integration MEAN plot
            image = self.image_all_stats["average"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration Mean {self.gain_c} Gain (ALL)"
            filename = name + f"_ChargeInt_Mean_{self.gain_c}Gain_All.png"
            key = f"CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

            # Charge integration MEDIAN plot
            image = self.image_all_stats["median"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration Median {self.gain_c} Gain (ALL)"
            filename = name + f"_ChargeInt_Median_{self.gain_c}Gain_All.png"
            key = f"CHARGE-INTEGRATION-IMAGE-ALL-MEDIAN-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

            # Charge integration STD plot
            image = self.image_all_stats["std"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration STD {self.gain_c} Gain (ALL)"
            filename = name + f"_ChargeInt_Std_{self.gain_c}Gain_All.png"
            key = f"CHARGE-INTEGRATION-IMAGE-ALL-STD-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

            # Charge integration RMS plot
            image = self.image_all_stats["rms"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration RMS {self.gain_c} Gain (ALL)"
            filename = name + f"_ChargeInt_Rms_{self.gain_c}Gain_All.png"
            key = f"CHARGE-INTEGRATION-IMAGE-ALL-RMS-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

        if self.counter_ped > 0:
            image = self.image_ped_stats["average"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration Mean {self.gain_c} Gain (PED)"
            filename = name + f"_ChargeInt_Mean_{self.gain_c}Gain_Ped.png"
            key = f"CHARGE-INTEGRATION-IMAGE-PED-AVERAGE-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

            image = self.image_ped_stats["median"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration Median {self.gain_c} Gain (PED)"
            filename = name + f"_ChargeInt_Median_{self.gain_c}Gain_Ped.png"
            key = f"CHARGE-INTEGRATION-IMAGE-PED-MEDIAN-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

            image = self.image_ped_all["std"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration STD {self.gain_c} Gain (PED)"
            filename = name + f"_ChargeInt_Std_{self.gain_c}Gain_Ped.png"
            key = f"CHARGE-INTEGRATION-IMAGE-PED-STD-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

            image = self.image_ped_stats["rms"]
            text = f"{self.gain_c} gain integrated charge (DC)"
            title = f"Charge Integration RMS {self.gain_c} Gain (PED)"
            filename = name + f"_ChargeInt_Rms_{self.gain_c}Gain_Ped.png"
            key = f"CHARGE-INTEGRATION-IMAGE-PED-RMS-{self.gain_c.upper()}-GAIN"
            self._plot_camera_image(image, title, text, filename, key, fig_path)

        # Charge integration SPECTRUM
        if self.counter_evt > 0:
            fig, _ = plt.subplots()
            for i in range(len(self.pixels)):
                plt.hist(
                    self.image_all[:, i],
                    100,
                    fill=False,
                    density=True,
                    stacked=True,
                    linewidth=1,
                    log=True,
                    alpha=0.01,
                )
            plt.hist(
                np.mean(self.image_all, axis=1),
                100,
                color="r",
                linewidth=1,
                log=True,
                alpha=1,
                label="Camera average",
            )
            plt.legend()
            plt.xlabel("Charge (DC)")
            plt.title("Charge spectrum %s gain (ALL)" % self.gain_c)

            full_name = name + "_Charge_Spectrum_%sGain_All.png" % self.gain_c
            FullPath = os.path.join(fig_path, full_name)
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-ALL-%s-GAIN" % self.gain_c
            ] = fig
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-ALL-%s-GAIN" % self.gain_c
            ] = FullPath

            plt.close(fig)
            del fig

        if self.counter_ped > 0:
            fig, _ = plt.subplots()
            for i in range(len(self.pixels)):
                plt.hist(
                    self.image_ped[:, i],
                    100,
                    fill=False,
                    density=True,
                    stacked=True,
                    linewidth=1,
                    log=True,
                    alpha=0.01,
                )
            plt.hist(
                np.mean(self.image_ped, axis=1),
                100,
                color="r",
                linewidth=1,
                log=True,
                alpha=1,
                label="Camera average",
            )
            plt.legend()
            plt.xlabel("Charge (DC)")
            plt.title("Charge spectrum %s gain (PED)" % self.gain_c)

            full_name = name + "_Charge_Spectrum_%sGain_Ped.png" % self.gain_c
            FullPath = os.path.join(fig_path, full_name)
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-PED-%s-GAIN" % self.gain_c
            ] = fig
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-PED-%s-GAIN" % self.gain_c
            ] = FullPath

            plt.close()

        return self.ChargeInt_Figures_Dict, self.ChargeInt_Figures_Names_Dict
