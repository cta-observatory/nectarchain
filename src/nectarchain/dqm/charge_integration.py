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
        self.cmap = None
        self.subarray = None
        self.integrator = None
        self.pixelBAD = None
        self.image_all = []
        self.image_all_median = None
        self.image_all_average = None
        self.image_all_std = None
        self.image_all_rms = None
        self.ped_all_median = None
        self.ped_all_average = None
        self.ped_all_std = None
        self.ped_all_rms = None
        self.image_ped = []
        self.image_ped_median = None
        self.image_ped_average = None
        self.image_ped_std = None
        self.image_ped_rms = None
        self.ped_ped_median = None
        self.ped_ped_average = None
        self.ped_ped_std = None
        self.ped_ped_rms = None
        self.ChargeInt_Results_Dict = {}
        self.ChargeInt_Figures_Dict = {}
        self.ChargeInt_Figures_Names_Dict = {}

    def ConfigureForRun(self, path, Pix, Samp, Reader1, **charges_kwargs):
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
                {"LocalPeakWindowSum": {"window_shift": 4, "window_width": 12}}
            )
            self.integrator = LocalPeakWindowSum(subarray, config=config)
        print("self.integrator", self.integrator)
        # can be replaced by: imageExtractor =
        # __class__._get_imageExtractor(method=method, subarray=subarray, **kwargs)

    def ProcessEvent(self, evt, noped):
        pixel = evt.nectarcam.tel[0].svc.pixel_ids

        if len(pixel) < self.Pix:
            pixel_masked_shutter = list(
                np.arange(0, self.Pix - len(pixel), 1, dtype=int)
            )
            pixel = list(pixel)
            pixels = np.concatenate([pixel_masked_shutter, pixel])
        else:
            pixels = pixel

        (
            broken_pixels_hg,
            broken_pixels_lg,
        ) = ArrayDataComponent._compute_broken_pixels_event(evt, pixels)

        if self.k == 0:
            self.pixelBAD = broken_pixels_hg
            channel = constants.HIGH_GAIN
        if self.k == 1:
            self.pixelBAD = broken_pixels_lg
            channel = constants.LOW_GAIN

        # self.pixelBAD = ArrayDataComponent._compute_broken_pixels(
        # evt.r0.tel[0].waveform[constants.HIGH_GAIN],
        # evt.r0.tel[0].waveform[constants.LOW_GAIN])
        # self.pixelBAD = evt.mon.tel[0].pixel_status.hardware_failing_pixels

        waveform = evt.r0.tel[0].waveform[self.k]
        ped = np.mean(waveform[:, 20])

        if noped:
            w_noped = waveform - ped
            output = CtapipeExtractor.get_image_peak_time(
                self.integrator(w_noped, 0, channel, self.pixelBAD)
            )

            image = output[0]
            peakpos = output[1]
            image = image[pixels]
            peakpos = peakpos[pixels]

        else:
            output = CtapipeExtractor.get_image_peak_time(
                self.integrator(waveform, 0, channel, self.pixelBAD)
            )

            image = output[0]
            peakpos = output[1]
            image = image[pixels]
            peakpos = peakpos[pixels]

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

    def FinishRun(self):
        self.peakpos_all = np.array(self.peakpos_all, dtype=float)
        if self.counter_ped > 0:
            self.peakpos_ped = np.array(self.peakpos_ped, dtype=float)

        # rms, percentile, mean deviation, median, mean,
        self.image_all = np.array(self.image_all, dtype=float)
        self.image_all_median = np.median(self.image_all, axis=0)
        self.image_all_average = np.mean(self.image_all, axis=0)
        self.image_all_std = np.std(self.image_all, axis=0)
        self.image_all_rms = np.sqrt(np.sum(self.image_all**2, axis=0))

        self.ped_all = np.array(self.ped_all, dtype=float)
        self.ped_all_average = np.mean(self.ped_all, axis=0)
        self.ped_all_median = np.median(self.ped_all, axis=0)
        self.ped_all_std = np.std(self.ped_all, axis=0)
        self.ped_all_rms = np.sqrt(np.sum(self.ped_all**2, axis=0))

        if self.counter_ped > 0:
            self.image_ped = np.array(self.image_ped, dtype=float)
            self.image_ped_median = np.median(self.image_ped, axis=0)
            self.image_ped_average = np.mean(self.image_ped, axis=0)
            self.image_ped_std = np.std(self.image_ped, axis=0)
            self.image_ped_rms = np.sqrt(np.sum(self.image_ped**2, axis=0))

            self.ped_ped = np.array(self.ped_ped, dtype=float)
            self.ped_ped_average = np.mean(self.ped_ped, axis=0)
            self.ped_ped_median = np.median(self.ped_ped, axis=0)
            self.ped_ped_std = np.std(self.ped_ped, axis=0)
            self.ped_ped_rms = np.sqrt(np.sum(self.ped_ped**2, axis=0))

    def GetResults(self):
        if self.k == 0:
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-HIGH-GAIN"
            ] = self.image_all_average
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-MEDIAN-HIGH-GAIN"
            ] = self.image_all_median
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-RMS-HIGH-GAIN"
            ] = self.image_all_rms
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-STD-HIGH-GAIN"
            ] = self.image_all_std

            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-AVERAGE-HIGH-GAIN"
            ] = self.ped_all_average
            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-MEDIAN-HIGH-GAIN"
            ] = self.ped_all_median
            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-RMS-HIGH-GAIN"
            ] = self.ped_all_rms
            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-STD-HIGH-GAIN"
            ] = self.ped_all_std

            if self.counter_ped > 0:
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-AVERAGE-HIGH-GAIN"
                ] = self.image_ped_average
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-MEDIAN-HIGH-GAIN"
                ] = self.image_ped_median
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-RMS-HIGH-GAIN"
                ] = self.image_ped_rms
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-STD-HIGH-GAIN"
                ] = self.image_ped_std

                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-AVERAGE-HIGH-GAIN"
                ] = self.ped_ped_average
                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-MEDIAN-HIGH-GAIN"
                ] = self.ped_ped_median
                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-RMS-HIGH-GAIN"
                ] = self.ped_ped_rms
                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-STD-HIGH-GAIN"
                ] = self.ped_ped_std

        if self.k == 1:
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-LOW-GAIN"
            ] = self.image_all_average
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-MEDIAN-LOW-GAIN"
            ] = self.image_all_median
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-RMS-LOW-GAIN"
            ] = self.image_all_rms
            self.ChargeInt_Results_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-STD-LOW-GAIN"
            ] = self.image_all_std

            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-AVERAGE-LOW-GAIN"
            ] = self.ped_all_average
            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-MEDIAN-LOW-GAIN"
            ] = self.ped_all_median
            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-RMS-LOW-GAIN"
            ] = self.ped_all_rms
            self.ChargeInt_Results_Dict[
                "PED-INTEGRATION-IMAGE-ALL-STD-LOW-GAIN"
            ] = self.ped_all_std

            if self.counter_ped > 0:
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-AVERAGE-LOW-GAIN"
                ] = self.image_ped_average
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-MEDIAN-LOW-GAIN"
                ] = self.image_ped_median
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-RMS-LOW-GAIN"
                ] = self.image_ped_rms
                self.ChargeInt_Results_Dict[
                    "CHARGE-INTEGRATION-PED-ALL-STD-LOW-GAIN"
                ] = self.image_ped_std

                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-AVERAGE-LOW-GAIN"
                ] = self.ped_ped_average
                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-MEDIAN-LOW-GAIN"
                ] = self.ped_ped_median
                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-RMS-LOW-GAIN"
                ] = self.ped_ped_rms
                self.ChargeInt_Results_Dict[
                    "PED-INTEGRATION-PED-ALL-STD-LOW-GAIN"
                ] = self.ped_ped_std

        return self.ChargeInt_Results_Dict

    def PlotResults(self, name, FigPath):
        # titles = ['All', 'Pedestals']
        if self.k == 0:
            gain_c = "High"
        if self.k == 1:
            gain_c = "Low"

        # Charge integration MEAN plot
        if self.counter_evt > 0:
            fig1, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            # disp = CameraDisplay(self.subarray.tels[0].camera)
            disp.image = self.image_all_average[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration Mean %s Gain (ALL)" % gain_c)

            full_name = name + "_ChargeInt_Mean_%sGain_All.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-%s-GAIN" % gain_c
            ] = fig1
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        if self.counter_ped > 0:
            fig2, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            disp.image = self.image_ped_average[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration Mean %s Gain (PED)" % gain_c)

            full_name = name + "_ChargeInt_Mean_%sGain_Ped.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-AVERAGE-%s-GAIN" % gain_c
            ] = fig2
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-AVERAGE-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        # Charge integration MEDIAN plot
        if self.counter_evt > 0:
            fig3, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            disp.image = self.image_all_median[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration Median %s Gain (ALL)" % gain_c)

            full_name = name + "_ChargeInt_Median_%sGain_All.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-MEDIAN-%s-GAIN" % gain_c
            ] = fig3
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-MEDIAN-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        if self.counter_ped > 0:
            fig4, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            disp.image = self.image_ped_median[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration Median %s Gain (PED)" % gain_c)

            full_name = name + "_ChargeInt_Median_%sGain_Ped.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-MEDIAN-%s-GAIN" % gain_c
            ] = fig4
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-MEDIAN-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        # Charge integration STD plot
        if self.counter_evt > 0:
            fig5, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            disp.image = self.image_all_std[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration STD %s Gain (ALL)" % gain_c)

            full_name = name + "_ChargeInt_Std_%sGain_All.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-STD-%s-GAIN" % gain_c
            ] = fig5
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-STD-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        if self.counter_ped > 0:
            fig6, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            disp.image = self.image_ped_std[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration STD %s Gain (PED)" % gain_c)

            full_name = name + "_ChargeInt_Std_%sGain_Ped.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-STD-%s-GAIN" % gain_c
            ] = fig6
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-STD-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        # Charge integration RMS plot
        if self.counter_evt > 0:
            fig7, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            disp.image = self.image_all_rms[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration RMS %s Gain (ALL)" % gain_c)

            full_name = name + "_ChargeInt_Rms_%sGain_All.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-RMS-%s-GAIN" % gain_c
            ] = fig7
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-ALL-RMS-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        if self.counter_ped > 0:
            fig8, disp = plt.subplots()
            disp = CameraDisplay(self.camera[~self.pixelBAD])
            disp.image = self.image_ped_rms[~self.pixelBAD]
            disp.cmap = plt.cm.coolwarm
            disp.axes.text(
                2,
                -0.8,
                f"{gain_c} gain integrated charge (DC)",
                fontsize=12,
                rotation=90,
            )
            disp.add_colorbar()

            plt.title("Charge Integration RMS %s Gain (PED)" % gain_c)

            full_name = name + "_ChargeInt_Rms_%sGain_Ped.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-RMS-%s-GAIN" % gain_c
            ] = fig8
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-IMAGE-PED-RMS-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        # Charge integration SPECTRUM
        if self.counter_evt > 0:
            fig9, disp = plt.subplots()
            for i in range(self.Pix):
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
            plt.title("Charge spectrum %s gain (ALL)" % gain_c)

            full_name = name + "_Charge_Spectrum_%sGain_All.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-ALL-%s-GAIN" % gain_c
            ] = fig9
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-ALL-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        if self.counter_ped > 0:
            fig10, disp = plt.subplots()
            for i in range(self.Pix):
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
            plt.title("Charge spectrum %s gain (PED)" % gain_c)

            full_name = name + "_Charge_Spectrum_%sGain_Ped.png" % gain_c
            FullPath = FigPath + full_name
            self.ChargeInt_Figures_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-PED-%s-GAIN" % gain_c
            ] = fig10
            self.ChargeInt_Figures_Names_Dict[
                "CHARGE-INTEGRATION-SPECTRUM-PED-%s-GAIN" % gain_c
            ] = FullPath

            plt.close()

        return self.ChargeInt_Figures_Dict, self.ChargeInt_Figures_Names_Dict
