import os

import numpy as np
from ctapipe.containers import EventType
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["MeanCameraDisplayHighLowGain"]


class MeanCameraDisplayHighLowGain(DQMSummary):
    """Compute and display mean camera images for physics and pedestal events.

    Averages waveform sums across events and samples to produce
    per-pixel mean charge maps.
    """

    def __init__(self, gaink, r0=False):
        """Initialize mean camera display processor.

        Parameters
        ----------
        gaink : int
            Gain index (0 for high gain, 1 for low gain).
        r0 : bool, optional
            Whether to use r0 waveforms (default False).
        """
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.tel_id = None
        self.counter_evt = None
        self.counter_ped = None
        self.camera = None
        self.cmap = None
        self.camera_average = None
        self.camera_average_ped = None
        self.camera_average_over_events = None
        self.camera_average_over_events_over_samp = None
        self.camera_average_ped_over_events = None
        self.camera_average_ped_over_events_over_samp = None
        self.pixel_bad = None
        self.pixel_ids = None
        self.MeanCameraDisplay_Results_Dict = {}
        self.MeanCameraDisplay_Figures_Dict = {}
        self.MeanCameraDisplay_Figures_Names_Dict = {}

        gain_c = "High" if self.k == 0 else "Low"
        self.gain_c = gain_c

        self.figure_keys = {
            "physical": f"CAMERA-AVERAGE-PHY-DISPLAY-{gain_c}-GAIN",
            "pedestals": f"CAMERA-AVERAGE-PED-DISPLAY-{gain_c}-GAIN",
        }

        self.figure_filenames = {
            "physical": f"_Camera_Mean_{gain_c}Gain.png",
            "pedestals": f"_Pedestal_Mean_{gain_c}Gain.png",
        }

        super().__init__(r0)

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        """Configure the processor for a new run.

        Parameters
        ----------
        path : str
            Path to the input data file.
        Pix : int
            Number of pixels.
        Samp : int
            Number of waveform samples.
        Reader1 : ctapipe_io_nectarcam.NectarCAMEventSource
            Event reader providing subarray and camera geometry.
        **kwargs
            Additional keyword arguments (unused).
        """
        self.Pix = Pix
        self.Samp = Samp

        self.tel_id = Reader1.subarray.tel_ids[0]

        self.counter_evt = 0
        self.counter_ped = 0

        self.camera = Reader1.subarray.tel[self.tel_id].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )

        self.cmap = "gnuplot2"

        # Pre-allocate accumulation arrays with float32 for memory efficiency
        self.camera_average = np.zeros(Pix, dtype=np.float32)
        self.camera_average_ped = np.zeros(Pix, dtype=np.float32)

        # Initialize pixel mapping - will be set from first event
        self.pixel_bad = None
        self.pixel_ids = None

    def process_event(self, evt, noped):
        """Sum waveform amplitudes for the current event.

        Separates physics and sky-pedestal events into distinct
        accumulators.

        Parameters
        ----------
        evt : ctapipe.io.DataEventContainer
            The event container.
        noped : bool
            Whether to subtract pedestal (unused here).
        """

        pixel_bad = evt.mon.tel[self.tel_id].pixel_status.hardware_failing_pixels
        pixel_ids = evt.nectarcam.tel[self.tel_id].svc.pixel_ids

        # Store pixel bad mask from first event (assumed constant per run)
        if self.pixel_bad is None:
            self.pixel_bad = pixel_bad

        # Handle pixel mapping - create full array if needed
        if len(pixel_ids) < self.Pix:
            padding = np.arange(0, self.Pix - len(pixel_ids), 1, dtype=int)
            pixels = np.concatenate([padding, pixel_ids])
        else:
            pixels = pixel_ids

        # Extract waveforms based on r0/r1 mode
        if self.r0:
            # This should accommodate cases were the shape of waveforms is 2D
            # (1855,60), or 3D (2, 1855, 60) for 2-gain channels or
            # (1, 1855, 60) for single-gain channel
            waveforms = evt.r0.tel[self.tel_id].waveform[self.k]
        else:
            wf = evt.r1.tel[self.tel_id].waveform
            if wf.ndim == 3:
                waveforms = wf[self.k]
            else:
                waveforms = wf

        # Detect event type with boolean flags (following waveforms.py pattern)
        is_ped = evt.trigger.event_type == EventType.SKY_PEDESTAL
        is_phy = evt.trigger.event_type == EventType.SUBARRAY

        # Sum waveforms over samples (axis=-1) to get charge per pixel
        charge_per_pixel = waveforms.sum(axis=-1)

        # Reindex using pixel mapping and accumulate directly into pre-allocated arrays
        charge_mapped = charge_per_pixel[pixels]

        if is_ped:
            self.counter_ped += 1
            self.camera_average_ped += charge_mapped
        elif is_phy:
            self.counter_evt += 1
            self.camera_average += charge_mapped
        else:
            # TODO: add ids for other event types, e.g., dark pedestals
            # TODO: this else is wrong, we should have a separate counter
            # for other event types, e.g., dark pedestals. It has to be implemented.
            self.counter_evt += 1
            self.camera_average += charge_mapped

        return None

    def finish_run(self):
        """Compute mean camera images averaged over events and samples."""
        if self.counter_evt > 0:
            # Compute mean over events
            self.camera_average_over_events = self.camera_average / self.counter_evt
            # Normalize by number of samples
            self.camera_average_over_events_over_samp = (
                self.camera_average_over_events / self.Samp
            )

        if self.counter_ped > 0:
            # Compute mean pedestals over events
            self.camera_average_ped_over_events = (
                self.camera_average_ped / self.counter_ped
            )
            # Normalize by number of samples
            self.camera_average_ped_over_events_over_samp = (
                self.camera_average_ped_over_events / self.Samp
            )

    def get_results(self):
        """Return the mean camera display results dictionary.

        Returns
        -------
        dict
            Dictionary with keys like CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-HIGH-GAIN
            containing per-pixel mean charge arrays.
        """

        if self.counter_evt > 0:
            # Handle NaN/Inf values (following waveforms.py pattern)
            valid_mask = ~(
                np.isinf(self.camera_average_over_events_over_samp)
                | np.isnan(self.camera_average_over_events_over_samp)
            )
            if not np.all(valid_mask):
                self.camera_average_over_events_over_samp[~valid_mask] = 0

            self.MeanCameraDisplay_Results_Dict[
                f"CAMERA-AVERAGE-PHY-OverEVENTS-OverSamp-{self.gain_c.upper()}-GAIN"
            ] = self.camera_average_over_events_over_samp

        if self.counter_ped > 0:
            # Handle NaN/Inf values for pedestal data
            valid_mask_ped = ~(
                np.isinf(self.camera_average_ped_over_events_over_samp)
                | np.isnan(self.camera_average_ped_over_events_over_samp)
            )
            if not np.all(valid_mask_ped):
                self.camera_average_ped_over_events_over_samp[~valid_mask_ped] = 0

            self.MeanCameraDisplay_Results_Dict[
                f"CAMERA-AVERAGE-PED-OverEVENTS-OverSamp-{self.gain_c.upper()}-GAIN"
            ] = self.camera_average_ped_over_events_over_samp

        return self.MeanCameraDisplay_Results_Dict

    def plot_results(self, name, fig_path):
        """Generate mean camera display figures for physics and pedestal data.

        Parameters
        ----------
        name : str
            Run name prefix for output filenames.
        fig_path : str
            Directory path for saving figure files.

        Returns
        -------
        tuple of dict
            (figures_dict, filenames_dict) mapping plot keys to
            matplotlib figures and their save paths.
        """

        if self.counter_evt > 0:
            fig_key = self.figure_keys["physical"]
            full_name = name + self.figure_filenames["physical"]
            fig_path_full = os.path.join(fig_path, full_name)

            fig, disp = plt.subplots()
            disp = CameraDisplay(
                geometry=self.camera[~self.pixel_bad[0]],
                image=self.camera_average_over_events_over_samp[~self.pixel_bad[0]],
                cmap=plt.cm.coolwarm,
            )
            disp.add_colorbar()
            disp.axes.text(2.0, 0, "Charge (DC)", rotation=90)
            plt.title(f"Camera average {self.gain_c} gain (ALL)")

            self.MeanCameraDisplay_Figures_Dict[fig_key] = fig
            self.MeanCameraDisplay_Figures_Names_Dict[fig_key] = fig_path_full
            plt.close()

        if self.counter_ped > 0:
            fig_key = self.figure_keys["pedestals"]
            full_name = name + self.figure_filenames["pedestals"]
            fig_path_full = os.path.join(fig_path, full_name)

            fig, disp = plt.subplots()
            disp = CameraDisplay(
                geometry=self.camera[~self.pixel_bad[0]],
                image=self.camera_average_ped_over_events_over_samp[~self.pixel_bad[0]],
                cmap=plt.cm.coolwarm,
            )
            disp.add_colorbar()
            disp.axes.text(2.0, 0, "Charge (DC)", rotation=90)
            plt.title(f"Camera average {self.gain_c} gain (PED)")

            self.MeanCameraDisplay_Figures_Dict[fig_key] = fig
            self.MeanCameraDisplay_Figures_Names_Dict[fig_key] = fig_path_full
            plt.close()

        return (
            self.MeanCameraDisplay_Figures_Dict,
            self.MeanCameraDisplay_Figures_Names_Dict,
        )
