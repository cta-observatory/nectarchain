import logging
import os

import numpy as np
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.visualization import CameraDisplay
from matplotlib import pyplot as plt

from .dqm_summary_processor import DQMSummary

__all__ = ["PingPongMonitoring"]

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class PingPongMonitoring(DQMSummary):
    """Monitor ping-pong (first-cell-id bit flips) across NectarCAM pixels.

    Detects unexpected bit-11 changes in the first cell ID to identify
    potential synchronisation issues, and counts mismatches per pixel.
    """

    def __init__(self, gaink, r0=False):
        """Initialize ping-pong monitoring processor.

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
        self.camera = None
        self.change = None
        self.pixel_ids = None
        self.cmap = None
        self.subarray = None
        # self.last_event = None
        self.nchanges = 0
        self.ref_state = None
        self.ref_parity = None
        self.tel_id = None
        self.event_id = []
        self.event_times = []
        self.run_start = None
        self.run_end = None
        self.PingPongMonitoring_Results_Dict = {}
        self.PingPongMonitoring_Figures_Dict = {}
        self.PingPongMonitoring_Figures_Names_Dict = {}
        super().__init__(r0)

    def configure_for_run(self, path, Pix, Samp, Reader1, **kwargs):
        """Configure the processor and establish the initial ping-pong state.

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
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.tel_id = Reader1.subarray.tel_ids[0]
        self.camera = Reader1.subarray.tel[self.tel_id].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )
        self.cmap = "gnuplot2"
        self.pixel_ids = np.arange(self.Pix, dtype=np.int64)
        self.subarray = Reader1.subarray

        # Pre-allocate change counter with explicit dtype
        self.change = np.zeros(self.Pix, dtype=np.int64)

        # Get first event to establish reference state
        # Use next(iter()) for efficiency instead of looping
        evt1 = next(iter(Reader1))
        self.run_start1 = evt1.nectarcam.tel[self.tel_id].svc.date
        cell_id = evt1.nectarcam.tel[self.tel_id].evt.first_cell_id
        event_id = evt1.index.event_id
        trigger_time = evt1.trigger.time.value

        # Check bit 11 (0x400 = 1024) of first_cell_id
        ping = (cell_id & 0x400).astype(bool)
        self.ref_parity = event_id % 2
        # ping is already a numpy array from .astype(bool)
        self.ref_state = ping

        # Check for discrepancies in first event
        pop1 = self.pixel_ids[ping]
        pop2 = self.pixel_ids[~ping]
        if len(pop1) != 0 and len(pop1) != len(self.pixel_ids):
            mismatches = min([pop1, pop2], key=len)
            log.warning(
                f"The first event has some discrepancies for pixels {mismatches}"
            )
            self.change[mismatches] += 1
            self.event_times.append(trigger_time)
            self.nchanges += 1

    def process_event(
        self,
        evt,
        noped,
    ):
        """Check ping-pong bit for consistency and count mismatches.

        Parameters
        ----------
        evt : ctapipe.io.DataEventContainer
            The event container.
        noped : bool
            Whether to subtract pedestal (unused here).
        """
        trigger_time = evt.trigger.time.value
        trigger_id = evt.index.event_id
        cell_id = evt.nectarcam.tel[self.tel_id].evt.first_cell_id

        # Check bit 11 (0x400) of first_cell_id
        ping = (cell_id & 0x400).astype(bool)
        parity = trigger_id % 2
        expected = self.ref_state if parity == self.ref_parity else ~self.ref_state

        self.event_id.append(trigger_id)
        if not np.array_equal(ping, expected):
            log.warning(
                f"Mismatch: Event {trigger_id}, ping={ping[:10]}"
                f" (expected {expected[:10]}), time={trigger_time}"
            )
            # Update reference state
            self.ref_state = ping
            self.ref_parity = parity
            mismatches = np.where(ping != expected)[0]
            self.change[mismatches] += 1
            self.event_times.append(trigger_time)
            self.nchanges += 1
            log.warning(
                f"Reset reference. Changes incremented at indices: {mismatches[:10]}..."
            )

    def finish_run(self):
        """Finalise ping-pong change counters and event arrays."""
        # self.change is already a numpy array from configure_for_run
        # Only need to convert the lists
        self.event_id = np.array(self.event_id, dtype=np.int64)
        self.event_times = np.array(self.event_times, dtype=np.float64)

    def get_results(self):
        """Return the ping-pong monitoring results dictionary.

        Returns
        -------
        dict
            Dictionary with keys CAMERA-PING-PONG-CHANGES (per-pixel
            change counts) and CAMERA-PING-PONG-CHANGES-TIMES.
        """
        self.PingPongMonitoring_Results_Dict["CAMERA-PING-PONG-CHANGES"] = self.change
        self.PingPongMonitoring_Results_Dict[
            "CAMERA-PING-PONG-CHANGES-TIMES"
        ] = self.event_times

        return self.PingPongMonitoring_Results_Dict

    def plot_results(self, name, fig_path):
        """Generate a camera display figure of ping-pong change counts.

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
        fig_pipo, disp = plt.subplots()
        disp = CameraDisplay(self.camera)
        disp.image = self.change
        disp.cmap = plt.cm.viridis

        # Handle edge case when there are no changes
        max_change = int(np.max(self.change)) if self.nchanges > 0 else 1
        bounds = np.linspace(0, max_change, min(int(self.nchanges) + 1, max_change + 1))

        disp.set_limits_minmax(zmin=0, zmax=max_change)
        disp.axes.text(2.0, -0.3, "Number of changes", fontsize=12, rotation=90)
        disp.add_colorbar(ticks=bounds)
        plt.title("Camera Ping Pong changes")

        full_name = name + "_CameraPingPongChanges.png"
        full_path = os.path.join(fig_path, full_name)
        self.PingPongMonitoring_Figures_Dict["CAMERA-PING-PONG-CHANGES"] = fig_pipo
        self.PingPongMonitoring_Figures_Names_Dict[
            "CAMERA-PING-PONG-CHANGES"
        ] = full_path

        plt.close()

        return (
            self.PingPongMonitoring_Figures_Dict,
            self.PingPongMonitoring_Figures_Names_Dict,
        )
