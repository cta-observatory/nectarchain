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
    def __init__(self, gaink, r0=False):
        self.k = gaink
        self.Pix = None
        self.Samp = None
        self.camera = None
        self.change = None
        self.pixel_ids = None
        self.cmap = None
        self.subarray = None
        self.first_event = None
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
        # define number of pixels and samples
        self.Pix = Pix
        self.Samp = Samp
        self.tel_id = Reader1.subarray.tel_ids[0]
        self.camera = Reader1.subarray.tel[self.tel_id].camera.geometry.transform_to(
            EngineeringCameraFrame()
        )
        self.cmap = "gnuplot2"
        self.pixel_ids = np.arange(self.Pix)
        self.subarray = Reader1.subarray

        self.change = np.zeros(len(self.pixel_ids))
        for i, evt1 in enumerate(Reader1):
            self.run_start1 = evt1.nectarcam.tel[self.tel_id].svc.date
            cell_id = evt1.nectarcam.tel[self.tel_id].evt.first_cell_id
            event_id = evt1.index.event_id
            trigger_time = evt1.trigger.time.value
            ping = (cell_id & 0x400).astype(bool)
            # This performs a bitwise AND operation between
            # the value of cell_id and the hexadecimal number 0x400
            # 0x400 in binary is 010000000000 (12 bits)
            # checks if the 11th bit (from the right, starting at 0)
            # is set in cell_id
            if event_id == 1 or i == 0:
                self.ref_parity = evt1.index.event_id % 2
                self.ref_state = np.array(ping)
                self.first_event = np.array(ping.view(np.int8))
                pop1 = self.pixel_ids[ping]
                pop2 = self.pixel_ids[~ping]
                if len(pop1) != 0 and len(pop1) != len(self.pixel_ids):
                    mismatches = min([pop1, pop2], key=len)
                    log.warning(
                        f"The first event has some discreptancies"
                        f" for pixels {mismatches}"
                    )
                    self.change[mismatches] += 1
                    self.event_times.append(trigger_time)
                    self.nchanges += 1

    def process_event(
        self,
        evt,
        noped,
    ):
        trigger_time = evt.trigger.time.value
        trigger_id = evt.index.event_id
        cell_id = evt.nectarcam.tel[self.tel_id].evt.first_cell_id
        ping = (cell_id & 0x400).astype(bool)
        parity = trigger_id % 2
        state = np.array(ping)
        expected = self.ref_state if parity == self.ref_parity else ~self.ref_state

        self.event_id.append(trigger_id)
        if not np.array_equal(state, expected):
            # TODO: add a comment explaining
            # why only logging the first ten elements
            log.warning(
                f"Mismatch: Event {trigger_id}, State: {state[:10]}"
                f" (expected {expected[:10]}), time={trigger_time}"
            )
            self.ref_state = state
            self.ref_parity = parity
            mismatches = np.where(state != expected)[0]
            self.change[mismatches] += 1
            self.event_times.append(trigger_time)
            self.nchanges += 1
            # TODO: add a comment explaining
            # why only logging the first ten elements
            log.warning(
                f"Reset reference. Changes incremented at indices: {mismatches[:10]}..."
            )

    def finish_run(self):
        try:
            self.change = np.array(self.change)
            self.event_id = np.array(self.event_id)
            self.event_times = np.array(self.event_times)

            # self.run_start = (
            #     self.event_times[self.event_id == np.min(self.event_id)] - 100
            # )
            # self.run_end = np.max(self.event_times) + 100
        except Exception as err:
            log.error(f"Data could not be retrieved. Received error code: {err}")

    def get_results(self):
        try:
            self.PingPongMonitoring_Results_Dict[
                "CAMERA-PING-PONG-CHANGES"
            ] = self.change
            self.PingPongMonitoring_Results_Dict[
                "CAMERA-PING-PONG-CHANGES-TIMES"
            ] = self.event_times
        except Exception as err:
            log.error(f"Data could not be retrieved. Received error code: {err}")

        return self.PingPongMonitoring_Results_Dict

    def plot_results(self, name, fig_path):
        try:
            fig_pipo, disp = plt.subplots()
            disp = CameraDisplay(self.camera)
            disp.image = self.change
            disp.cmap = plt.cm.viridis
            bounds = np.linspace(0, int(np.max(self.change)), int(self.nchanges) + 1)
            disp.set_limits_minmax
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

        except Exception as err:
            log.error(f"Received error code: {err}")

        return (
            self.PingPongMonitoring_Figures_Dict,
            self.PingPongMonitoring_Figures_Names_Dict,
        )
