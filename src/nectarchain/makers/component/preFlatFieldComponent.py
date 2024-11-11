import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from ctapipe.containers import EventType, Field
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.core import Component
from ctapipe.core.traits import ComponentNameList, Dict, Float, Integer, List, Tuple
from ctapipe.instrument import CameraGeometry
from ctapipe.io import HDF5TableReader
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from nectarchain.data.container import (
    ArrayDataContainer,
    NectarCAMContainer,
    TriggerMapContainer,
)
from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.component import ArrayDataComponent, NectarCAMComponent
from nectarchain.utils import ComponentUtils

__all__ = ["preFlatFieldComponent"]


class preFlatFieldComponent(NectarCAMComponent):
    window_shift = Integer(
        default_value=5,
        help="the time in ns before the peak to integrate charge",
    ).tag(config=True)

    window_width = Integer(
        default_value=12,
        help="the duration of the extraction window in ns",
    ).tag(config=True)
    ## --< final window is 14 samples ! >--

    g = Float(
        default_value=58.0,
        help="defaut gain value",
    ).tag(config=True)

    hi_lo_ratio = Float(
        default_value=13.0,
        help="defaut gain value",
    ).tag(config=True)

    ## Simple function to substract the pedestal using the 20 first samples of each trace
    def substract_pedestal(wfs, window=20):
        ped_mean = np.mean(wfs[0][:, :, 0:window], axis=2)
        wfs_pedsub = wfs - np.expand_dims(ped_mean, axis=-1)
        return wfs_pedsub

    ## Function to make an array that will be used as a mask on the waveform for the calculation of the integrated amplitude of the signal.
    def make_masked_array(t_peak, window_shift, window_width):
        masked_wfs = [
            np.array([np.zeros(constants.N_SAMPLES)] * constants.N_PIXELS)
        ] * constants.N_GAINS
        masked_wfs = np.array(masked_wfs)

        t_signal_start = t_peak - window_shift
        t_signal_stop = t_peak + window_width - window_shift

        for g in range(0, constants.N_GAINS):
            for i in range(0, constants.N_PIXELS):
                masked_wfs[g][i][
                    t_signal_start[0, g, i] : t_signal_stop[0, g, i]
                ] = True
                # --< I didn't find a better way to do than using this masked array >--

    return masked_wfs

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )

        self.__ucts_timestamp = []
        self.__event_type = []
        self.__event_id = []
        self.__amp_int_per_pix_per_event = []
        self.__FF_coef = []

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        wfs = []
        wfs_pedsub = []

        if event.trigger.event_type.value == EventType.FLATFIELD.value:
            # print("event :", (self.__event_id, self.__event_type))
            self.__event_id.append(np.uint32(event.index.event_id))
            self.__event_type.append(event.trigger.event_type.value)
            self.__ucts_timestamp.append(event.nectarcam.tel[0].evt.ucts_timestamp)

            wfs.append(event.r0.tel[0].waveform)  # not saved

            # substract pedestal using the mean of the 20 first samples
            wfs_pedsub = substract_pedestal(wfs, 20)

            # get the masked array for integration window
            t_peak = np.argmax(wfs_pedsub, axis=3)
            masked_wfs = make_masked_array(t_peak, self.window_shift, self.window_width)

            # get integrated amplitude and mean amplitude over all pixels per event
            amp_int_per_pix_per_event = np.sum(
                wfs_pedsub[0], axis=2, where=masked_wfs.astype("bool")
            )
            self.__amp_int_per_pix_per_event.append(amp_int_per_pix_per_event)
            mean_amp_cam_per_event = np.mean(amp_int_per_pix_per_event, axis=-1)

            # get efficiency and flat field coefficient
            gain = [self.g, self.g / self.hi_lo_ratio]

            eff = np.divide(
                (amp_int_per_pix_per_event[:] / (np.expand_dims(gain[:], axis=-1))),
                np.expand_dims((mean_amp_cam_per_event[:] / gain[:]), axis=-1),
            )  # efficacité relative
            FF_coef = np.ma.array(1.0 / eff, mask=eff == 0)
            self.__FF_coef.append(FF_coef)

    def finish(self):
        output = FlatFieldContainer(
            run_number=FlatFieldContainer.fields["run_number"].type(
                self._run_number
            ),
            npixels=FlatFieldContainer.fields["npixels"].type(self._npixels),
            pixels_id=FlatFieldContainer.fields["pixels_id"].dtype.type(
                self._pixels_id
            ),
            ucts_timestamp=FlatFieldContainer.fields["ucts_timestamp"].dtype.type(
                self.__ucts_timestamp
            ),
            event_type=FlatFieldContainer.fields["event_type"].dtype.type(
                self.__event_type
            ),
            event_id=FlatFieldContainer.fields["event_id"].dtype.type(
                self.__event_id
            ),
            amp_int_per_pix_per_event=FlatFieldContainer.fields[
                "amp_int_per_pix_per_event"
            ].dtype.type(self.__amp_int_per_pix_per_event),
            FF_coef=FlatFieldContainer.fields["FF_coef"].dtype.type(self.__FF_coef),
        )
        return output

