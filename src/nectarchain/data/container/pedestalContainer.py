import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import numpy as np
from ctapipe.containers import Field

from .core import NectarCAMContainer

from enum import IntFlag, auto, unique


@unique
class PedestalFlagBits(IntFlag):
    '''
    Define the bits corresponding to pedestal-related flags
    NEVENTS:        Number of events below the acceptable minimum value
    MEAN_PEDESTAL:  Mean pedestal outside acceptable range
    STD_SAMPLE:     Pedestal standard deviation for all samples in channel/pixel below the
                    minimum acceptable value
    STD_PIXEL:      Pedestal standard deviation in a chennel/pixel above the maximum
                    acceptable value
    '''
    NEVENTS = auto()
    MEAN_PEDESTAL = auto()
    STD_SAMPLE = auto()
    STD_PIXEL = auto()


class NectarCAMPedestalContainer(NectarCAMContainer):
    """
    A container that holds estimated pedestals

    Fields:
        nsamples (int): The number of samples in the waveforms.
        nevents (np.ndarray): The number of events used to estimate the pedestals for each pixel.
        pixels_id (np.ndarray): An array of pixel IDs.
        ucts_timestamp_min (int): The minimum of the input events UCTS timestamps.
        ucts_timestamp_max (int): The maximum of the input events UCTS timestamps.
        pedestal_mean_hg (np.ndarray): An array of high gain mean pedestals.
        pedestal_mean_lg (np.ndarray): An array of low gain mean pedestals.
        pedestal_std_hg (np.ndarray): An array of standard deviations of high gain pedestals.
        pedestal_std_lg (np.ndarray): An array of standard deviations of low gain pedestals.
    """

    nsamples = Field(
        type=np.uint8,
        description="number of samples in the waveforms",
    )

    nevents = Field(
        type=np.ndarray, dtype=np.float64, ndim=1,
        description="number of events used to estimate the pedestals for each pixel",
    )

    pixels_id = Field(type=np.ndarray, dtype=np.uint16, ndim=1, description="pixel ids")

    ucts_timestamp_min = Field(
        type=np.uint64,
        description="minimum of the input events UCTS timestamps",
    )

    ucts_timestamp_max = Field(
        type=np.uint64,
        description="maximum of the input events UCTS timestamps",
    )

    pedestal_mean_hg = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="high gain mean pedestals"
    )

    pedestal_mean_lg = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="low gain mean pedestals"
    )

    pedestal_std_hg = Field(
        type=np.ndarray, dtype=np.float64, ndim=2,
        description="high gain pedestals standard deviations"
    )
    pedestal_std_lg = Field(
        type=np.ndarray, dtype=np.float64, ndim=2,
        description="low gain pedestals standard deviations"
    )

    pixel_mask = Field(
        type=np.ndarray, dtype=np.int8, ndim=2,
        description="Flag that identifies bad pixels. The flag is a binary mask. \
        The meaning of the mask bits is defined in the class\
         ~nectarchain.data.container.PedestalFlagBits"
    )
