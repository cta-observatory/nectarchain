import logging

import numpy as np
from ctapipe.containers import Field

from .core import NectarCAMContainer

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["FlatFieldContainer"]


class FlatFieldContainer(NectarCAMContainer):
    """
    Container that holds flat field coefficients and other useful information

    Fields:
        run_number (np.uint16): Number of the run
        npixels (np.uint16): Number of pixels
        pixels_id (np.ndarray): Array of pixel's ID
        ucts_timestamp (np.ndarray) : Array of time stamps of each event (UTC)
        event_type (np.ndarray): Array of trigger event types (should be all flat
        field events)
        event_id (np.ndarray): Array of the IDs of each event
        amp_int_per_pix_per_event (np.ndarray): Array of integrated amplitude of each
        pulse
        t_peak_per_pix_per_event (np.ndarray): Array of samples containing the pulse
        maximum
        FF_coef (np.ndarray): Array of flat field coefficients
        bad_pixels (List): List of pixel identified as outliers
    """

    run_number = Field(
        type=np.uint16,
        description="run number associated to the waveforms",
    )

    npixels = Field(
        type=np.uint16,
        description="number of effective pixels",
    )

    pixels_id = Field(type=np.ndarray, dtype=np.uint16, ndim=1, description="pixel ids")

    ucts_timestamp = Field(
        type=np.ndarray, dtype=np.uint64, ndim=1, description="events ucts timestamp"
    )

    event_type = Field(
        type=np.ndarray, dtype=np.uint8, ndim=1, description="trigger event type"
    )

    event_id = Field(type=np.ndarray, dtype=np.uint32, ndim=1, description="event ids")

    amp_int_per_pix_per_event = Field(
        type=np.ndarray,
        dtype=np.float32,
        ndim=3,
        description="amplitude integrated over the window width, per pixel per event",
    )

    # t_peak_per_pix_per_event = Field(
    #    type=np.ndarray,
    #    dtype=np.float32,
    #    ndim=3,
    #    description="sample containing the pulse maximum, per pixel and per event",
    # )

    FF_coef = Field(
        type=np.ndarray,
        dtype=np.float32,
        ndim=3,
        description="the flat field coefficients, per event",
    )

    bad_pixels = Field(
        type=np.ndarray,
        dtype=np.uint16,
        ndim=1,
        description="pixels considered as bad in at least one gain channels",
    )
