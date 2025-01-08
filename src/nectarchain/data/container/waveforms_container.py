import logging

import numpy as np
from ctapipe.containers import Field, Map, partial

from .core import ArrayDataContainer, TriggerMapContainer

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["WaveformsContainer", "WaveformsContainers"]


class WaveformsContainer(ArrayDataContainer):
    """A container that holds information about waveforms from a specific run.

    Fields:
        nsamples (int): The number of samples in the waveforms.
        subarray (SubarrayDescription): The subarray description instance.
        wfs_hg (np.ndarray): An array of high gain waveforms.
        wfs_lg (np.ndarray): An array of low gain waveforms.
    """

    nsamples = Field(
        type=np.uint8,
        description="number of samples in the waveforms",
    )
    # subarray = Field(type=SubarrayDescription,
    # description="The subarray  description")
    wfs_hg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=3, description="high gain waveforms"
    )
    wfs_lg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=3, description="low gain waveforms"
    )


class WaveformsContainers(TriggerMapContainer):
    """Class representing a container for waveforms from specific runs.

    This class inherits from the `TriggerMapContainer` class and is used to store
    trigger or slices of data mappings of `WaveformsContainer` instances.

    Attributes:
        containers (Field): A field representing the trigger or slices of data mapping
        of `WaveformsContainer` instances.
    """

    containers = Field(
        default_factory=partial(Map, WaveformsContainer),
        description="trigger or slices of data mapping of WaveformContainer",
    )
