import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import numpy as np
from ctapipe.containers import Field, Map, partial

from .core import ArrayDataContainer, TriggerMapContainer

__all__ = ["ChargesContainer", "ChargesContainers"]


class ChargesContainer(ArrayDataContainer):
    """
    A container that holds information about charges from a specific run.

    Fields:
      charges_hg (np.ndarray): An array of high gain charges.
      charges_lg (np.ndarray): An array of low gain charges.
      peak_hg (np.ndarray): An array of high gain peak time.
      peak_lg (np.ndarray): An array of low gain peak time.
      method (str): The charge extraction method used.
    """

    charges_hg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=2, description="The high gain charges"
    )
    charges_lg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=2, description="The low gain charges"
    )
    peak_hg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=2, description="The high gain peak time"
    )
    peak_lg = Field(
        type=np.ndarray, dtype=np.uint16, ndim=2, description="The low gain peak time"
    )
    method = Field(type=str, description="The charge extraction method used")


class ChargesContainers(TriggerMapContainer):
    containers = Field(
        default_factory=partial(Map, ChargesContainer),
        description="trigger or slices of data mapping of ChargesContainer",
    )
