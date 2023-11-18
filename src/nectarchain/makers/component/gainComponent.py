import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from astropy.table import Column

from ctapipe_io_nectarcam.containers import NectarCAMDataContainer


from .core import NectarCAMComponent

__all__ = ["GainNectarCAMComponent"]


class GainNectarCAMComponent(NectarCAMComponent):
    @abstractmethod
    def finish(self):
        pass

    