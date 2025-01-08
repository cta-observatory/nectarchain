import logging
from abc import abstractmethod

from .core import NectarCAMComponent
logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers



__all__ = ["GainNectarCAMComponent"]


class GainNectarCAMComponent(NectarCAMComponent):
    @abstractmethod
    def finish(self):
        pass
