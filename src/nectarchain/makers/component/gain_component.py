import logging
from abc import abstractmethod

from .core import NectarCAMComponent

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["GainNectarCAMComponent"]


class GainNectarCAMComponent(NectarCAMComponent):
    """
    Abstract base class for gain-related components.

    Subclasses must implement the `finish` method to finalize
    the gain computation.
    """

    @abstractmethod
    def finish(self):
        """
        Finalize the gain computation.

        Returns
        -------
        gain_container
            The container with the computed gain values.
        """
        pass
