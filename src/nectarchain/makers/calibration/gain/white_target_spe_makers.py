import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["GainMaker", "WhiteTargetSPEMaker"]


class GainMaker:
    """Base class for gain-making algorithms.

    Intended to be subclassed by specific gain calibration implementations.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the gain maker.

        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        """
        raise NotImplementedError("This class is not yet implemented")


class WhiteTargetSPEMaker(GainMaker):
    """Placeholder for white-target SPE calibration.

    The computation of the white target calibration is not yet implemented.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the white-target SPE maker.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the parent initializer.
        **kwargs : dict
            Keyword arguments passed to the parent initializer.
        """
        super().__init__(*args, **kwargs)

    def make(self):
        """Execute the white-target calibration computation.

        Raises
        ------
        NotImplementedError
            Always raised since this method is not yet implemented.
        """
        raise NotImplementedError(
            "The computation of the white target calibration is not yet implemented,"
            "feel free to contribute !:)"
        )
