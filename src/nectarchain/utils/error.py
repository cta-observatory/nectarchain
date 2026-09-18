import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class TooMuchFileException(Exception):
    """Exception raised when there are too many files matching a pattern."""

    pass


class DifferentPixelsID(Exception):
    """Exception raised when two sets of pixel IDs do not match.

    Parameters
    ----------
    message : str
        Description of the mismatch.
    """

    def __init__(self, message):
        """Initialise the exception.

        Parameters
        ----------
        message : str
            Description of the mismatch.
        """
        self.__message = message

    @property
    def message(self):
        """str: Description of the mismatch."""
        return self.__message


class PedestalValueError(ValueError):
    """Exception raised for invalid pedestal values.

    Parameters
    ----------
    message : str
        Description of the error.
    """

    def __init__(self, message):
        """Initialise the exception.

        Parameters
        ----------
        message : str
            Description of the error.
        """
        self.__message = message

    @property
    def message(self):
        """str: Description of the error."""
        return self.__message


class MeanValueError(ValueError):
    """Exception raised for invalid mean values.

    Parameters
    ----------
    message : str
        Description of the error.
    """

    def __init__(self, message):
        """Initialise the exception.

        Parameters
        ----------
        message : str
            Description of the error.
        """
        self.__message = message

    @property
    def message(self):
        """str: Description of the error."""
        return self.__message
