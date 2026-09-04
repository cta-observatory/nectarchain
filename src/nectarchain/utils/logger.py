import copy
import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class KeepLoggingUnchanged:
    """Context manager that preserves logging internals across a block.

    Saves and restores ``logging._nameToLevel``, ``logging._levelToName``,
    and ``logging._srcfile``.
    """

    def __init__(self):
        """Initialise the context manager with placeholders."""
        self._nameToLevel = None
        self._levelToName = None
        self._srcfile = None
        # self._lock = None

    def __enter__(self):
        """Save logging internal state."""
        self._nameToLevel = copy.copy(logging._nameToLevel)
        self._levelToName = copy.copy(logging._levelToName)
        self._srcfile = copy.copy(logging._srcfile)
        # self._lock = copy.copy(logging._lock)

    def __exit__(self, type, value, traceback):
        """Restore logging internal state."""
        logging._levelToName = self._levelToName
        logging._nameToLevel = self._nameToLevel
        logging._srcfile = self._srcfile
        # logging._lock = self._lock
