import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy


class KeepLoggingUnchanged:
    def __init__(self):
        self._nameToLevel = None
        self._levelToName = None
        self._srcfile = None
        # self._lock = None

    def __enter__(self):
        self._nameToLevel = copy.copy(logging._nameToLevel)
        self._levelToName = copy.copy(logging._levelToName)
        self._srcfile = copy.copy(logging._srcfile)
        # self._lock = copy.copy(logging._lock)

    def __exit__(self, type, value, traceback):
        logging._levelToName = self._levelToName
        logging._nameToLevel = self._nameToLevel
        logging._srcfile = self._srcfile
        # logging._lock = self._lock
