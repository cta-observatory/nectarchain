import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class TooMuchFileException(Exception):
    pass


class DifferentPixelsID(Exception):
    def __init__(self, message):
        self.__message = message

    @property
    def message(self):
        return self.__message


class PedestalValueError(ValueError):
    def __init__(self, message):
        self.__message = message

    @property
    def message(self):
        return self.__message


class MeanValueError(ValueError):
    def __init__(self, message):
        self.__message = message

    @property
    def message(self):
        return self.__message
