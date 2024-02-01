import logging
import sys

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class StdoutRecord:
    def __init__(self, keyword):
        self.console = sys.stdout
        self.keyword = keyword
        self.output = []

    def write(self, message):
        if self.keyword in message:
            self.console.write(message)
            self.console.write("\n")
            self.output.append(message)

    def flush(self):
        self.console.flush()
