import logging
import sys

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


class StdoutRecord:
    """Redirect stdout messages containing a keyword to a list.

    Parameters
    ----------
    keyword : str
        Keyword to filter stdout messages.
    """

    def __init__(self, keyword):
        """Initialise the stdout record.

        Parameters
        ----------
        keyword : str
            Keyword to filter stdout messages.
        """
        self.console = sys.stdout
        self.keyword = keyword
        self.output = []

    def write(self, message):
        """Write message to console if it contains the keyword.

        Parameters
        ----------
        message : str
            Message to write.
        """
        if self.keyword in message:
            self.console.write(message)
            self.console.write("\n")
            self.output.append(message)

    def flush(self):
        """Flush the console output."""
        self.console.flush()
