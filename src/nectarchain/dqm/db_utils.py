import transaction
from BTrees.OOBTree import OOBTree
from ZEO import ClientStorage
from ZODB import DB

__all__ = ["DQMDB"]


class DQMDB:
    """ZEO-backed object database client for DQM data persistence."""

    def __init__(self, read_only=True):
        """Initialise a connection to the ZEO database server.

        Parameters
        ----------
        read_only : bool, optional
            Whether the connection should be opened in read-only mode.
        """
        self.server = "localhost"
        addr = self.server, 8100
        zeo = ClientStorage.ClientStorage(addr, read_only=read_only)
        self.db = DB(zeo)
        conn = self.db.open()
        self.root = conn.root()

    def insert(self, key=None, value=None):
        """Insert a value into the database under the given key.

        Parameters
        ----------
        key : str, optional
            The key under which to store the data.
        value : dict, optional
            The data to store (converted to an OOBTree).

        Returns
        -------
        bool
            True if the insert succeeded, False otherwise.
        """
        if key is not None and value is not None:
            try:
                self.root[key] = OOBTree(value)
                return True
            except AttributeError:
                return False

    def commit_and_close(self):
        """Commit the current transaction and close the database connection."""
        transaction.commit()
        self.db.close()

    def abort_and_close(self):
        """Abort the current transaction and close the database connection."""
        transaction.abort()
        self.db.close()
