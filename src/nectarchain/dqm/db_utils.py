import transaction
from ZEO import ClientStorage
from ZODB import DB

__all__ = ["DQMDB"]


class DQMDB:
    def __init__(self, read_only=True):
        self.server = "localhost"
        addr = self.server, 8100
        zeo = ClientStorage.ClientStorage(addr, read_only=read_only)
        self.db = DB(zeo)
        conn = self.db.open()
        self.root = conn.root()

    def insert(self, key=None, value=None):
        if key is not None and value is not None:
            try:
                self.root[key] = value
                return True
            except AttributeError:
                return False

    def commit_and_close(self):
        transaction.commit()
        self.db.close()

    def abort_and_close(self):
        transaction.abort()
        self.db.close()
