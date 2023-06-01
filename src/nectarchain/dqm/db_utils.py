from ZODB import DB
from ZEO import ClientStorage
import persistent
import transaction

class SaveDB(persistent.Persistent):
    def __int__(self):
        addr = 'localhost', 8100
        zeo = ClientStorage.ClientStorage(addr)
        self.db = DB(zeo)
        conn = self.db.open()
        self.root = conn.root()

    def insert(self, key=None, value=None):
        self.root[key] = value
    def commit_and_close(self):
        transaction.commit()
        self.db.close()

    def abort_and_close(self):
        transaction.abort()
        self.db.close()