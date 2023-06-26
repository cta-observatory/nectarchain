from ZODB import DB
from ZEO import ClientStorage
import requests
import transaction


class DQMDB():
    def __init__(self, read_only=True):
        self.server = 'localhost'
        addr = self.server, 8100
        zeo = ClientStorage.ClientStorage(addr, read_only=read_only)
        self.db = DB(zeo)
        conn = self.db.open()
        self.root = conn.root()

    def get_token(self, user=None, password=None):
        action = f'http://{self.server}:8080/NectarCAM/@login'
        headers = """"{'Accept': 'application/json', 'Content-Type': 'application/json'}"""
        credentials = {'login': user, 'password': password}
        result = requests.post(action,
                               headers=headers,
                               json=credentials)
        token = None
        # A 200 HTML response means 'OK' (cf. https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/200)
        if result.status_code == 200:
            token = result.json()["token"]
        return token

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
