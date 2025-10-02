import sys

import pytest
import transaction
from ZODB import DB


@pytest.mark.skipif(sys.platform == "darwin")
class TestDQMDB:
    @classmethod
    def setup_class(cls):
        # Perform tests in in-memory ZODB instance
        cls.db = DB(None)
        cls.conn = cls.db.open()
        cls.root = cls.conn.root()

    def test_db(self):
        assert self.root is not None

    def test_insert(self):
        key = "mykey"
        value = "myvalue"
        self.root[key] = value
        assert self.root[key] == value

    def test_commit_and_close(self):
        transaction.commit()
        self.db.close()
        assert self.conn.opened is None

    def test_abort_and_close(self):
        transaction.abort()
        self.db.close()
        assert self.conn.opened is None
