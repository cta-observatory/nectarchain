import sys

from nectarchain.utils.io import StdoutRecord


class TestStdoutRecord:
    def test_constructor(self):
        sr = StdoutRecord(keyword="test")
        assert sr.keyword == "test"
        assert sr.output == []
        assert sr.console is sys.stdout

    def test_write_matching(self):
        sr = StdoutRecord(keyword="test")
        sr.write("this is a test")
        assert sr.output == ["this is a test"]

    def test_write_non_matching(self):
        sr = StdoutRecord(keyword="test")
        sr.write("no match")
        assert sr.output == []

    def test_flush(self):
        StdoutRecord(keyword="x").flush()
