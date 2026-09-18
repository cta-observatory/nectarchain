import logging

from nectarchain.utils.logger import KeepLoggingUnchanged


class TestKeepLoggingUnchanged:
    def test_preserves_state(self):
        before = logging._nameToLevel.copy()
        with KeepLoggingUnchanged():
            logging._nameToLevel["X"] = 999
        assert logging._nameToLevel == before

    def test_restores_after_exception(self):
        before = logging._nameToLevel.copy()
        try:
            with KeepLoggingUnchanged():
                logging._nameToLevel["Y"] = 42
                raise ValueError
        except ValueError:
            pass
        assert logging._nameToLevel == before
