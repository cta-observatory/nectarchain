import unittest
from unittest.mock import patch

from nectarchain.trr_verification_package import TestRunner


class TestTestRunner(unittest.TestCase):
    @patch(
        "nectarchain.trr_verification_package.gui.TestRunner.init_ui", return_value=None
    )
    def test_init_pyqt5(self, mock_init_ui):
        try:
            from PyQt5.QtWidgets import QWidget  # noqa: F401

            with patch("QWidget.__init__", return_value=None) as mock_qwidget:
                runner = TestRunner()
                assert isinstance(runner, TestRunner)
                mock_qwidget.assert_called_once()
                mock_init_ui.assert_called_once()
        except ImportError:
            self.skipTest("PyQt5 is not available")
