from unittest.mock import MagicMock, patch

from nectarchain.trr_test_suite import TestRunner

try:
    import PyQt6  # noqa: F401

    pyqt_module = "PyQt6"
except ImportError:
    import PyQt5  # noqa: F401

    pyqt_module = "PyQt5"


class TestTestRunner:
    @patch(f"{pyqt_module}.QtWidgets.QWidget.__init__", return_value=None)
    @patch(f"{pyqt_module}.QtWidgets.QApplication", MagicMock())
    @patch("nectarchain.trr_test_suite.gui.TestRunner.init_ui", return_value=None)
    def test_init(self, mock_init_ui, mock_qwidget):
        runner = TestRunner()
        assert isinstance(runner, TestRunner)
        mock_qwidget.assert_called_once()
        mock_init_ui.assert_called_once()
