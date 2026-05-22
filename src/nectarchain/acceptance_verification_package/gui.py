import sys

from PyQt5.QtWidgets import QApplication

from nectarchain.acceptance_verification_package import hillas_validation
from nectarchain.trr_test_suite.gui import TestRunner


class AcceptanceTestRunner(TestRunner):
    # redefine list of test modules
    test_modules = {
        "Hillas parameter validation": hillas_validation,
    }


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AcceptanceTestRunner()
    sys.exit(app.exec())
