from pathlib import Path

import pytest

from nectarchain.makers.calibration.gain import HiLoNectarCAMCalibrationTool


class TestHiLoNectarCAMCalibrationTool:
    @pytest.fixture
    def instance(self):
        tool = HiLoNectarCAMCalibrationTool()
        return tool

    def test_init_output_path(self, instance):
        instance.gain_file = Path("/tmp/gain.h5")
        instance._init_output_path()
        assert instance.output_path == Path("/tmp/gain_hilo_corrected.h5")
