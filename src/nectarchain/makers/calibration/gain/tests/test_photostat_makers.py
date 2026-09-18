import os
from unittest.mock import patch


class TestPhotoStatisticNectarCAMCalibrationTool:
    def test_default_run_number(self):
        from nectarchain.makers.calibration.gain.photostat_makers import (
            PhotoStatisticNectarCAMCalibrationTool,
        )

        tool = PhotoStatisticNectarCAMCalibrationTool()
        assert tool.run_number == -1
        assert tool.Ped_run_number == -1

    def test_name(self):
        from nectarchain.makers.calibration.gain.photostat_makers import (
            PhotoStatisticNectarCAMCalibrationTool,
        )

        assert (
            PhotoStatisticNectarCAMCalibrationTool().name == "PhotoStatisticNectarCAM"
        )

    def test_run_file_is_none(self):
        from nectarchain.makers.calibration.gain.photostat_makers import (
            PhotoStatisticNectarCAMCalibrationTool,
        )

        assert PhotoStatisticNectarCAMCalibrationTool().run_file is None

    def test_events_per_slice_is_none(self):
        from nectarchain.makers.calibration.gain.photostat_makers import (
            PhotoStatisticNectarCAMCalibrationTool,
        )

        assert PhotoStatisticNectarCAMCalibrationTool().events_per_slice is None

    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_init_output_path(self):
        from nectarchain.makers.calibration.gain.photostat_makers import (
            PhotoStatisticNectarCAMCalibrationTool,
        )

        tool = PhotoStatisticNectarCAMCalibrationTool()
        tool.run_number = 42
        tool.Ped_run_number = 7
        tool.method = "FullWaveformSum"
        tool.extractor_kwargs = {}
        tool._init_output_path()
        assert "PhotoStatisticNectarCAM_FFrun42" in str(tool.output_path)
        assert "Pedrun7" in str(tool.output_path)

    @patch.dict(os.environ, {"NECTARCAMDATA": "/test/data"}, clear=True)
    def test_init_output_path_with_max_events(self):
        from nectarchain.makers.calibration.gain.photostat_makers import (
            PhotoStatisticNectarCAMCalibrationTool,
        )

        tool = PhotoStatisticNectarCAMCalibrationTool()
        tool.run_number = 1
        tool.Ped_run_number = 2
        tool.max_events = 100
        tool.method = "FullWaveformSum"
        tool.extractor_kwargs = {}
        tool._init_output_path()
        assert "maxevents100" in str(tool.output_path)
