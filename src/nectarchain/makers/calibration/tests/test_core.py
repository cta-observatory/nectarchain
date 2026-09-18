class TestNectarCAMCalibrationTool:
    def test_name(self):
        from nectarchain.makers.calibration.core import NectarCAMCalibrationTool

        assert NectarCAMCalibrationTool().name == "CalibrationTool"

    def test_pixels_id_default(self):
        from nectarchain.makers.calibration.core import NectarCAMCalibrationTool

        assert NectarCAMCalibrationTool().pixels_id is None
