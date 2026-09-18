class TestGainNectarCAMCalibrationTool:
    def test_reload_events_default(self):
        from nectarchain.makers.calibration.gain.core import (
            GainNectarCAMCalibrationTool,
        )

        tool = GainNectarCAMCalibrationTool()
        assert tool.reload_events is False

    def test_set_reload_events(self):
        from nectarchain.makers.calibration.gain.core import (
            GainNectarCAMCalibrationTool,
        )

        tool = GainNectarCAMCalibrationTool()
        tool.reload_events = True
        assert tool.reload_events is True
