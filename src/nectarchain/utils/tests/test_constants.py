from nectarchain.utils.constants import (
    ALLOWED_CAMERAS,
    FLATFIELD_DEFAULT,
    GAIN_DEFAULT,
    GAIN_LINEAR_RANGE,
    GROUP_NAMES_PEDESTAL,
    HILO_DEFAULT,
    PEDESTAL_DEFAULT,
    get_allowed_cameras,
)


class TestGetAllowedCameras:
    def test_returns_nine_cameras(self):
        cameras = get_allowed_cameras()
        assert len(cameras) == 9

    def test_includes_all_expected(self):
        cameras = get_allowed_cameras()
        assert "NectarCAMQM" in cameras
        assert all(c.startswith("NectarCAM") for c in cameras)


class TestConstants:
    def test_allowed_cameras(self):
        assert "NectarCAMQM" in ALLOWED_CAMERAS

    def test_pedestal_default(self):
        assert PEDESTAL_DEFAULT == 250.0

    def test_gain_default(self):
        assert GAIN_DEFAULT == 58.0

    def test_hilo_default(self):
        assert HILO_DEFAULT == 13.0

    def test_flatfield_default(self):
        assert FLATFIELD_DEFAULT == 1.0

    def test_gain_linear_range(self):
        assert GAIN_LINEAR_RANGE == [10, 200]

    def test_group_names_pedestal(self):
        assert GROUP_NAMES_PEDESTAL == ["data", "data_combined"]
