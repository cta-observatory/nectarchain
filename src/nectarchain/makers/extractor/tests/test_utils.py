from nectarchain.makers.extractor.utils import CtapipeExtractor


class TestCtapipeExtractor:
    def test_get_image_peak_time_valid_object(self):
        from ctapipe.containers import DL1CameraContainer

        container = DL1CameraContainer()
        container.image = [1, 2, 3, 4, 5]
        container.peak_time = [10, 4, 5, 6, 9]

        result_image, result_peak_time = CtapipeExtractor.get_image_peak_time(container)

        assert result_image == [1, 2, 3, 4, 5]
        assert result_peak_time == [10, 4, 5, 6, 9]

    def test_get_image_peak_time_empty(self):
        from ctapipe.containers import DL1CameraContainer

        container = DL1CameraContainer()
        container.image = []
        container.peak_time = []

        result_image, result_peak_time = CtapipeExtractor.get_image_peak_time(container)

        assert result_image == []
        assert result_peak_time == []

    def test_get_extractor_kwargs_str_default(self):
        result = CtapipeExtractor.get_extractor_kwargs_str("FullWaveformSum", {})
        assert result == "default"

    def test_get_extractor_kwargs_str_unrecognized_is_default(self):
        result = CtapipeExtractor.get_extractor_kwargs_str(
            "FullWaveformSum", {"window_width": 15}
        )
        assert result == "default"
