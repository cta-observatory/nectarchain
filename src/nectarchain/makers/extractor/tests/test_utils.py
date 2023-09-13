import pytest

class TestCtapipeExtractor:
    
    @pytest.mark.skip('numba conflict')
    # Tests that the function returns the image and peak_time values from a valid DL1CameraContainer object.
    def test_get_image_peak_time_valid_object(self):
        from ctapipe.containers import DL1CameraContainer
        from nectarchain.makers.extractor.utils import CtapipeExtractor
        # Create a valid DL1CameraContainer object
        container = DL1CameraContainer()
        container.image = [1, 2, 3, 4, 5]
        container.peak_time = [10, 4, 5, 6, 9]
    
        # Call the function under test
        result_image, result_peak_time = CtapipeExtractor.get_image_peak_time(container)
    
        # Check the result
        assert result_image == [1, 2, 3, 4, 5]
        assert result_peak_time == [10, 4, 5, 6, 9]