from nectarchain.makers.calibration.core import CalibrationMaker
import numpy as np
from pathlib import Path

import pytest

class CalibrationMakerforTest(CalibrationMaker) : 
    _reduced_name = "test"
    def make() : 
        pass


class TestflatfieldMaker:

    # Tests that the constructor initializes the object with the correct attributes and metadata when valid input is provided
    def test_constructor_with_valid_input(self):
        pixels_id = [1, 2, 3]
        calibration_maker = CalibrationMakerforTest(pixels_id)
    
        assert np.equal(calibration_maker._pixels_id,pixels_id).all()
        assert np.equal(calibration_maker._results[calibration_maker.PIXELS_ID_COLUMN],np.array(pixels_id)).all()
        assert calibration_maker._results.meta[calibration_maker.NP_PIXELS] == len(pixels_id)
        assert isinstance(calibration_maker._results.meta['comments'],str)


    # Tests that the constructor raises an error when a non-iterable pixels_id is provided
    def test_constructor_with_non_iterable_pixels_id(self):
        pixels_id = 123
        with pytest.raises(TypeError):
            CalibrationMakerforTest(pixels_id)

    # Tests that saving the results to an existing file with overwrite=False raises an error
    def test_save_to_existing_file_with_overwrite_false(self, tmp_path = Path('/tmp')):
        pixels_id = [1, 2, 3]
        calibration_maker = CalibrationMakerforTest(pixels_id)

        # Create a temporary file
        file_path = tmp_path / "results_Calibration.ecsv"
        file_path.touch()
    
        with pytest.raises(FileExistsError):
            calibration_maker.save(file_path, overwrite=False)

    # Tests that changing the pixels_id attribute updates the results table with the expected values
    def test_change_pixels_id_attribute(self):
        pixels_id = [1, 2, 3]
        calibration_maker = CalibrationMakerforTest(pixels_id)
    
        new_pixels_id = [4, 5, 6]
        calibration_maker._pixels_id = np.array(new_pixels_id)
    
        assert np.equal(calibration_maker._pixels_id,new_pixels_id).all()


        # Tests that an instance of CalibrationMaker cannot be created with an empty list of pixel ids as input.
    def test_create_instance_with_empty_pixel_ids(self):
        with pytest.raises(TypeError):
            gain_maker = CalibrationMakerforTest([])
