from nectarchain.makers.calibration.gain.FlatFieldSPEMakers import FlatFieldSPEMaker
from nectarchain.makers.calibration.gain.parameters import Parameter,Parameters
from nectarchain.makers.calibration.gain import FlatFieldSingleHHVSPEMaker,FlatFieldSingleHHVStdSPEMaker
import astropy.units as u
from nectarchain.data.container import ChargeContainer
import numpy as np

import pytest

class FlatFieldSPEMakerforTest(FlatFieldSPEMaker) : 
    def make() : 
        pass

def create_fake_chargeContainer() : 
    pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
    nevents = 40
    npixels = 10
    rng = np.random.default_rng()
    charge_hg = rng.integers(low=0, high=1000, size= (nevents,npixels))
    charge_lg = rng.integers(low=0, high=1000, size= (nevents,npixels))
    peak_hg = rng.integers(low=0, high=60, size= (nevents,npixels))
    peak_lg = rng.integers(low=0, high=60, size= (nevents,npixels))
    run_number = 1234
    return ChargeContainer(
        charge_hg = charge_hg ,
        charge_lg = charge_lg,
        peak_hg = peak_hg,
        peak_lg = peak_lg,
        run_number = run_number,
        pixels_id = pixels_id,
        nevents = nevents,
        npixels = npixels
    )

class TestFlatFieldSPEMaker:

    # Tests that the object can be initialized without errors
    def test_initialize_object(self):
        pixels_id = [2,3,5]
        flat_field_spe_maker = FlatFieldSPEMakerforTest(pixels_id)
        assert isinstance(flat_field_spe_maker, FlatFieldSPEMakerforTest)

    # Tests that parameters can be read from a YAML file
    def test_read_parameters_from_yaml(self):
        pixels_id = [2,3,5]
        flat_field_spe_maker = FlatFieldSPEMakerforTest(pixels_id)
        flat_field_spe_maker.read_param_from_yaml("parameters_signal.yaml")
        assert flat_field_spe_maker.parameters.size == 6
        assert isinstance(flat_field_spe_maker.parameters,Parameters)

    # Tests that parameters can be updated from a YAML file
    def test_update_parameters_from_yaml(self):
        pixels_id = [2,3,5]
        flat_field_spe_maker = FlatFieldSPEMakerforTest(pixels_id)
        flat_field_spe_maker.read_param_from_yaml("parameters_signal.yaml")
        flat_field_spe_maker.read_param_from_yaml("parameters_signalStd.yaml",only_update = True)
        assert flat_field_spe_maker.parameters.parameters[-2].value == 0.697

    # Tests that parameters can be updated from a fit
    def test_update_parameters_from_fit(self):
        pixels_id = [2]
        flat_field_spe_maker = FlatFieldSPEMakerforTest(pixels_id)
        flat_field_spe_maker.read_param_from_yaml("parameters_signal.yaml")
        updated_parameters = flat_field_spe_maker._update_parameters(flat_field_spe_maker.parameters, charge=[1, 2, 3,4,5,6,7,8,9,10], counts=[1,3,9,5,3,5,6,3,2,1])

    # Tests that the table can be updated from parameters
    def test_update_table_from_parameters(self):
        pixels_id = [2,3,5]
        flat_field_spe_maker = FlatFieldSPEMakerforTest(pixels_id)
        flat_field_spe_maker._parameters.append(Parameter(name="param1", value=1, unit=u.dimensionless_unscaled))
        flat_field_spe_maker._parameters.append(Parameter(name="param2", value=2, unit=u.dimensionless_unscaled))

        flat_field_spe_maker._update_table_from_parameters()

        assert "param1" in flat_field_spe_maker._results.colnames
        assert "param1_error" in flat_field_spe_maker._results.colnames
        assert "param2" in flat_field_spe_maker._results.colnames
        assert "param2_error" in flat_field_spe_maker._results.colnames


class TestFlatFieldSingleHHVSPEMaker:

    # Tests that creating an instance of FlatFieldSingleHHVSPEMaker with valid input parameters is successful
    def test_create_instance_valid_input(self):
        charge = [1, 2, 3]
        counts = [10, 20, 30]
        pixels_id = [2,3,5]
        maker = FlatFieldSingleHHVSPEMaker(charge, counts, pixels_id)
        assert isinstance(maker, FlatFieldSingleHHVSPEMaker)

    # Tests that creating an instance of FlatFieldSingleHHVSPEMaker with invalid input parameters raises an error
    def test_create_instance_invalid_input(self):
        charge = [1, 2, 3]
        counts = [10, 20]  # Invalid input, counts and charge must have the same length
        pixels_id = [2,3,5]

        with pytest.raises(Exception):
            FlatFieldSingleHHVSPEMaker(charge, counts,pixels_id)

    # Tests that calling create_from_chargeContainer method with valid input parameters is successful
    def test_create_from_ChargeContainer_valid_input(self):
        chargeContainer = create_fake_chargeContainer()
        maker = FlatFieldSingleHHVSPEMaker.create_from_chargeContainer(chargeContainer)
        assert isinstance(maker, FlatFieldSingleHHVSPEMaker)


    def test_fill_results_table_from_dict(self):
        pass
    def test_NG_Likelihood_Chi2(self):pass
    def test_cost(self):pass
    def test_make_fit_array_from_parameters(self):pass
    def test_run_fit(self) : pass
    def test_make(self):pass
    def test_plot_single(self) : pass
    def test_display(self) : pass

class TestFlatFieldSingleHHVStdSPEMaker:
    def test_create_instance(self):
        charge = [1, 2, 3]
        counts = [10, 20,30]  # Invalid input, counts and charge must have the same length
        pixels_id = [2,3,5]
        instance = FlatFieldSingleHHVStdSPEMaker(charge, counts,pixels_id)
        assert isinstance(instance,FlatFieldSingleHHVStdSPEMaker)


class TestFlatFieldSingleNominalSPEMaker:
    def test_create_instance(self):
        pass
