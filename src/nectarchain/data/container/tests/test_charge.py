from nectarchain.data.container import ChargesContainer,ChargesContainerIO
from nectarchain.makers import ChargesMaker
import glob
import numpy as np

def create_fake_chargeContainer() : 
    nevents = TestChargesContainer.nevents
    npixels = TestChargesContainer.npixels
    rng = np.random.default_rng()
    return ChargesContainer(
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10]),
        nevents =nevents,
        npixels =npixels,  
        charges_hg = rng.integers(low=0, high=1000, size= (nevents,npixels)),
        charges_lg = rng.integers(low=0, high=1000, size= (nevents,npixels)),
        peak_hg = rng.integers(low=0, high=60, size= (nevents,npixels)),
        peak_lg = rng.integers(low=0, high=60, size= (nevents,npixels)),
        run_number = TestChargesContainer.run_number,
        camera = 'TEST',
        broken_pixels_hg = rng.integers(low=0, high=1, size= (nevents,npixels)),
        broken_pixels_lg = rng.integers(low=0, high=1, size= (nevents,npixels)),
        ucts_timestamp =rng.integers(low=0, high=100, size= (nevents)),
        ucts_busy_counter =rng.integers(low=0, high=100, size= (nevents)),
        ucts_event_counter =rng.integers(low=0, high=100, size= (nevents)),
        event_type =rng.integers(low=0, high=1, size= (nevents)),
        event_id =rng.integers(low=0, high=1000, size= (nevents)),
        trig_pattern_all = rng.integers(low=0, high=1, size= (nevents,npixels,4)),
        trig_pattern = rng.integers(low=0, high=1, size= (nevents,npixels)),
        multiplicity =rng.integers(low=0, high=1, size= (nevents))
    )

class TestChargesContainer:
    run_number = 1234
    nevents = 140
    npixels = 10
    # Tests that a ChargeContainer object can be created with valid input parameters.
    def test_create_charge_container(self):
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = TestChargesContainer.nevents
        npixels = TestChargesContainer.npixels
        run_number = TestChargesContainer.run_number
        charges_hg = np.random.randn(nevents,npixels)
        charges_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        method = 'FullWaveformSum'
        charge_container = ChargesContainer(
            charges_hg = charges_hg ,
            charges_lg = charges_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels,
            method = method
        )
    
        assert np.allclose(charge_container.charges_hg,charges_hg)
        assert np.allclose(charge_container.charges_lg,charges_lg)
        assert np.allclose(charge_container.peak_hg,peak_hg)
        assert np.allclose(charge_container.peak_lg,peak_lg)
        assert charge_container.run_number == run_number
        assert charge_container.pixels_id.tolist() == pixels_id.tolist()
        assert charge_container.nevents == nevents
        assert charge_container.npixels == npixels
        assert charge_container.method == method

    # Tests that the from_waveforms method can be called with a valid waveformContainer and method parameter.
    #def test_from_waveforms_valid_input(self):
    #    waveform_container = WaveformsContainer(...)
    #    method = 'FullWaveformSum'
    #
    #    charge_container = ChargeContainer.from_waveforms(waveform_container, method)
    #
    #    assert isinstance(charge_container, ChargeContainer)

    # Tests that the ChargeContainer object can be written to a file and the file is created.
    def test_write_charge_container(self, tmp_path = "/tmp"):
        charge_container = create_fake_chargeContainer()
        tmp_path += f"/{np.random.randn(1)[0]}"
    
        ChargesContainerIO.write(tmp_path,charge_container)
    
        assert len(glob.glob(f"{tmp_path}/charge_run{TestChargesContainer.run_number}.fits")) == 1

    # Tests that a ChargeContainer object can be loaded from a file and the object is correctly initialized.
    def test_load_charge_container(self, tmp_path = "/tmp"):
        charge_container = create_fake_chargeContainer()
        tmp_path += f"/{np.random.randn(1)[0]}"
    
        ChargesContainerIO.write(tmp_path,charge_container)
    
        loaded_charge_container = ChargesContainerIO.load(tmp_path,TestChargesContainer.run_number )
    
        assert isinstance(loaded_charge_container, ChargesContainer)
        assert np.allclose(loaded_charge_container.charges_hg,charge_container.charges_hg)
        assert np.allclose(loaded_charge_container.charges_lg,charge_container.charges_lg)
        assert np.allclose(loaded_charge_container.peak_hg,charge_container.peak_hg)
        assert np.allclose(loaded_charge_container.peak_lg,charge_container.peak_lg)
        assert loaded_charge_container.run_number == charge_container.run_number
        assert loaded_charge_container.pixels_id.tolist() == charge_container.pixels_id.tolist()
        assert loaded_charge_container.nevents == charge_container.nevents
        assert loaded_charge_container.npixels == charge_container.npixels
        assert loaded_charge_container.method == charge_container.method

    # Tests that the ChargeContainer object can be sorted by event_id and the object is sorted accordingly.
    def test_sort_charge_container(self):  
        charge_container = create_fake_chargeContainer()
    
        sorted_charge_container = ChargesMaker.sort(charge_container)
    
        assert sorted_charge_container.event_id.tolist() == sorted(charge_container.event_id.tolist())

    # Tests that the run_number, pixels_id, npixels, nevents, method, multiplicity, and trig_pattern properties of the ChargeContainer object can be accessed and the values are correct.
    def test_access_properties(self):
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charges_hg = np.random.randn(nevents,npixels)
        charges_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = 1234
        method = 'FullWaveformSum'
        charge_container =  ChargesContainer(
            charges_hg = charges_hg ,
            charges_lg = charges_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels,
            method = method
        )
    
        assert charge_container.run_number == run_number
        assert charge_container.pixels_id.tolist() == pixels_id.tolist()
        assert charge_container.npixels == npixels
        assert charge_container.nevents == nevents
        assert charge_container.method == method