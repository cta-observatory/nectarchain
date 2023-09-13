from nectarchain.data.container.charge import ChargeContainers, ChargeContainer
from nectarchain.data.container.waveforms import WaveformsContainers
import glob
import numpy as np

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

class TestChargeContainer:

    # Tests that a ChargeContainer object can be created with valid input parameters.
    def test_create_charge_container(self):
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charge_hg = np.random.randn(nevents,npixels)
        charge_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = 1234
        method = 'FullWaveformSum'
        charge_container = ChargeContainer(
            charge_hg = charge_hg ,
            charge_lg = charge_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels,
            method = method
        )
    
        assert np.allclose(charge_container.charge_hg,charge_hg)
        assert np.allclose(charge_container.charge_lg,charge_lg)
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
    
        charge_container.write(tmp_path)
    
        assert len(glob.glob(f"{tmp_path}/charge_run1234.fits")) == 1

    # Tests that a ChargeContainer object can be loaded from a file and the object is correctly initialized.
    def test_load_charge_container(self, tmp_path = "/tmp"):
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charge_hg = np.random.randn(nevents,npixels)
        charge_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = np.random.randn(1)[0]
        method = 'FullWaveformSum'
        charge_container =  ChargeContainer(
            charge_hg = charge_hg ,
            charge_lg = charge_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels,
            method = method
        )
    
        charge_container.write(tmp_path)
    
        loaded_charge_container = ChargeContainer.from_file(tmp_path, run_number)
    
        assert isinstance(loaded_charge_container, ChargeContainer)
        assert np.allclose(loaded_charge_container.charge_hg,charge_hg)
        assert np.allclose(loaded_charge_container.charge_lg,charge_lg)
        assert np.allclose(loaded_charge_container.peak_hg,peak_hg)
        assert np.allclose(loaded_charge_container.peak_lg,peak_lg)
        assert loaded_charge_container.run_number == run_number
        assert loaded_charge_container.pixels_id.tolist() == pixels_id.tolist()
        assert loaded_charge_container.nevents == nevents
        assert loaded_charge_container.npixels == npixels
        assert loaded_charge_container.method == method

    # Tests that the ChargeContainer object can be sorted by event_id and the object is sorted accordingly.
    def test_sort_charge_container(self):  
        charge_container = create_fake_chargeContainer()
    
        charge_container.sort()
    
        assert charge_container.event_id.tolist() == sorted(charge_container.event_id.tolist())

    # Tests that the run_number, pixels_id, npixels, nevents, method, multiplicity, and trig_pattern properties of the ChargeContainer object can be accessed and the values are correct.
    def test_access_properties(self):
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charge_hg = np.random.randn(nevents,npixels)
        charge_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = 1234
        method = 'FullWaveformSum'
        charge_container =  ChargeContainer(
            charge_hg = charge_hg ,
            charge_lg = charge_lg,
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
        assert charge_container.multiplicity.shape  == (nevents,)
        assert charge_container.trig_pattern.shape == (nevents,4)


class TestChargeContainers:

    # Tests that an instance of ChargeContainers can be created with default arguments
    def test_create_instance_with_default_arguments(self):
        charge_containers = ChargeContainers()
        assert len(charge_containers.chargeContainers) == 0
        assert charge_containers.nChargeContainer == 0

    # Tests that an instance of ChargeContainers can be created from a WaveformsContainers instance
    #def test_create_instance_from_waveforms(self):
    #    waveform_containers = WaveformsContainers(run_number = 1234)
    #    charge_containers = ChargeContainers.from_waveforms(waveform_containers)
    #    assert len(charge_containers.chargeContainers) == waveform_containers.nWaveformsContainer
    #    assert charge_containers.nChargeContainer == waveform_containers.nWaveformsContainer

    # Tests that ChargeContainers can be written to disk
    def test_write_to_disk(self, tmpdir):
        charge_containers = ChargeContainers()

        charge_container = create_fake_chargeContainer()

        charge_containers.append(charge_container)
        path = str(tmpdir)
        charge_containers.write(path)
        assert len(glob.glob(f"{path}/charge_run*1234*.fits")) == 1

    # Tests that ChargeContainers can be loaded from disk
    def test_load_from_disk(self, tmpdir):
        charge_containers = ChargeContainers()

        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charge_hg = np.random.randn(nevents,npixels)
        charge_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = np.random.randn(1)[0]

        charge_container = ChargeContainer(
            charge_hg = charge_hg ,
            charge_lg = charge_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels
        )
        charge_containers.append(charge_container)
        path = str(tmpdir)
        charge_containers.write(path)
        loaded_charge_containers = ChargeContainers.from_file(path, run_number)
        assert len(loaded_charge_containers.chargeContainers) == 1
        assert loaded_charge_containers.nChargeContainer == 1

    # Tests that a ChargeContainer can be appended to a ChargeContainers instance
    def test_append_charge_container(self):
        charge_containers = ChargeContainers()
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charge_hg = np.random.randn(nevents,npixels)
        charge_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = 1234

        charge_container = ChargeContainer(
            charge_hg = charge_hg ,
            charge_lg = charge_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels
        )
        charge_containers.append(charge_container)
        assert len(charge_containers.chargeContainers) == 1
        assert charge_containers.nChargeContainer == 1

    # Tests that a ChargeContainers instance can be merged into a single ChargeContainer
    def test_merge_into_single_charge_container(self):
        charge_containers = ChargeContainers()
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charge_hg = np.random.randn(nevents,npixels)
        charge_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = 1234

        charge_container1 = ChargeContainer(
            charge_hg = charge_hg ,
            charge_lg = charge_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels
        )
        pixels_id = np.array([2,4,3,8,6,9,7,1,5,10])
        nevents = 40
        npixels = 10
        charge_hg = np.random.randn(nevents,npixels)
        charge_lg = np.random.randn(nevents,npixels)
        peak_hg = np.random.randn(nevents,npixels)
        peak_lg = np.random.randn(nevents,npixels)
        run_number = 1234

        charge_container2 = ChargeContainer(
            charge_hg = charge_hg ,
            charge_lg = charge_lg,
            peak_hg = peak_hg,
            peak_lg = peak_lg,
            run_number = run_number,
            pixels_id = pixels_id,
            nevents = nevents,
            npixels = npixels
        )
        charge_containers.append(charge_container1)
        charge_containers.append(charge_container2)
        merged_charge_container = charge_containers.merge()
        assert merged_charge_container.nevents == charge_container1.nevents + charge_container2.nevents