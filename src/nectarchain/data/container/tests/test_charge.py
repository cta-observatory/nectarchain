import glob

import numpy as np
import pytest
from ctapipe.containers import EventType
from ctapipe.io import HDF5TableWriter

from nectarchain.data.container import ChargesContainer, ChargesContainers


def create_fake_chargeContainer():
    nevents = TestChargesContainer.nevents
    npixels = TestChargesContainer.npixels
    rng = np.random.default_rng()
    charge = ChargesContainer(
        pixels_id=np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16),
        nevents=np.uint64(nevents),
        npixels=np.uint16(npixels),
        charges_hg=rng.integers(
            low=0, high=1000, size=(nevents, npixels), dtype=np.uint16
        ),
        charges_lg=rng.integers(
            low=0, high=1000, size=(nevents, npixels), dtype=np.uint16
        ),
        peak_hg=rng.integers(low=0, high=60, size=(nevents, npixels), dtype=np.uint16),
        peak_lg=rng.integers(low=0, high=60, size=(nevents, npixels), dtype=np.uint16),
        run_number=np.uint16(TestChargesContainer.run_number),
        camera="TEST",
        method="test_method",
        broken_pixels_hg=rng.integers(
            low=0, high=1.1, size=(nevents, npixels), dtype=bool
        ),
        broken_pixels_lg=rng.integers(
            low=0, high=1.1, size=(nevents, npixels), dtype=bool
        ),
        ucts_timestamp=rng.integers(low=0, high=100, size=(nevents), dtype=np.uint64),
        ucts_busy_counter=rng.integers(
            low=0, high=100, size=(nevents), dtype=np.uint32
        ),
        ucts_event_counter=rng.integers(
            low=0, high=100, size=(nevents), dtype=np.uint32
        ),
        event_type=rng.integers(low=0, high=1, size=(nevents), dtype=np.uint8),
        event_id=rng.integers(low=0, high=1000, size=(nevents), dtype=np.uint32),
        trig_pattern_all=rng.integers(
            low=0, high=1, size=(nevents, npixels, 4), dtype=bool
        ),
        trig_pattern=rng.integers(low=0, high=1, size=(nevents, npixels), dtype=bool),
        multiplicity=rng.integers(low=0, high=1, size=(nevents), dtype=np.uint16),
    )
    charge.validate()
    return charge


def create_fake_chargeContainers():
    charge_1 = create_fake_chargeContainer()
    charge_2 = create_fake_chargeContainer()
    charge = ChargesContainers()
    charge.containers[EventType.FLATFIELD] = charge_1
    charge.containers[EventType.SKY_PEDESTAL] = charge_2
    return charge


class TestChargesContainer:
    run_number = 1234
    nevents = 140
    npixels = 10

    # Tests that a ChargeContainer object can be created with valid input parameters.
    def test_create_charge_container(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16)
        nevents = np.uint64(TestChargesContainer.nevents)
        npixels = np.uint16(TestChargesContainer.npixels)
        run_number = np.uint16(TestChargesContainer.run_number)
        charges_hg = np.uint16(np.random.randn(nevents, npixels))
        charges_lg = np.uint16(np.random.randn(nevents, npixels))
        peak_hg = np.uint16(np.random.randn(nevents, npixels))
        peak_lg = np.uint16(np.random.randn(nevents, npixels))
        method = "FullWaveformSum"
        charge_container = ChargesContainer(
            charges_hg=charges_hg,
            charges_lg=charges_lg,
            peak_hg=peak_hg,
            peak_lg=peak_lg,
            run_number=run_number,
            pixels_id=pixels_id,
            nevents=nevents,
            npixels=npixels,
            method=method,
        )
        charge_container.validate()
        assert np.allclose(charge_container.charges_hg, charges_hg)
        assert np.allclose(charge_container.charges_lg, charges_lg)
        assert np.allclose(charge_container.peak_hg, peak_hg)
        assert np.allclose(charge_container.peak_lg, peak_lg)
        assert charge_container.run_number == run_number
        assert charge_container.pixels_id.tolist() == pixels_id.tolist()
        assert charge_container.nevents == nevents
        assert charge_container.npixels == npixels
        assert charge_container.method == method

    # Tests that the from_waveforms method can be called with a valid
    # waveformContainer and method parameter.
    # def test_from_waveforms_valid_input(self):
    #    waveform_container = WaveformsContainer(...)
    #    method = 'FullWaveformSum'
    #
    #    charge_container = ChargeContainer.from_waveforms(waveform_container, method)
    #
    #    assert isinstance(charge_container, ChargeContainer)
    # Tests that the ChargeContainer object can be written to a file and the file is
    # created.
    def test_write_charge_container(self, tmp_path="/tmp"):
        charge_container = create_fake_chargeContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="ChargesContainer", containers=charge_container)
        writer.close()

    # Tests that a ChargeContainer object can be loaded from a file and the object is
    # correctly initialized.
    def test_from_hdf5(self, tmp_path="/tmp"):
        charge_container = create_fake_chargeContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="ChargesContainer", containers=charge_container)
        writer.close()

        loaded_charge_container = next(ChargesContainer.from_hdf5(tmp_path))

        assert isinstance(loaded_charge_container, ChargesContainer)
        assert np.allclose(
            loaded_charge_container.charges_hg, charge_container.charges_hg
        )
        assert np.allclose(
            loaded_charge_container.charges_lg, charge_container.charges_lg
        )
        assert np.allclose(loaded_charge_container.peak_hg, charge_container.peak_hg)
        assert np.allclose(loaded_charge_container.peak_lg, charge_container.peak_lg)
        assert loaded_charge_container.run_number == charge_container.run_number
        assert (
            loaded_charge_container.pixels_id.tolist()
            == charge_container.pixels_id.tolist()
        )
        assert loaded_charge_container.nevents == charge_container.nevents
        assert loaded_charge_container.npixels == charge_container.npixels
        assert loaded_charge_container.method == charge_container.method

    def test_access_properties(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10])
        nevents = 40
        npixels = 10
        charges_hg = np.random.randn(nevents, npixels)
        charges_lg = np.random.randn(nevents, npixels)
        peak_hg = np.random.randn(nevents, npixels)
        peak_lg = np.random.randn(nevents, npixels)
        run_number = 1234
        method = "FullWaveformSum"
        charge_container = ChargesContainer(
            charges_hg=charges_hg,
            charges_lg=charges_lg,
            peak_hg=peak_hg,
            peak_lg=peak_lg,
            run_number=run_number,
            pixels_id=pixels_id,
            nevents=nevents,
            npixels=npixels,
            method=method,
        )
        assert charge_container.run_number == run_number
        assert charge_container.pixels_id.tolist() == pixels_id.tolist()
        assert charge_container.npixels == npixels
        assert charge_container.nevents == nevents
        assert charge_container.method == method


class TestChargesContainers:
    run_number = 1234
    nevents = 140
    npixels = 10

    def test_create_charge_container(self):
        chargesContainers = create_fake_chargeContainers()
        assert isinstance(chargesContainers, ChargesContainers)
        for key in chargesContainers.containers.keys():
            assert isinstance(key, EventType)

    def test_write_charge_containers(self, tmp_path="/tmp"):
        charge_containers = create_fake_chargeContainers()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        for key, container in charge_containers.containers.items():
            writer.write(table_name=f"{key.name}", containers=container)
        writer.close()

    def test_from_hdf5(self, tmp_path="/tmp"):
        charge_containers = create_fake_chargeContainers()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        for key, container in charge_containers.containers.items():
            writer.write(table_name=f"{key.name}", containers=container)
        writer.close()

        loaded_charge_container = next(ChargesContainers.from_hdf5(tmp_path))

        assert isinstance(loaded_charge_container, ChargesContainers)
        for key, container in loaded_charge_container.containers.items():
            assert np.allclose(
                container.charges_hg, charge_containers.containers[key].charges_hg
            )
            assert np.allclose(
                container.charges_lg, charge_containers.containers[key].charges_lg
            )
            assert np.allclose(
                container.peak_hg, charge_containers.containers[key].peak_hg
            )
            assert np.allclose(
                container.peak_lg, charge_containers.containers[key].peak_lg
            )
            assert container.run_number == charge_containers.containers[key].run_number
            assert (
                container.pixels_id.tolist()
                == charge_containers.containers[key].pixels_id.tolist()
            )
            assert container.nevents == charge_containers.containers[key].nevents
            assert container.npixels == charge_containers.containers[key].npixels
            assert container.method == charge_containers.containers[key].method


if __name__ == "__main__":
    pass
