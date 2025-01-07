import glob

import numpy as np
import pytest
from ctapipe.containers import EventType
from ctapipe.io import HDF5TableWriter

from nectarchain.data.container import WaveformsContainer, WaveformsContainers


def create_fake_waveformContainer():
    nevents = TestWaveformsContainer.nevents
    npixels = TestWaveformsContainer.npixels
    nsamples = TestWaveformsContainer.nsamples

    rng = np.random.default_rng()
    waveform = WaveformsContainer(
        pixels_id=np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16),
        nevents=np.uint64(nevents),
        npixels=np.uint16(npixels),
        nsamples=np.uint8(nsamples),
        wfs_hg=rng.integers(
            low=0, high=1000, size=(nevents, npixels, nsamples), dtype=np.uint16
        ),
        wfs_lg=rng.integers(
            low=0, high=1000, size=(nevents, npixels, nsamples), dtype=np.uint16
        ),
        run_number=np.uint16(TestWaveformsContainer.run_number),
        camera="TEST",
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
    waveform.validate()
    return waveform


def create_fake_waveformContainers():
    waveform_1 = create_fake_waveformContainer()
    waveform_2 = create_fake_waveformContainer()
    waveform = WaveformsContainers()
    waveform.containers[EventType.FLATFIELD] = waveform_1
    waveform.containers[EventType.SKY_PEDESTAL] = waveform_2
    return waveform


class TestWaveformsContainer:
    run_number = 1234
    nevents = 140
    npixels = 10
    nsamples = 6

    # Tests that a ChargeContainer object can be created with valid input parameters.
    def test_create_waveform_container(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16)
        nevents = np.uint64(TestWaveformsContainer.nevents)
        npixels = np.uint16(TestWaveformsContainer.npixels)
        nsamples = np.uint8(TestWaveformsContainer.nsamples)

        run_number = np.uint16(TestWaveformsContainer.run_number)
        wfs_hg = np.uint16(np.random.randn(nevents, npixels, nsamples))
        wfs_lg = np.uint16(np.random.randn(nevents, npixels, nsamples))
        waveform_container = WaveformsContainer(
            wfs_hg=wfs_hg,
            wfs_lg=wfs_lg,
            run_number=run_number,
            pixels_id=pixels_id,
            nevents=nevents,
            npixels=npixels,
            nsamples=nsamples,
        )
        waveform_container.validate()
        assert waveform_container.run_number == run_number
        assert waveform_container.pixels_id.tolist() == pixels_id.tolist()
        assert waveform_container.nevents == nevents
        assert waveform_container.npixels == npixels
        assert waveform_container.nsamples == nsamples

    def test_write_waveform_container(self, tmp_path="/tmp"):
        waveform_container = create_fake_waveformContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="WaveformsContainer", containers=waveform_container)
        writer.close()

    def test_from_hdf5(self, tmp_path="/tmp"):
        waveform_container = create_fake_waveformContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="WaveformsContainer_0", containers=waveform_container)
        writer.close()

        loaded_waveform_container = next(WaveformsContainer.from_hdf5(tmp_path))

        assert isinstance(loaded_waveform_container, WaveformsContainer)
        assert np.allclose(loaded_waveform_container.wfs_hg, waveform_container.wfs_hg)
        assert np.allclose(loaded_waveform_container.wfs_lg, waveform_container.wfs_lg)
        assert loaded_waveform_container.run_number == waveform_container.run_number
        assert (
            loaded_waveform_container.pixels_id.tolist()
            == waveform_container.pixels_id.tolist()
        )
        assert loaded_waveform_container.nevents == waveform_container.nevents
        assert loaded_waveform_container.npixels == waveform_container.npixels
        assert loaded_waveform_container.nsamples == waveform_container.nsamples

    def test_access_properties(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10])
        nevents = 40
        npixels = 10
        nsamples = 6
        wfs_hg = np.random.randn(nevents, npixels, nsamples)
        wfs_lg = np.random.randn(nevents, npixels, nsamples)
        run_number = 1234
        waveform_container = WaveformsContainer(
            wfs_hg=wfs_hg,
            wfs_lg=wfs_lg,
            run_number=run_number,
            pixels_id=pixels_id,
            nevents=nevents,
            npixels=npixels,
            nsamples=nsamples,
        )
        assert waveform_container.run_number == run_number
        assert waveform_container.pixels_id.tolist() == pixels_id.tolist()
        assert waveform_container.npixels == npixels
        assert waveform_container.nevents == nevents
        assert waveform_container.nsamples == nsamples


class TestWaveformsContainers:
    run_number = 1234
    nevents = 140
    npixels = 10
    nsamples = 6

    def test_create_waveform_container(self):
        waveformsContainers = create_fake_waveformContainers()
        assert isinstance(waveformsContainers, WaveformsContainers)
        for key in waveformsContainers.containers.keys():
            assert isinstance(key, EventType)

    def test_write_waveform_containers(self, tmp_path="/tmp"):
        waveform_containers = create_fake_waveformContainers()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        for key, container in waveform_containers.containers.items():
            writer.write(table_name=f"{key.name}", containers=container)
        writer.close()

    def test_from_hdf5(self, tmp_path="/tmp"):
        waveform_containers = create_fake_waveformContainers()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        for key, container in waveform_containers.containers.items():
            writer.write(table_name=f"{key.name}", containers=container)
        writer.close()

        loaded_waveform_container = next(WaveformsContainers.from_hdf5(tmp_path))

        assert isinstance(loaded_waveform_container, WaveformsContainers)
        for key, container in loaded_waveform_container.containers.items():
            assert np.allclose(
                container.wfs_hg, waveform_containers.containers[key].wfs_hg
            )
            assert np.allclose(
                container.wfs_lg, waveform_containers.containers[key].wfs_lg
            )
            assert (
                container.run_number == waveform_containers.containers[key].run_number
            )
            assert (
                container.pixels_id.tolist()
                == waveform_containers.containers[key].pixels_id.tolist()
            )
            assert container.nevents == waveform_containers.containers[key].nevents
            assert container.npixels == waveform_containers.containers[key].npixels
            assert container.nsamples == waveform_containers.containers[key].nsamples


if __name__ == "__main__":
    pass
