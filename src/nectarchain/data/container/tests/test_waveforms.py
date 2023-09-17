import glob

import numpy as np
from ctapipe.instrument import SubarrayDescription

from nectarchain.data.container import WaveformsContainer, WaveformsContainerIO
from nectarchain.makers import WaveformsMaker


def create_fake_waveformsContainer():
    nevents = TestWaveformsContainer.nevents
    npixels = TestWaveformsContainer.npixels
    nsamples = TestWaveformsContainer.nsamples
    rng = np.random.default_rng()
    faked_subarray = SubarrayDescription(name="TEST")

    return WaveformsContainer(
        pixels_id=np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10]),
        nevents=nevents,
        npixels=npixels,
        wfs_hg=rng.integers(low=0, high=1000, size=(nevents, npixels, nsamples)),
        wfs_lg=rng.integers(low=0, high=1000, size=(nevents, npixels, nsamples)),
        run_number=TestWaveformsContainer.run_number,
        camera="TEST",
        subarray=faked_subarray,
        broken_pixels_hg=rng.integers(low=0, high=1, size=(nevents, npixels)),
        broken_pixels_lg=rng.integers(low=0, high=1, size=(nevents, npixels)),
        ucts_timestamp=rng.integers(low=0, high=100, size=(nevents)),
        ucts_busy_counter=rng.integers(low=0, high=100, size=(nevents)),
        ucts_event_counter=rng.integers(low=0, high=100, size=(nevents)),
        event_type=rng.integers(low=0, high=1, size=(nevents)),
        event_id=rng.integers(low=0, high=1000, size=(nevents)),
        trig_pattern_all=rng.integers(low=0, high=1, size=(nevents, npixels, 4)),
        trig_pattern=rng.integers(low=0, high=1, size=(nevents, npixels)),
        multiplicity=rng.integers(low=0, high=1, size=(nevents)),
    )


class TestWaveformsContainer:
    run_number = 1234
    nevents = 140
    npixels = 10
    nsamples = 5

    # Tests that a ChargeContainer object can be created with valid input parameters.
    def test_create_waveform_container(self):
        waveform_container = create_fake_waveformsContainer()
        assert isinstance(waveform_container, WaveformsContainer)

    # Tests that the ChargeContainer object can be written to a file and the file is created.
    def test_write_waveform_container(self, tmp_path="/tmp"):
        waveform_container = create_fake_waveformsContainer()
        tmp_path += f"/{np.random.randn(1)[0]}"

        WaveformsContainerIO.write(tmp_path, waveform_container)

        assert (
            len(glob.glob(f"{tmp_path}/*_run{TestWaveformsContainer.run_number}.fits"))
            == 1
        )

    # Tests that a ChargeContainer object can be loaded from a file and the object is correctly initialized.
    def test_load_waveform_container(self, tmp_path="/tmp"):
        waveform_container = create_fake_waveformsContainer()
        tmp_path += f"/{np.random.randn(1)[0]}"

        WaveformsContainerIO.write(tmp_path, waveform_container)

        loaded_waveform_container = WaveformsContainerIO.load(
            tmp_path, TestWaveformsContainer.run_number
        )

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

    # Tests that the ChargeContainer object can be sorted by event_id and the object is sorted accordingly.
    def test_sort_waveform_container(self):
        waveform_container = create_fake_waveformsContainer()

        sorted_waveform_container = WaveformsMaker.sort(waveform_container)

        assert sorted_waveform_container.event_id.tolist() == sorted(
            waveform_container.event_id.tolist()
        )


if __name__ == "__main__":
    TestWaveformsContainer().test_create_waveform_container()
