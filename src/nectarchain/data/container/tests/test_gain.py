import glob

import numpy as np
import pytest
from ctapipe.containers import EventType
from ctapipe.io import HDF5TableWriter

from nectarchain.data.container import GainContainer, SPEfitContainer


def create_fake_GainContainer():
    npixels = TestGainContainer.npixels
    rng = np.random.default_rng()
    gain = GainContainer(
        pixels_id=np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16),
        is_valid=rng.integers(low=0, high=1.1, size=(npixels,), dtype=bool),
        high_gain=rng.random(size=(npixels, 3), dtype=np.float64),
        low_gain=rng.random(size=(npixels, 3), dtype=np.float64),
    )
    gain.validate()
    return gain


def create_fake_SPEfitContainer():
    npixels = TestSPEfitContainer.npixels
    rng = np.random.default_rng()
    gain = SPEfitContainer(
        pixels_id=np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16),
        is_valid=rng.integers(low=0, high=1.1, size=(npixels,), dtype=bool),
        high_gain=rng.random(size=(npixels, 3), dtype=np.float64),
        low_gain=rng.random(size=(npixels, 3), dtype=np.float64),
        likelihood=rng.random(size=(npixels,), dtype=np.float64),
        p_value=rng.random(size=(npixels,), dtype=np.float64),
        pedestal=rng.random(size=(npixels, 3), dtype=np.float64),
        pedestalWidth=rng.random(size=(npixels, 3), dtype=np.float64),
        resolution=rng.random(size=(npixels, 3), dtype=np.float64),
        luminosity=rng.random(size=(npixels, 3), dtype=np.float64),
        mean=rng.random(size=(npixels, 3), dtype=np.float64),
        n=rng.random(size=(npixels, 3), dtype=np.float64),
        pp=rng.random(size=(npixels, 3), dtype=np.float64),
    )
    gain.validate()
    return gain


class TestGainContainer:
    npixels = 10

    def test_create_gain_container(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16)
        npixels = np.uint16(TestGainContainer.npixels)
        high_gain = np.float64(np.random.randn(npixels, 3))
        low_gain = np.float64(np.random.randn(npixels, 3))
        is_valid = np.array(np.random.randn(npixels), dtype=bool)
        gain_container = GainContainer(
            pixels_id=pixels_id,
            is_valid=is_valid,
            high_gain=high_gain,
            low_gain=low_gain,
        )
        gain_container.validate()
        assert gain_container.pixels_id.tolist() == pixels_id.tolist()
        assert gain_container.is_valid.tolist() == is_valid.tolist()
        assert gain_container.high_gain.tolist() == high_gain.tolist()
        assert gain_container.low_gain.tolist() == low_gain.tolist()

    def test_write_gain_container(self, tmp_path="/tmp"):
        gain_container = create_fake_GainContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="GainContainer", containers=gain_container)
        writer.close()

    def test_from_hdf5(self, tmp_path="/tmp"):
        gain_container = create_fake_GainContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="GainContainer", containers=gain_container)
        writer.close()

        loaded_gain_container = next(GainContainer.from_hdf5(tmp_path))

        assert isinstance(loaded_gain_container, GainContainer)
        assert (
            gain_container.pixels_id.tolist()
            == loaded_gain_container.pixels_id.tolist()
        )
        assert (
            gain_container.is_valid.tolist() == loaded_gain_container.is_valid.tolist()
        )
        assert (
            gain_container.high_gain.tolist()
            == loaded_gain_container.high_gain.tolist()
        )
        assert (
            gain_container.low_gain.tolist() == loaded_gain_container.low_gain.tolist()
        )


class TestSPEfitContainer:
    npixels = 10

    def test_create_SPEfit_container(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16)
        npixels = np.uint16(TestSPEfitContainer.npixels)
        high_gain = np.float64(np.random.randn(npixels, 3))
        low_gain = np.float64(np.random.randn(npixels, 3))
        is_valid = np.array(np.random.randn(npixels), dtype=bool)
        likelihood = np.float64(np.random.randn(npixels))
        p_value = np.float64(np.random.randn(npixels))
        pedestal = np.float64(np.random.randn(npixels, 3))
        pedestalWidth = np.float64(np.random.randn(npixels, 3))
        resolution = np.float64(np.random.randn(npixels, 3))
        luminosity = np.float64(np.random.randn(npixels, 3))
        mean = np.float64(np.random.randn(npixels, 3))
        n = np.float64(np.random.randn(npixels, 3))
        pp = np.float64(np.random.randn(npixels, 3))
        SPEfit_container = SPEfitContainer(
            pixels_id=pixels_id,
            is_valid=is_valid,
            high_gain=high_gain,
            low_gain=low_gain,
            likelihood=likelihood,
            p_value=p_value,
            pedestal=pedestal,
            pedestalWidth=pedestalWidth,
            resolution=resolution,
            luminosity=luminosity,
            mean=mean,
            n=n,
            pp=pp,
        )
        SPEfit_container.validate()
        assert SPEfit_container.pixels_id.tolist() == pixels_id.tolist()
        assert SPEfit_container.is_valid.tolist() == is_valid.tolist()
        assert SPEfit_container.high_gain.tolist() == high_gain.tolist()
        assert SPEfit_container.low_gain.tolist() == low_gain.tolist()
        assert SPEfit_container.likelihood.tolist() == likelihood.tolist()
        assert SPEfit_container.p_value.tolist() == p_value.tolist()
        assert SPEfit_container.pedestal.tolist() == pedestal.tolist()
        assert SPEfit_container.pedestalWidth.tolist() == pedestalWidth.tolist()
        assert SPEfit_container.resolution.tolist() == resolution.tolist()
        assert SPEfit_container.luminosity.tolist() == luminosity.tolist()
        assert SPEfit_container.mean.tolist() == mean.tolist()
        assert SPEfit_container.n.tolist() == n.tolist()
        assert SPEfit_container.pp.tolist() == pp.tolist()

    def test_write_SPEfit_container(self, tmp_path="/tmp"):
        SPEfit_container = create_fake_SPEfitContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="SPEfitContainer", containers=SPEfit_container)
        writer.close()

    def test_from_hdf5(self, tmp_path="/tmp"):
        SPEfit_container = create_fake_SPEfitContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="SPEfitContainer", containers=SPEfit_container)
        writer.close()

        loaded_SPEfit_container = next(SPEfitContainer.from_hdf5(tmp_path))

        assert isinstance(loaded_SPEfit_container, SPEfitContainer)
        assert (
            SPEfit_container.pixels_id.tolist()
            == loaded_SPEfit_container.pixels_id.tolist()
        )
        assert (
            SPEfit_container.is_valid.tolist()
            == loaded_SPEfit_container.is_valid.tolist()
        )
        assert (
            SPEfit_container.high_gain.tolist()
            == loaded_SPEfit_container.high_gain.tolist()
        )
        assert (
            SPEfit_container.low_gain.tolist()
            == loaded_SPEfit_container.low_gain.tolist()
        )
        assert (
            SPEfit_container.likelihood.tolist()
            == loaded_SPEfit_container.likelihood.tolist()
        )
        assert (
            SPEfit_container.p_value.tolist()
            == loaded_SPEfit_container.p_value.tolist()
        )
        assert (
            SPEfit_container.pedestal.tolist()
            == loaded_SPEfit_container.pedestal.tolist()
        )
        assert (
            SPEfit_container.pedestalWidth.tolist()
            == loaded_SPEfit_container.pedestalWidth.tolist()
        )
        assert (
            SPEfit_container.resolution.tolist()
            == loaded_SPEfit_container.resolution.tolist()
        )
        assert (
            SPEfit_container.luminosity.tolist()
            == loaded_SPEfit_container.luminosity.tolist()
        )
        assert SPEfit_container.mean.tolist() == loaded_SPEfit_container.mean.tolist()
        assert SPEfit_container.n.tolist() == loaded_SPEfit_container.n.tolist()
        assert SPEfit_container.pp.tolist() == loaded_SPEfit_container.pp.tolist()


if __name__ == "__main__":
    pass
