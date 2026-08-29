import numpy as np
from ctapipe.utils import get_dataset_path

from nectarchain.data.container import ChargesContainer, SPEfitContainer
from nectarchain.makers.component.photostatistic_algorithm import (
    PhotoStatisticAlgorithm,
)
from nectarchain.makers.core import BaseNectarCAMCalibrationTool

# Real pixel IDs from a NectarCAM run
_RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
_eventsource = BaseNectarCAMCalibrationTool.load_run(
    3938, max_events=1, run_file=_RUN_FILE
)
_tel_id = list(_eventsource.subarray.tel_ids)[0]
_geom = _eventsource.subarray.tel[_tel_id].camera.geometry
_REAL_PIXEL_IDS = _geom.pix_id.astype(np.uint16)
_N_PIXELS_REAL = len(_REAL_PIXEL_IDS)


class TestPhotoStatisticAlgorithm:
    def _make(self):
        return PhotoStatisticAlgorithm(
            pixels_id=np.array([1, 2, 3], dtype=np.uint16),
            FFcharge_hg=np.array([[100.0, 110.0, 120.0]], dtype=np.float32),
            FFcharge_lg=np.array([[10.0, 11.0, 12.0]], dtype=np.float32),
            Pedcharge_hg=np.array([[50.0, 55.0, 60.0]], dtype=np.float32),
            Pedcharge_lg=np.array([[5.0, 5.5, 6.0]], dtype=np.float32),
            coefCharge_FF_Ped=0.5,
            SPE_resolution=np.array([[0.2, 0.1, 0.1]], dtype=np.float32),
            SPE_high_gain=np.array([[58.0, 60.0, 62.0]], dtype=np.float32),
        )

    def test_constructor(self):
        algo = self._make()
        assert algo.npixels == 3

    def test_meanPedHG(self):
        expected = np.mean([[50.0, 55.0, 60.0]], axis=0) * 0.5
        np.testing.assert_allclose(self._make().meanPedHG, expected)

    def test_properties_return_arrays(self):
        algo = self._make()
        assert algo.sigmaPedHG.shape == (3,)
        assert algo.meanChargeHG.shape == (3,)
        assert algo.sigmaChargeHG.shape == (3,)
        assert algo.gainHG.shape == (3,)
        assert algo.gainLG.shape == (3,)

    def test_run_returns_zero(self):
        assert self._make().run() == 0

    def test_run_sets_results(self):
        algo = self._make()
        algo.run()
        r = algo.results
        assert r.is_valid.shape == (3,)
        assert r.high_gain.shape == (3, 3)

    def test_run_with_mask(self):
        algo = self._make()
        algo.run(pixels_id=np.array([1, 2]))
        r = algo.results
        assert r.is_valid[0] and r.is_valid[1] and not r.is_valid[2]

    def test_plot_correlation(self):
        fig = PhotoStatisticAlgorithm.plot_correlation(
            np.array([30.0, 40.0, 50.0, 60.0, 70.0]),
            np.array([32.0, 42.0, 48.0, 62.0, 68.0]),
        )
        assert fig is not None

    def test_spe_resolution_deepcopy(self):
        import copy

        algo = self._make()
        r1 = algo.results
        assert copy.deepcopy(r1) is not None

    def test_npixels_property(self):
        assert self._make().npixels == 3


class TestPhotoStatisticAlgorithmFactory:
    def test_get_charges_FF_Ped_reshaped(self):
        npix, nev = 10, 5
        pid = _REAL_PIXEL_IDS[:npix]
        FF = ChargesContainer(
            charges_hg=np.ones((nev, npix), dtype=np.float32) * 100.0,
            charges_lg=np.ones((nev, npix), dtype=np.float32) * 10.0,
            peak_hg=np.zeros((nev, npix), dtype=np.float32),
            peak_lg=np.zeros((nev, npix), dtype=np.float32),
            pixels_id=pid,
            run_number=np.uint16(1),
            nevents=np.uint64(nev),
            npixels=np.uint16(npix),
            method="test",
        )
        FF.validate()
        Ped = ChargesContainer(
            charges_hg=np.ones((nev, npix), dtype=np.float32) * 50.0,
            charges_lg=np.ones((nev, npix), dtype=np.float32) * 5.0,
            peak_hg=np.zeros((nev, npix), dtype=np.float32),
            peak_lg=np.zeros((nev, npix), dtype=np.float32),
            pixels_id=pid,
            run_number=np.uint16(2),
            nevents=np.uint64(nev),
            npixels=np.uint16(npix),
            method="test",
        )
        Ped.validate()
        SPE = SPEfitContainer(
            is_valid=np.array([True] * npix, dtype=bool),
            high_gain=np.tile(np.array([58.0, 1.0, 1.0]), (npix, 1)),
            low_gain=np.zeros((npix, 3)),
            pixels_id=pid,
            likelihood=np.ones(npix) * 0.1,
            p_value=np.ones(npix) * 0.5,
            pedestal=np.tile(np.array([250.0, 1.0, 1.0]), (npix, 1)),
            pedestalWidth=np.tile(np.array([5.0, 1.0, 1.0]), (npix, 1)),
            resolution=np.tile(np.array([0.2, 0.01, 0.01]), (npix, 1)),
            luminosity=np.tile(np.array([10.0, 1.0, 1.0]), (npix, 1)),
            mean=np.tile(np.array([58.0, 1.0, 1.0]), (npix, 1)),
            n=np.tile(np.array([1.0, 0.1, 0.1]), (npix, 1)),
            pp=np.tile(np.array([0.5, 0.05, 0.05]), (npix, 1)),
        )
        SPE.validate()
        r = PhotoStatisticAlgorithm._PhotoStatisticAlgorithm__get_charges_FF_Ped_reshaped(  # noqa: E501
            FF, Ped, SPE
        )
        assert "pixels_id" in r
        assert "FFcharge_hg" in r
        assert "SPE_resolution" in r
        assert "SPE_high_gain" in r
