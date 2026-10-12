import numpy as np

from nectarchain.makers.component.spe.parameters import Parameter, Parameters


def _two_peak_data():
    np.random.seed(42)
    charge = np.linspace(200, 450, 500)
    pedestal = np.exp(-0.5 * ((charge - 250.0) / 5.0) ** 2) * 200
    spe = np.exp(-0.5 * ((charge - 310.0) / 8.0) ** 2) * 80
    counts = (pedestal + spe).astype(int)
    return charge, counts


class TestSPEalgorithm:
    def test_constructor(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm

        algo = SPEalgorithm(pixels_id=np.array([1, 2, 3], dtype=np.uint16))
        assert algo.npixels == 3
        assert list(algo.pixels_id) == [1, 2, 3]

    def test_parameters_property(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm

        assert SPEalgorithm(pixels_id=np.array([1])).parameters is not None

    def test_results_property(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm

        r = SPEalgorithm(pixels_id=np.array([1])).results
        assert r.is_valid.shape == (1,)

    def test_get_mean_gaussian_fit(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm

        SPEalgorithm.window_length.default_value = 40
        SPEalgorithm.order.default_value = 2
        charge, counts = _two_peak_data()
        coeff, coeff_mean = SPEalgorithm._get_mean_gaussian_fit(charge, counts)
        assert len(coeff) == 3
        assert len(coeff_mean) == 3
        assert 245 < coeff[1] < 255

    def test_get_mean_gaussian_fit_with_pixel_id(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm

        SPEalgorithm.window_length.default_value = 40
        SPEalgorithm.order.default_value = 2
        charge, counts = _two_peak_data()
        coeff, coeff_mean = SPEalgorithm._get_mean_gaussian_fit(
            charge, counts, pixel_id=5
        )
        assert len(coeff) == 3

    def test_NG_Likelihood_Chi2(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEnominalalgorithm

        charge = np.array([250.0, 300.0])
        counts = np.array([100, 50])
        result = SPEnominalalgorithm._NG_Likelihood_Chi2(
            0.5,
            0.2,
            58.0,
            1.0,
            250.0,
            5.0,
            10.0,
            charge,
            counts,
            ntotalPE=10,
        )
        assert result is not None

    def test_update_parameters(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm

        SPEalgorithm.window_length.default_value = 40
        SPEalgorithm.order.default_value = 2
        params = Parameters()
        params.append(Parameter(name="pedestal", value=0))
        params.append(Parameter(name="pedestalWidth", value=1.0))
        params.append(Parameter(name="mean", value=0))
        charge, counts = _two_peak_data()

        updated = SPEalgorithm._update_parameters(params, charge, counts)
        assert 245 < updated["pedestal"].value < 255


class TestSPEnominalalgorithm:
    def test_constructor(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEnominalalgorithm

        npix = 3
        charge = np.ma.MaskedArray(
            np.tile(np.linspace(240, 400, 200), (npix, 1)),
            mask=np.zeros((npix, 200), dtype=bool),
        )
        counts = np.ma.MaskedArray(
            np.tile(np.ones(200) * 10, (npix, 1)),
            mask=np.zeros((npix, 200), dtype=bool),
        )
        algo = SPEnominalalgorithm(
            pixels_id=np.array([1, 2, 3], dtype=np.uint16),
            charge=charge,
            counts=counts,
        )
        assert algo.npixels == 3

    def test_charge_and_counts_shapes(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEnominalalgorithm

        data = np.tile(np.linspace(240, 400, 100), (2, 1))
        charge = np.ma.MaskedArray(data, mask=np.zeros((2, 100), dtype=bool))
        counts = np.ma.MaskedArray(
            np.ones((2, 100)) * 10, mask=np.zeros((2, 100), dtype=bool)
        )
        algo = SPEnominalalgorithm(
            pixels_id=np.array([1, 2]),
            charge=charge,
            counts=counts,
        )
        assert algo._charge.shape == (2, 100)

    def test_fill_results_table_from_dict(self):
        from nectarchain.makers.component.spe.spe_algorithm import (
            ContextFit,
            SPEnominalalgorithm,
        )

        charge = np.ma.MaskedArray(
            np.array([[250.0, 260.0, 270.0, 280.0, 290.0, 300.0]]),
            mask=np.zeros((1, 6), dtype=bool),
        )
        counts = np.ma.MaskedArray(
            np.array([[100.0, 80.0, 60.0, 40.0, 20.0, 10.0]]),
            mask=np.zeros((1, 6), dtype=bool),
        )
        algo = SPEnominalalgorithm(
            pixels_id=np.array([1]),
            charge=charge,
            counts=counts,
        )
        dico = {
            0: {
                "values_0": np.array([250.0, 0.5, 10.0, 0.2, 55.0, 1.0, 5.0, 0.0]),
                "errors_0": np.array([2.0, 0.05, 1.0, 0.02, 1.0, 0.1, 0.5, 0.0]),
                "fit_status_0": {
                    "is_valid": True,
                    "has_valid_parameters": True,
                    "has_parameters_at_limit": False,
                    "has_reached_call_limit": False,
                    "nfit": 7,
                    "values": 10.5,
                },
            }
        }
        minuit_arr = np.empty(1, dtype=object)
        minuit_arr[0] = {"names": ["p"], "values": {"p": 0.0}}
        with ContextFit(SPEnominalalgorithm, minuit_arr, charge, counts):
            result = algo._fill_results_table_from_dict(
                dico, np.array([1]), return_fit_array=True
            )
        assert result is not None
        assert algo._results.is_valid[0]

    def test_plot_single_matplotlib(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEnominalalgorithm

        fig, ax = SPEnominalalgorithm.plot_single_matplotlib(
            pixel_id=1,
            charge=np.array([250.0, 260.0, 270.0, 280.0, 290.0, 300.0]),
            counts=np.array([100, 80, 60, 40, 20, 10]),
            pp=0.5,
            resolution=0.2,
            gain=58.0,
            gain_error=1.0,
            n=1.0,
            pedestal=250.0,
            pedestalWidth=5.0,
            luminosity=10.0,
            likelihood=0.1,
        )
        assert fig is not None and ax is not None


class TestSPEHHValgorithm:
    def test_constructor(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEHHValgorithm

        data = np.tile(np.linspace(240, 400, 100), (2, 1))
        charge = np.ma.MaskedArray(data, mask=np.zeros((2, 100), dtype=bool))
        counts = np.ma.MaskedArray(
            np.ones((2, 100)) * 10, mask=np.zeros((2, 100), dtype=bool)
        )
        algo = SPEHHValgorithm(
            pixels_id=np.array([1, 2]),
            charge=charge,
            counts=counts,
        )
        assert algo.tol == 1e5


class TestSPEnominalStdalgorithm:
    def test_constructor_fixes_params(self):
        from nectarchain.makers.component.spe.spe_algorithm import (
            SPEnominalStdalgorithm,
        )

        data = np.linspace(240, 400, 100).reshape(1, -1)
        charge = np.ma.MaskedArray(data, mask=np.zeros((1, 100), dtype=bool))
        counts = np.ma.MaskedArray(
            np.ones((1, 100)) * 10, mask=np.zeros((1, 100), dtype=bool)
        )
        algo = SPEnominalStdalgorithm(
            pixels_id=np.array([1]),
            charge=charge,
            counts=counts,
        )
        assert algo._parameters["pp"].frozen
        assert algo._parameters["n"].frozen


class TestSPEHHVStdalgorithm:
    def test_constructor(self):
        from nectarchain.makers.component.spe.spe_algorithm import SPEHHVStdalgorithm

        data = np.linspace(240, 400, 100).reshape(1, -1)
        charge = np.ma.MaskedArray(data, mask=np.zeros((1, 100), dtype=bool))
        counts = np.ma.MaskedArray(
            np.ones((1, 100)) * 10, mask=np.zeros((1, 100), dtype=bool)
        )
        algo = SPEHHVStdalgorithm(
            pixels_id=np.array([1]),
            charge=charge,
            counts=counts,
        )
        assert algo is not None
