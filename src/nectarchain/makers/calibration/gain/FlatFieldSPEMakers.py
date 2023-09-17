import logging
import sys

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy
import os
import time
from inspect import signature
from multiprocessing import Pool
from typing import Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.table import Column, QTable
from iminuit import Minuit
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.special import gammainc

from ....data.container import ChargesContainer
from ...chargesMakers import ChargesMaker
from .gainMakers import GainMaker
from .parameters import Parameter, Parameters
from .utils import (
    MPE2,
    MeanValueError,
    PedestalValueError,
    Statistics,
    UtilsMinuit,
    weight_gaussian,
)

__all__ = ["FlatFieldSingleHHVSPEMaker", "FlatFieldSingleHHVStdSPEMaker"]


class FlatFieldSPEMaker(GainMaker):

    """
    The `FlatFieldSPEMaker` class is used for flat field single photoelectron (SPE) calibration calculations on data. It inherits from the `GainMaker` class and adds functionality specific to flat field SPE calibration.

    Example Usage:
        # Create an instance of the FlatFieldSPEMaker class
        flat_field_maker = FlatFieldSPEMaker()

        # Read parameters from a YAML file
        flat_field_maker.read_param_from_yaml("parameters.yaml")

        # Update parameters based on data
        flat_field_maker._update_parameters(parameters, charge, counts)

        # Update the result table based on the parameters
        flat_field_maker._update_table_from_parameters()

    Main functionalities:
    - Inherits from the `GainMaker` class and adds functionality specific to flat field SPE calibration.
    - Reads parameters from a YAML file and updates the internal parameters of the class.
    - Updates the parameters based on data, such as charge and counts.
    - Updates a result table based on the parameters.

    Methods:
    - `read_param_from_yaml(parameters_file, only_update)`: Reads parameters from a YAML file and updates the internal parameters of the class. If `only_update` is True, only the parameters that exist in the YAML file will be updated.
    - `_update_parameters(parameters, charge, counts, **kwargs)`: Updates the parameters based on data, such as charge and counts. It performs a Gaussian fit on the data to determine the pedestal and mean values, and updates the corresponding parameters accordingly.
    - `_get_mean_gaussian_fit(charge, counts, extension, **kwargs)`: Performs a Gaussian fit on the data to determine the pedestal and mean values. It returns the fit coefficients.
    - `_update_table_from_parameters()`: Updates a result table based on the parameters. It adds columns to the table for each parameter and its corresponding error.

    Attributes:
    - `_Windows_lenght`: A class attribute that represents the length of the windows used for smoothing the data.
    - `_Order`: A class attribute that represents the order of the polynomial used for smoothing the data.

    Members:
    - `npixels`: A property that returns the number of pixels.
    - `parameters`: A property that returns a deep copy of the internal parameters of the class.
    - `_parameters`: A property that returns the internal parameters of the class.
    """

    _Windows_lenght = 40
    _Order = 2

    # constructors
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the FlatFieldSPEMaker class.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

        Returns:
        None
        """
        super().__init__(*args, **kwargs)
        self.__parameters = Parameters()

    @property
    def npixels(self):
        """
        Returns the number of pixels.

        Returns:
            int: The number of pixels.
        """
        return len(self._pixels_id)

    @property
    def parameters(self):
        """
        Returns a deep copy of the internal parameters.

        Returns:
            dict: A deep copy of the internal parameters.
        """
        return copy.deepcopy(self.__parameters)

    @property
    def _parameters(self):
        """
        Returns the internal parameters.

        Returns:
            dict: The internal parameters.
        """
        return self.__parameters

    # methods
    def read_param_from_yaml(self, parameters_file, only_update=False) -> None:
        """
        Reads parameters from a YAML file and updates the internal parameters of the FlatFieldSPEMaker class.

        Args:
            parameters_file (str): The name of the YAML file containing the parameters.
            only_update (bool, optional): If True, only the parameters that exist in the YAML file will be updated. Default is False.

        Returns:
            None
        """
        with open(
            f"{os.path.dirname(os.path.abspath(__file__))}/{parameters_file}"
        ) as parameters:
            param = yaml.safe_load(parameters)
            if only_update:
                for i, name in enumerate(self.__parameters.parnames):
                    dico = param.get(name, False)
                    if dico:
                        self._parameters.parameters[i].value = dico.get("value")
                        self._parameters.parameters[i].min = dico.get("min", np.nan)
                        self._parameters.parameters[i].max = dico.get("max", np.nan)
            else:
                for name, dico in param.items():
                    setattr(
                        self,
                        f"__{name}",
                        Parameter(
                            name=name,
                            value=dico["value"],
                            min=dico.get("min", np.nan),
                            max=dico.get("max", np.nan),
                            unit=dico.get("unit", u.dimensionless_unscaled),
                        ),
                    )
                    self._parameters.append(eval(f"self.__{name}"))

    @staticmethod
    def _update_parameters(
        parameters: Parameters, charge: np.ndarray, counts: np.ndarray, **kwargs
    ) -> Parameters:
        """
        Update the parameters of the FlatFieldSPEMaker class based on the input charge and counts data.

        Args:
            parameters (Parameters): An instance of the Parameters class that holds the internal parameters of the FlatFieldSPEMaker class.
            charge (np.ndarray): An array of charge values.
            counts (np.ndarray): An array of corresponding counts values.
            **kwargs: Additional keyword arguments.

        Returns:
            Parameters: The updated parameters object with the pedestal and mean values and their corresponding limits.
        """
        try:
            coeff_ped, coeff_mean = __class__._get_mean_gaussian_fit(
                charge, counts, **kwargs
            )
            pedestal = parameters["pedestal"]
            pedestal.value = coeff_ped[1]
            pedestal.min = coeff_ped[1] - coeff_ped[2]
            pedestal.max = coeff_ped[1] + coeff_ped[2]
            log.debug(f"pedestal updated: {pedestal}")
            pedestalWidth = parameters["pedestalWidth"]
            pedestalWidth.value = pedestal.max - pedestal.value
            pedestalWidth.max = 3 * pedestalWidth.value
            log.debug(f"pedestalWidth updated: {pedestalWidth.value}")

            if (coeff_mean[1] - pedestal.value < 0) or (
                (coeff_mean[1] - coeff_mean[2]) - pedestal.max < 0
            ):
                raise MeanValueError("mean gaussian fit not good")
            mean = parameters["mean"]
            mean.value = coeff_mean[1] - pedestal.value
            mean.min = (coeff_mean[1] - coeff_mean[2]) - pedestal.max
            mean.max = (coeff_mean[1] + coeff_mean[2]) - pedestal.min
            log.debug(f"mean updated: {mean}")
        except MeanValueError as e:
            log.warning(e, exc_info=True)
            log.warning("mean parameters limits and starting value not changed")
        except Exception as e:
            log.warning(e, exc_info=True)
            log.warning(
                "pedestal and mean parameters limits and starting value not changed"
            )
        return parameters

    @staticmethod
    def _get_mean_gaussian_fit(
        charge: np.ndarray, counts: np.ndarray, extension: str = "", **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a Gaussian fit on the data to determine the pedestal and mean values.

        Args:
            charge (np.ndarray): An array of charge values.
            counts (np.ndarray): An array of corresponding counts.
            extension (str, optional): An extension string. Defaults to "".
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple of fit coefficients for the pedestal and mean.

        Example Usage:
            flat_field_maker = FlatFieldSPEMaker()
            charge = np.array([1, 2, 3, 4, 5])
            counts = np.array([10, 20, 30, 40, 50])
            coeff, coeff_mean = flat_field_maker._get_mean_gaussian_fit(charge, counts)
            print(coeff)  # Output: [norm,peak_value, peak_width]
            print(coeff_mean)  # Output: [norm,peak_value_mean, peak_width_mean]
        """
        windows_lenght = __class__._Windows_lenght
        order = __class__._Order
        histo_smoothed = savgol_filter(counts, windows_lenght, order)
        peaks = find_peaks(histo_smoothed, 10)
        peak_max = np.argmax(histo_smoothed[peaks[0]])
        peak_pos, peak_value = charge[peaks[0][peak_max]], counts[peaks[0][peak_max]]
        coeff, _ = curve_fit(
            weight_gaussian,
            charge[: peaks[0][peak_max]],
            histo_smoothed[: peaks[0][peak_max]],
            p0=[peak_value, peak_pos, 1],
        )
        if log.getEffectiveLevel() == logging.DEBUG and kwargs.get("display", False):
            log.debug("plotting figures with prefit parameters computation")
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.errorbar(
                charge, counts, np.sqrt(counts), zorder=0, fmt=".", label="data"
            )
            ax.plot(
                charge,
                histo_smoothed,
                label=f"smoothed data with savgol filter (windows lenght : {windows_lenght}, order : {order})",
            )
            ax.plot(
                charge,
                weight_gaussian(charge, coeff[0], coeff[1], coeff[2]),
                label="gaussian fit of the pedestal, left tail only",
            )
            ax.set_xlim([peak_pos - 500, None])
            ax.vlines(
                coeff[1],
                0,
                peak_value,
                label=f"pedestal initial value = {coeff[1]:.0f}",
                color="red",
            )
            ax.add_patch(
                Rectangle(
                    (coeff[1] - coeff[2], 0),
                    2 * coeff[2],
                    peak_value,
                    fc=to_rgba("red", 0.5),
                )
            )
            ax.set_xlabel("Charge (ADC)", size=15)
            ax.set_ylabel("Events", size=15)
            ax.legend(fontsize=7)
            os.makedirs(
                f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/",
                exist_ok=True,
            )
            fig.savefig(
                f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/initialization_pedestal_pixel{extension}_{os.getpid()}.pdf"
            )
            fig.clf()
            plt.close(fig)
            del fig, ax
        mask = charge > coeff[1] + 3 * coeff[2]
        peaks_mean = find_peaks(histo_smoothed[mask])
        peak_max_mean = np.argmax(histo_smoothed[mask][peaks_mean[0]])
        peak_pos_mean, peak_value_mean = (
            charge[mask][peaks_mean[0][peak_max_mean]],
            histo_smoothed[mask][peaks_mean[0][peak_max_mean]],
        )
        mask = (charge > ((coeff[1] + peak_pos_mean) / 2)) * (
            charge < (peak_pos_mean + (peak_pos_mean - coeff[1]) / 2)
        )
        coeff_mean, _ = curve_fit(
            weight_gaussian,
            charge[mask],
            histo_smoothed[mask],
            p0=[peak_value_mean, peak_pos_mean, 1],
        )
        if log.getEffectiveLevel() == logging.DEBUG and kwargs.get("display", False):
            log.debug("plotting figures with prefit parameters computation")
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.errorbar(
                charge, counts, np.sqrt(counts), zorder=0, fmt=".", label="data"
            )
            ax.plot(
                charge,
                histo_smoothed,
                label=f"smoothed data with savgol filter (windows lenght : {windows_lenght}, order : {order})",
            )
            ax.plot(
                charge,
                weight_gaussian(charge, coeff_mean[0], coeff_mean[1], coeff_mean[2]),
                label="gaussian fit of the SPE",
            )
            ax.vlines(
                coeff_mean[1],
                0,
                peak_value,
                label=f"mean initial value = {coeff_mean[1] - coeff[1]:.0f}",
                color="red",
            )
            ax.add_patch(
                Rectangle(
                    (coeff_mean[1] - coeff_mean[2], 0),
                    2 * coeff_mean[2],
                    peak_value_mean,
                    fc=to_rgba("red", 0.5),
                )
            )
            ax.set_xlim([peak_pos - 500, None])
            ax.set_xlabel("Charge (ADC)", size=15)
            ax.set_ylabel("Events", size=15)
            ax.legend(fontsize=7)
            os.makedirs(
                f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/",
                exist_ok=True,
            )
            fig.savefig(
                f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/initialization_mean_pixel{extension}_{os.getpid()}.pdf"
            )
            fig.clf()
            plt.close(fig)
            del fig, ax
        return coeff, coeff_mean

    def _update_table_from_parameters(self) -> None:
        """
        Update the result table based on the parameters of the FlatFieldSPEMaker class.
        This method adds columns to the table for each parameter and its corresponding error.
        """

        for param in self._parameters.parameters:
            if not (param.name in self._results.colnames):
                self._results.add_column(
                    Column(
                        data=np.empty((self.npixels), dtype=np.float64),
                        name=param.name,
                        unit=param.unit,
                    )
                )
                self._results.add_column(
                    Column(
                        data=np.empty((self.npixels), dtype=np.float64),
                        name=f"{param.name}_error",
                        unit=param.unit,
                    )
                )


class FlatFieldSingleHHVSPEMaker(FlatFieldSPEMaker):
    """
    This class represents a FlatFieldSingleHHVSPEMaker object.

    Args:
        charge (np.ma.masked_array or array-like): The charge data.
        counts (np.ma.masked_array or array-like): The counts data.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.
    Attributes:
        __charge (np.ma.masked_array): The charge data as a masked array.
        __counts (np.ma.masked_array): The counts data as a masked array.
        __pedestal (Parameter): The pedestal value.
        _parameters (list): List of parameters.
        __parameters_file (str): The path to the parameters file.
        _results (Table): Table of results.
    Methods:
    __init__: Initializes the FlatFieldSingleHHVSPEMaker object.
    create_from_chargesContainer: Creates an instance of FlatFieldSingleHHVSPEMaker using charge and counts data from a ChargesContainer object.
    create_from_run_number(cls, run_number, **kwargs): Class method that creates an instance from a run number.
    make(self, pixels_id=None, multiproc=True, display=True, **kwargs): Method that performs the fit on the specified pixels and returns the fit results.
    display(self, pixels_id, **kwargs): Method that plots the fit for the specified pixels.
    """

    __parameters_file = "parameters_signal.yaml"
    __fit_array = None
    _reduced_name = "FlatFieldSingleSPE"
    __nproc_default = 8
    __chunksize_default = 1

    # constructors
    def __init__(self, charge, counts, *args, **kwargs) -> None:
        """
        Initializes the FlatFieldSingleHHVSPEMaker object.
        Args:
            charge (np.ma.masked_array or array-like): The charge data.
            counts (np.ma.masked_array or array-like): The counts data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        if isinstance(charge, np.ma.masked_array):
            self.__charge = charge
        else:
            self.__charge = np.ma.asarray(charge)
        if isinstance(counts, np.ma.masked_array):
            self.__counts = counts
        else:
            self.__counts = np.ma.asarray(counts)
        self.__pedestal = Parameter(
            name="pedestal",
            value=(
                np.min(self.__charge)
                + np.sum(self.__charge * self.__counts) / np.sum(self.__counts)
            )
            / 2,
            min=np.min(self.__charge),
            max=np.sum(self.__charge * self.__counts) / np.sum(self.__counts),
            unit=u.dimensionless_unscaled,
        )
        self._parameters.append(self.__pedestal)
        self.read_param_from_yaml(kwargs.get("parameters_file", self.__parameters_file))
        self._update_table_from_parameters()
        self._results.add_column(
            Column(
                np.zeros((self.npixels), dtype=np.float64),
                "likelihood",
                unit=u.dimensionless_unscaled,
            )
        )
        self._results.add_column(
            Column(
                np.zeros((self.npixels), dtype=np.float64),
                "pvalue",
                unit=u.dimensionless_unscaled,
            )
        )

    @classmethod
    def create_from_chargesContainer(cls, signal: ChargesContainer, **kwargs):
        """
        Creates an instance of FlatFieldSingleHHVSPEMaker using charge and counts data from a ChargesContainer object.
        Args:
            signal (ChargesContainer): The ChargesContainer object.
            **kwargs: Additional keyword arguments.
        Returns:
            FlatFieldSingleHHVSPEMaker: An instance of FlatFieldSingleHHVSPEMaker.
        """
        histo = ChargesMaker.histo_hg(signal, autoscale=True)
        return cls(
            charge=histo[1], counts=histo[0], pixels_id=signal.pixels_id, **kwargs
        )

    @classmethod
    def create_from_run_number(cls, run_number: int, **kwargs):
        raise NotImplementedError(
            "Need to implement here the use of the WaveformsMaker and ChargesMaker to produce the chargesContainer to be pass into the __ini__"
        )

    # getters and setters
    @property
    def charge(self):
        """
        Returns a deep copy of the __charge attribute.
        """
        return copy.deepcopy(self.__charge)

    @property
    def _charge(self):
        """
        Returns the __charge attribute.
        """
        return self.__charge

    @property
    def counts(self):
        """
        Returns a deep copy of the __counts attribute.
        """
        return copy.deepcopy(self.__counts)

    @property
    def _counts(self):
        """
        Returns the __counts attribute.
        """
        return self.__counts

    # methods
    def _fill_results_table_from_dict(self, dico: dict, pixels_id: np.ndarray) -> None:
        """
        Populates the results table with fit values and errors for each pixel based on the dictionary provided as input.

        Args:
            dico (dict): A dictionary containing fit values and errors for each pixel.
            pixels_id (np.ndarray): An array of pixel IDs.

        Returns:
            None
        """
        chi2_sig = signature(__class__.cost(self._charge, self._counts))
        for i in range(len(pixels_id)):
            values = dico[i].get(f"values_{i}", None)
            errors = dico[i].get(f"errors_{i}", None)
            if not ((values is None) or (errors is None)):
                index = np.argmax(self._results["pixels_id"] == pixels_id[i])
                if len(values) != len(chi2_sig.parameters):
                    e = Exception(
                        "the size out the minuit output parameters values array does not fit the signature of the minimized cost function"
                    )
                    log.error(e, exc_info=True)
                    raise e
                for j, key in enumerate(chi2_sig.parameters):
                    self._results[key][index] = values[j]
                    self._results[f"{key}_error"][index] = errors[j]
                    if key == "mean":
                        self._high_gain[index] = values[j]
                        self._results[f"high_gain_error"][index] = [
                            errors[j],
                            errors[j],
                        ]
                        self._results[f"high_gain"][index] = values[j]
                self._results["is_valid"][index] = True
                self._results["likelihood"][index] = __class__.__fit_array[i].fcn(
                    __class__.__fit_array[i].values
                )
                ndof = (
                    self._counts.data[index][~self._counts.mask[index]].shape[0]
                    - __class__.__fit_array[i].nfit
                )
                self._results["pvalue"][index] = Statistics.chi2_pvalue(
                    ndof, __class__.__fit_array[i].fcn(__class__.__fit_array[i].values)
                )

    @staticmethod
    def _NG_Likelihood_Chi2(
        pp: float,
        res: float,
        mu2: float,
        n: float,
        muped: float,
        sigped: float,
        lum: float,
        charge: np.ndarray,
        counts: np.ndarray,
        **kwargs,
    ):
        """
        Calculates the chi-square value using the MPE2 function.
        Parameters:
        pp (float): The pp parameter.
        res (float): The res parameter.
        mu2 (float): The mu2 parameter.
        n (float): The n parameter.
        muped (float): The muped parameter.
        sigped (float): The sigped parameter.
        lum (float): The lum parameter.
        charge (np.ndarray): An array of charge values.
        counts (np.ndarray): An array of count values.
        **kwargs: Additional keyword arguments.
        Returns:
        float: The chi-square value.
        """
        pdf = MPE2(charge, pp, res, mu2, n, muped, sigped, lum, **kwargs)
        # log.debug(f"pdf : {np.sum(pdf)}")
        Ntot = np.sum(counts)
        # log.debug(f'Ntot : {Ntot}')
        mask = counts > 0
        Lik = np.sum(
            ((pdf * Ntot - counts)[mask]) ** 2 / counts[mask]
        )  # 2 times faster
        return Lik

    @staticmethod
    def cost(charge: np.ndarray, counts: np.ndarray):
        """
        Defines a function called Chi2 that calculates the chi-square value using the _NG_Likelihood_Chi2 method.
        Parameters:
        charge (np.ndarray): An array of charge values.
        counts (np.ndarray): An array of count values.
        Returns:
        function: The Chi2 function.
        """

        def Chi2(
            pedestal: float,
            pp: float,
            luminosity: float,
            resolution: float,
            mean: float,
            n: float,
            pedestalWidth: float,
        ):
            # assert not(np.isnan(pp) or np.isnan(resolution) or np.isnan(mean) or np.isnan(n) or np.isnan(pedestal) or np.isnan(pedestalWidth) or np.isnan(luminosity))
            for i in range(1000):
                if gammainc(i + 1, luminosity) < 1e-5:
                    ntotalPE = i
                    break
            kwargs = {"ntotalPE": ntotalPE}
            return __class__._NG_Likelihood_Chi2(
                pp,
                resolution,
                mean,
                n,
                pedestal,
                pedestalWidth,
                luminosity,
                charge,
                counts,
                **kwargs,
            )

        return Chi2

    # @njit(parallel=True,nopython = True)
    def _make_fit_array_from_parameters(
        self, pixels_id: np.ndarray = None, **kwargs
    ) -> np.ndarray:
        """
        Create an array of Minuit fit instances based on the parameters and data for each pixel.

        Args:
            pixels_id (optional): An array of pixel IDs. If not provided, all pixels will be used.

        Returns:
            np.ndarray: An array of Minuit fit instances, one for each pixel.
        """
        if pixels_id is None:
            npix = self.npixels
            pixels_id = self.pixels_id
        else:
            npix = len(pixels_id)

        fit_array = np.empty((npix), dtype=np.object_)

        for i, _id in enumerate(pixels_id):
            index = np.where(self.pixels_id == _id)[0][0]
            parameters = __class__._update_parameters(
                self.parameters,
                self._charge[index].data[~self._charge[index].mask],
                self._counts[index].data[~self._charge[index].mask],
                pixel_id=_id,
                **kwargs,
            )
            minuitParameters = UtilsMinuit.make_minuit_par_kwargs(parameters)
            minuit_kwargs = {
                parname: minuitParameters["values"][parname]
                for parname in minuitParameters["values"]
            }
            log.info(f"creation of fit instance for pixel: {_id}")
            fit_array[i] = Minuit(
                __class__.cost(
                    self._charge[index].data[~self._charge[index].mask],
                    self._counts[index].data[~self._charge[index].mask],
                ),
                **minuit_kwargs,
            )
            log.debug("fit created")
            fit_array[i].errordef = Minuit.LIKELIHOOD
            fit_array[i].strategy = 0
            fit_array[i].tol = 1e40
            fit_array[i].print_level = 1
            fit_array[i].throw_nan = True
            UtilsMinuit.set_minuit_parameters_limits_and_errors(
                fit_array[i], minuitParameters
            )
            log.debug(fit_array[i].values)
            log.debug(fit_array[i].limits)
            log.debug(fit_array[i].fixed)

        return fit_array

    @staticmethod
    def run_fit(i: int) -> dict:
        """
        Perform a fit on a specific pixel using the Minuit package.

        Args:
            i (int): The index of the pixel to perform the fit on.

        Returns:
            dict: A dictionary containing the fit values and errors for the specified pixel.
                  The keys are "values_i" and "errors_i", where "i" is the index of the pixel.
        """
        log.info("Starting")
        __class__.__fit_array[i].migrad()
        __class__.__fit_array[i].hesse()
        _values = np.array([params.value for params in __class__.__fit_array[i].params])
        _errors = np.array([params.error for params in __class__.__fit_array[i].params])
        log.info("Finished")
        return {f"values_{i}": _values, f"errors_{i}": _errors}

    def make(
        self,
        pixels_id: np.ndarray = None,
        multiproc: bool = True,
        display: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Perform a fit on specified pixels and return the fit results.

        Args:
            pixels_id (np.ndarray, optional): An array of pixel IDs to perform the fit on. If not provided, the fit will be performed on all pixels. Default is None.
            multiproc (bool, optional): A boolean indicating whether to use multiprocessing for the fit. Default is True.
            display (bool, optional): A boolean indicating whether to display the fit results. Default is True.
            **kwargs (optional): Additional keyword arguments.

        Returns:
            np.ndarray: An array of fit instances.

        Example Usage:
            # Initialize the FlatFieldSingleHHVSPEMaker object
            maker = FlatFieldSingleHHVSPEMaker(charge, counts)

            # Perform the fit on all pixels and display the fit results
            results = maker.make()

            # Perform the fit on specific pixels and display the fit results
            results = maker.make(pixels_id=[1, 2, 3])
        """
        log.info("running maker")
        log.info("checking asked pixels id")
        if pixels_id is None:
            pixels_id = self.pixels_id
            npix = self.npixels
        else:
            log.debug("checking that asked pixels id are in data")
            pixels_id = np.asarray(pixels_id)
            mask = np.array([_id in self.pixels_id for _id in pixels_id], dtype=bool)
            if False in mask:
                log.debug(f"The following pixels are not in data : {pixels_id[~mask]}")
                pixels_id = pixels_id[mask]
            npix = len(pixels_id)

        if npix == 0:
            log.warning("The asked pixels id are all out of the data")
            return None
        else:
            log.info("creation of the fits instance array")
            __class__.__fit_array = self._make_fit_array_from_parameters(
                pixels_id=pixels_id, display=display, **kwargs
            )

            log.info("running fits")
            if multiproc:
                nproc = kwargs.get("nproc", __class__.__nproc_default)
                chunksize = kwargs.get(
                    "chunksize",
                    max(__class__.__chunksize_default, npix // (nproc * 10)),
                )
                log.info(f"pooling with nproc {nproc}, chunksize {chunksize}")

                t = time.time()
                with Pool(nproc) as pool:
                    result = pool.starmap_async(
                        __class__.run_fit,
                        [(i,) for i in range(npix)],
                        chunksize=chunksize,
                    )
                    result.wait()
                try:
                    res = result.get()
                except Exception as e:
                    log.error(e, exc_info=True)
                    raise e
                log.debug(res)
                log.info(
                    f"time for multiproc with starmap_async execution is {time.time() - t:.2e} sec"
                )
            else:
                log.info("running in mono-cpu")
                t = time.time()
                res = [__class__.run_fit(i) for i in range(npix)]
                log.debug(res)
                log.info(f"time for singleproc execution is {time.time() - t:.2e} sec")

            log.info("filling result table from fits results")
            self._fill_results_table_from_dict(res, pixels_id)

            output = copy.copy(__class__.__fit_array)
            __class__.__fit_array = None

            if display:
                log.info("plotting")
                self.display(pixels_id, **kwargs)

            return output

    def plot_single(
        pixel_id: int,
        charge: np.ndarray,
        counts: np.ndarray,
        pp: float,
        resolution: float,
        gain: float,
        gain_error: float,
        n: float,
        pedestal: float,
        pedestalWidth: float,
        luminosity: float,
        likelihood: float,
    ) -> tuple:
        """
        Generate a plot of the data and a model fit for a specific pixel.

        Args:
            pixel_id (int): The ID of the pixel for which the plot is generated.
            charge (np.ndarray): An array of charge values.
            counts (np.ndarray): An array of event counts corresponding to the charge values.
            pp (float): The value of the `pp` parameter.
            resolution (float): The value of the `resolution` parameter.
            gain (float): The value of the `gain` parameter.
            gain_error (float): The value of the `gain_error` parameter.
            n (float): The value of the `n` parameter.
            pedestal (float): The value of the `pedestal` parameter.
            pedestalWidth (float): The value of the `pedestalWidth` parameter.
            luminosity (float): The value of the `luminosity` parameter.
            likelihood (float): The value of the `likelihood` parameter.

        Returns:
            tuple: A tuple containing the generated plot figure and the axes of the plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.errorbar(charge, counts, np.sqrt(counts), zorder=0, fmt=".", label="data")
        ax.plot(
            charge,
            np.trapz(counts, charge)
            * MPE2(
                charge,
                pp,
                resolution,
                gain,
                n,
                pedestal,
                pedestalWidth,
                luminosity,
            ),
            zorder=1,
            linewidth=2,
            label=f"SPE model fit \n gain : {gain - gain_error:.2f} < {gain:.2f} < {gain + gain_error:.2f} ADC/pe,\n likelihood :  {likelihood:.2f}",
        )
        ax.set_xlabel("Charge (ADC)", size=15)
        ax.set_ylabel("Events", size=15)
        ax.set_title(f"SPE fit pixel id : {pixel_id}")
        ax.set_xlim([pedestal - 6 * pedestalWidth, None])
        ax.legend(fontsize=18)
        return fig, ax

    def display(self, pixels_id: np.ndarray, **kwargs) -> None:
        """
        Display and save the plot for each specified pixel ID.

        Args:
            pixels_id (np.ndarray): An array of pixel IDs.
            **kwargs: Additional keyword arguments.
                figpath (str): The path to save the generated plot figures. Defaults to "/tmp/NectarGain_pid{os.getpid()}".
        """
        figpath = kwargs.get("figpath", f"/tmp/NectarGain_pid{os.getpid()}")
        os.makedirs(figpath, exist_ok=True)
        for _id in pixels_id:
            index = np.argmax(self._results["pixels_id"] == _id)
            fig, ax = __class__.plot_single(
                _id,
                self._charge[index],
                self._counts[index],
                self._results["pp"][index].value,
                self._results["resolution"][index].value,
                self._results["high_gain"][index].value,
                self._results["high_gain_error"][index].value.mean(),
                self._results["n"][index].value,
                self._results["pedestal"][index].value,
                self._results["pedestalWidth"][index].value,
                self._results["luminosity"][index].value,
                self._results["likelihood"][index],
            )
            fig.savefig(f"{figpath}/fit_SPE_pixel{_id}.pdf")
            fig.clf()
            plt.close(fig)
            del fig, ax


class FlatFieldSingleHHVStdSPEMaker(FlatFieldSingleHHVSPEMaker):
    """class to perform fit of the SPE signal with n and pp fixed"""

    __parameters_file = "parameters_signalStd.yaml"
    _reduced_name = "FlatFieldSingleStdSPE"

    def __init__(self, charge: np.ndarray, counts: np.ndarray, *args, **kwargs) -> None:
        """
        Initializes a new instance of the FlatFieldSingleHHVStdSPEMaker class.

        Args:
            charge (np.ndarray): The charge data.
            counts (np.ndarray): The counts data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(charge, counts, *args, **kwargs)
        self.__fix_parameters()

    def __fix_parameters(self) -> None:
        """
        Fixes the values of the n and pp parameters by setting their frozen attribute to True.
        """
        log.info("updating parameters by fixing pp and n")
        pp = self._parameters["pp"]
        pp.frozen = True
        n = self._parameters["n"]
        n.frozen = True


class FlatFieldSingleNominalSPEMaker(FlatFieldSingleHHVSPEMaker):
    """
    A class to perform a fit of the single photoelectron (SPE) signal at nominal voltage using fitted data obtained from a 1400V run.
    Inherits from FlatFieldSingleHHVSPEMaker.
    Fixes the parameters n, pp, and res.
    Optionally fixes the luminosity parameter.

    Args:
        charge (np.ndarray): The charge values.
        counts (np.ndarray): The counts values.
        nectarGainSPEresult (str): The path to the fitted data obtained from a 1400V run.
        same_luminosity (bool, optional): Whether to fix the luminosity parameter. Defaults to False.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        __parameters_file (str): The path to the parameters file for the fit at nominal voltage.
        _reduced_name (str): The name of the reduced data for the fit at nominal voltage.
        __same_luminosity (bool): Whether the luminosity parameter should be fixed.
        __nectarGainSPEresult (QTable): The fitted data obtained from a 1400V run, filtered for valid pixels.

    Example Usage:
        # Create an instance of FlatFieldSingleNominalSPEMaker
        maker = FlatFieldSingleNominalSPEMaker(charge, counts, nectarGainSPEresult='fit_result.txt', same_luminosity=True)

        # Perform the fit on the specified pixels and return the fit results
        results = maker.make(pixels_id=[1, 2, 3])

        # Plot the fit for the specified pixels
        maker.display(pixels_id=[1, 2, 3])
    """

    __parameters_file = "parameters_signal_fromHHVFit.yaml"
    _reduced_name = "FlatFieldSingleNominalSPE"

    def __init__(
        self,
        charge: np.ndarray,
        counts: np.ndarray,
        nectarGainSPEresult: str,
        same_luminosity: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes an instance of FlatFieldSingleNominalSPEMaker.

        Args:
            charge (np.ndarray): The charge values.
            counts (np.ndarray): The counts values.
            nectarGainSPEresult (str): The path to the fitted data obtained from a 1400V run.
            same_luminosity (bool, optional): Whether to fix the luminosity parameter. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(charge, counts, *args, **kwargs)
        self.__fix_parameters(same_luminosity)
        self.__same_luminosity = same_luminosity
        self.__nectarGainSPEresult = self._read_SPEresult(nectarGainSPEresult)
        if len(self.__nectarGainSPEresult) == 0:
            log.warning(
                "The intersection between pixels id from the data and those valid from the SPE fit result is empty"
            )

    @property
    def nectarGainSPEresult(self):
        """
        QTable: The fitted data obtained from a 1400V run, filtered for valid pixels.
        """
        return copy.deepcopy(self.__nectarGainSPEresult)

    @property
    def same_luminosity(self):
        """
        bool: Whether the luminosity parameter should be fixed.
        """
        return copy.deepcopy(self.__same_luminosity)

    def _read_SPEresult(self, nectarGainSPEresult: str):
        """
        Reads the fitted data obtained from a 1400V run and returns a filtered table of valid pixels.

        Args:
            nectarGainSPEresult (str): The path to the fitted data obtained from a 1400V run.

        Returns:
            QTable: The filtered table of valid pixels.
        """
        table = QTable.read(nectarGainSPEresult, format="ascii.ecsv")
        table = table[table["is_valid"]]
        argsort = []
        mask = []
        for _id in self._pixels_id:
            if _id in table["pixels_id"]:
                argsort.append(np.where(_id == table["pixels_id"])[0][0])
                mask.append(True)
            else:
                mask.append(False)
        self._pixels_id = self._pixels_id[np.array(mask)]
        return table[np.array(argsort)]

    def __fix_parameters(self, same_luminosity: bool) -> None:
        """
        Fixes the parameters n, pp, res, and possibly luminosity.

        Args:
            same_luminosity (bool): Whether to fix the luminosity parameter.
        """
        log.info("updating parameters by fixing pp, n and res")
        pp = self._parameters["pp"]
        pp.frozen = True
        n = self._parameters["n"]
        n.frozen = True
        resolution = self._parameters["resolution"]
        resolution.frozen = True
        if same_luminosity:
            log.info("fixing luminosity")
            luminosity = self._parameters["luminosity"]
            luminosity.frozen = True

    def _make_fit_array_from_parameters(self, pixels_id=None, **kwargs):
        """
        Generates the fit array from the fixed parameters and the fitted data obtained from a 1400V run.

        Args:
            pixels_id (array-like, optional): The pixels to generate the fit array for. Defaults to None.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            array-like: The fit array.
        """
        return super()._make_fit_array_from_parameters(
            pixels_id=pixels_id,
            nectarGainSPEresult=self.__nectarGainSPEresult,
            **kwargs,
        )

    @staticmethod
    def _update_parameters(
        parameters: Parameters,
        charge: np.ndarray,
        counts: np.ndarray,
        pixel_id,
        nectarGainSPEresult: QTable,
        **kwargs,
    ):
        """
        Updates the parameters with the fixed values from the fitted data obtained from a 1400V run.

        Args:
            parameters (Parameters): The parameters to update.
            charge (np.ndarray): The charge values.
            counts (np.ndarray): The counts values.
            pixel_id (int): The pixel ID.
            nectarGainSPEresult (QTable): The fitted data obtained from a 1400V run.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            dict: The updated parameters.
        """
        param = super()._update_parameters(parameters, charge, counts, **kwargs)
        luminosity = param["luminosity"]
        resolution = param["resolution"]
        pp = param["pp"]
        n = param["n"]

        index = np.where(pixel_id == nectarGainSPEresult["pixels_id"])[0][0]

        resolution.value = nectarGainSPEresult[index]["resolution"].value
        pp.value = nectarGainSPEresult[index]["pp"].value
        n.value = nectarGainSPEresult[index]["n"].value

        if luminosity.frozen:
            luminosity.value = nectarGainSPEresult[index]["luminosity"].value
        return param
