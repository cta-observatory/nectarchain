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
import matplotlib

import matplotlib.style as mplstyle
mplstyle.use('fast')

import numpy as np
import yaml
from abc import ABC, abstractmethod

from astropy.table import Column, QTable
from iminuit import Minuit
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.special import gammainc

from ctapipe.core.component import Component

from ctapipe.core.traits import Integer,Unicode,Bool,Path,Float
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer


from ....data.container import ChargesContainer,SPEfitContainer,merge_map_ArrayDataContainer
from ..chargesComponent import ChargesComponent
from .parameters import Parameter, Parameters
from ....utils import (
    MPE2,
    MeanValueError,
    PedestalValueError,
    Statistics,
    UtilsMinuit,
    weight_gaussian,
)

__all__ = ["SPEHHValgorithm","SPEHHVStdalgorithm","SPEnominalStdalgorithm","SPEnominalalgorithm","SPECombinedalgorithm"]

class SPEalgorithm(Component) : 
    Windows_lenght = Integer(40,
                            read_only = True,
                            help = "The windows leght used for the savgol filter algorithm",
    ).tag(config = True)
    
    Order = Integer(2,
                    read_only = True,
                    help = "The order of the polynome used in the savgol filter algorithm",
    ).tag(config = True)

    def __init__(self,pixels_id : np.ndarray,config=None, parent=None, **kwargs) -> None:
        super().__init__(config = config, parent = parent, **kwargs)
        self.__pixels_id = pixels_id
        self.__pedestal = Parameter(
            name="pedestal",
            value = 0,
        )
        self.__parameters = Parameters()
        self.__parameters.append(self.__pedestal)
        self.__results = SPEfitContainer(
            is_valid = np.zeros((self.npixels),dtype = bool),
            high_gain = np.empty((self.npixels,3)),
            low_gain = np.empty((self.npixels,3)),
            pixels_id = self.__pixels_id,
            likelihood = np.empty((self.npixels)),
            p_value = np.empty((self.npixels)),
            pedestal = np.empty((self.npixels,3)),
            pedestalWidth = np.empty((self.npixels,3)),
            resolution = np.empty((self.npixels,3)),
            luminosity = np.empty((self.npixels,3)),
            mean = np.empty((self.npixels,3)),
            n = np.empty((self.npixels,3)),
            pp = np.empty((self.npixels,3)),
        )

    @property
    def parameters(self):
        return copy.deepcopy(self.__parameters)

    @property
    def _parameters(self):
        return self.__parameters
    
    @property
    def results(self):
        return copy.deepcopy(self.__results)

    @property
    def _results(self):
        return self.__results

    @property
    def pixels_id(self) : 
        return copy.deepcopy(self.__pixels_id)
    
    @property
    def _pixels_id(self) : 
        return self.__pixels_id

    @property
    def npixels(self) : 
        return len(self.__pixels_id)

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
        charge: np.ndarray, counts: np.ndarray, pixel_id = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a Gaussian fit on the data to determine the pedestal and mean values.

        Args:
            charge (np.ndarray): An array of charge values.
            counts (np.ndarray): An array of corresponding counts.
            pixel_id (int): The id of the current pixel. Default to None
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
        windows_lenght = __class__.Windows_lenght.default_value
        order = __class__.Order.default_value
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
                f"{os.environ.get('NECTARCHAIN_LOG')}/{os.getpid()}/figures/initialization_pedestal_pixel{pixel_id}_{os.getpid()}.pdf"
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
                f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{os.getpid()}/figures/",
                exist_ok=True,
            )
            fig.savefig(
                f"{os.environ.get('NECTARCHAIN_LOG','/tmp')}/{os.getpid()}/figures/initialization_mean_pixel{pixel_id}_{os.getpid()}.pdf"
            )
            fig.clf()
            plt.close(fig)
            del fig, ax
        return coeff, coeff_mean

'''
    def _update_table_from_parameters(self) -> None:
        """
        Update the result table based on the parameters of the FlatFieldSPEMaker class.
        This method adds columns to the table for each parameter and its corresponding error.
        """

        for param in self._parameters.parameters:
            if not (param.name in self._results.keys()):
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
'''

class SPEnominalalgorithm(SPEalgorithm):

    parameters_file = Unicode("parameters_SPEnominal.yaml",
                                read_only = True,
                                help = "The name of the SPE fit parameters file",
    ).tag(config = True)

    __fit_array = None

    tol = Float(1e-1,
                help="The tolerance used for minuit",
                read_only = True,
    ).tag(config=True)
    
    nproc = Integer(8,
                    help = "The Number of cpu used for SPE fit",
    ).tag(config = True)

    chunksize = Integer(1,
                    help = "The chunk size for multi-processing",
    ).tag(config = True)

    multiproc = Bool(True,
                    help = "flag to active multi-processing",
    ).tag(config = True)

    def __init__(self,pixels_id : np.ndarray, charge : np.ndarray, counts: np.ndarray, config=None, parent=None, **kwargs) -> None:
        """
        Initializes the FlatFieldSingleHHVSPEMaker object.
        Args:
            charge (np.ma.masked_array or array-like): The charge data.
            counts (np.ma.masked_array or array-like): The counts data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(pixels_id = pixels_id, config=config, parent=parent, **kwargs)
        if isinstance(charge, np.ma.masked_array):
            self.__charge = charge
        else:
            self.__charge = np.ma.asarray(charge)
        if isinstance(counts, np.ma.masked_array):
            self.__counts = counts
        else:
            self.__counts = np.ma.asarray(counts)

        self.read_param_from_yaml(kwargs.get("parameters_file", self.parameters_file))


    @classmethod
    def create_from_chargesContainer(cls, signal: ChargesContainer, config=None, parent=None, **kwargs):
        """
        Creates an instance of FlatFieldSingleHHVSPEMaker using charge and counts data from a ChargesContainer object.
        Args:
            signal (ChargesContainer): The ChargesContainer object.
            **kwargs: Additional keyword arguments.
        Returns:
            FlatFieldSingleHHVSPEMaker: An instance of FlatFieldSingleHHVSPEMaker.
        """
        histo = ChargesComponent.histo_hg(signal, autoscale=True)
        return cls(
            pixels_id=signal.pixels_id,
            charge=histo[1], 
            counts=histo[0], 
            config=config, 
            parent=parent, 
            **kwargs
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
        ########NEED TO BE OPTIMIZED!!!###########
        chi2_sig = signature(__class__.cost(self._charge, self._counts))
        for i in range(len(pixels_id)):
            values = dico[i].get(f"values_{i}", None)
            errors = dico[i].get(f"errors_{i}", None)
            if not ((values is None) or (errors is None)):
                index = np.argmax(self._results.pixels_id == pixels_id[i])
                if len(values) != len(chi2_sig.parameters):
                    e = Exception(
                        "the size out the minuit output parameters values array does not fit the signature of the minimized cost function"
                    )
                    self.log.error(e, exc_info=True)
                    raise e
                for j, key in enumerate(chi2_sig.parameters):
                    self._results[key][index][0] = values[j]
                    self._results[key][index][1] = errors[j]
                    self._results[key][index][2] = errors[j]
                    if key == "mean":
                        self._results.high_gain[index][0] = values[j]
                        self._results.high_gain[index][1] = errors[j]
                        self._results.high_gain[index][2] = errors[j]
                self._results.is_valid[index] = True
                self._results.likelihood[index] = __class__.__fit_array[i].fcn(
                    __class__.__fit_array[i].values
                )
                ndof = (
                    self._counts.data[index][~self._counts.mask[index]].shape[0]
                    - __class__.__fit_array[i].nfit
                )
                self._results.p_value[index] = Statistics.chi2_pvalue(
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
            self.log.info(f"creation of fit instance for pixel: {_id}")
            fit_array[i] = Minuit(
                __class__.cost(
                    self._charge[index].data[~self._charge[index].mask],
                    self._counts[index].data[~self._charge[index].mask],
                ),
                **minuit_kwargs,
            )
            self.log.debug("fit created")
            fit_array[i].errordef = Minuit.LIKELIHOOD
            fit_array[i].strategy = 0
            fit_array[i].tol = self.tol
            fit_array[i].print_level = 1
            fit_array[i].throw_nan = True
            UtilsMinuit.set_minuit_parameters_limits_and_errors(
                fit_array[i], minuitParameters
            )
            #self.log.debug(fit_array[i].values)
            #self.log.debug(fit_array[i].limits)
            #self.log.debug(fit_array[i].fixed)

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

    def run(
        self,
        pixels_id: np.ndarray = None,
        display: bool = True,
        **kwargs,
    ) -> np.ndarray:
        
        self.log.info("running maker")
        self.log.info("checking asked pixels id")
        if pixels_id is None:
            pixels_id = self.pixels_id
            npix = self.npixels
        else:
            self.log.debug("checking that asked pixels id are in data")
            pixels_id = np.asarray(pixels_id)
            mask = np.array([_id in self.pixels_id for _id in pixels_id], dtype=bool)
            if False in mask:
                self.log.debug(f"The following pixels are not in data : {pixels_id[~mask]}")
                pixels_id = pixels_id[mask]
            npix = len(pixels_id)

        if npix == 0:
            self.log.warning("The asked pixels id are all out of the data")
            return None
        else:
            self.log.info("creation of the fits instance array")
            __class__.__fit_array = self._make_fit_array_from_parameters(
                pixels_id=pixels_id, display=display, **kwargs
            )

            self.log.info("running fits")
            if self.multiproc:
                nproc = kwargs.get("nproc", self.nproc)
                chunksize = kwargs.get(
                    "chunksize",
                    max(self.chunksize, npix // (nproc * 10)),
                )
                self.log.info(f"pooling with nproc {nproc}, chunksize {chunksize}")

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
                    self.log.error(e, exc_info=True)
                    raise e
                self.log.debug(str(res))
                self.log.info(
                    f"time for multiproc with starmap_async execution is {time.time() - t:.2e} sec"
                )
            else:
                self.log.info("running in mono-cpu")
                t = time.time()
                res = [__class__.run_fit(i) for i in range(npix)]
                self.log.debug(res)
                self.log.info(f"time for singleproc execution is {time.time() - t:.2e} sec")

            self.log.info("filling result table from fits results")
            self._fill_results_table_from_dict(res, pixels_id)

            output = copy.copy(__class__.__fit_array)
            __class__.__fit_array = None

            if display:
                self.log.info("plotting")
                t = time.time()
                self.display(pixels_id, **kwargs)
                log.info(f"time for plotting {len(pixels_id)} pixels : {time.time() - t:.2e} sec")

            return output
    def plot_single_pyqtgraph(
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
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtGui, QtCore
        #from pyqtgraph.Qt import QtGui

        app = pg.mkQApp(name = 'minimal')
#
        ## Create a window
        win = pg.GraphicsLayoutWidget(show=False)
        win.setWindowTitle(f"SPE fit pixel id : {pixel_id}")

        # Add a plot to the window
        plot = win.addPlot(title = f"SPE fit pixel id : {pixel_id}")
        
        plot.addLegend()
        error = pg.ErrorBarItem(
            x=charge,
            y=counts,
            top=np.sqrt(counts), 
            bottom=np.sqrt(counts), 
            beam=0.5)
        plot.addItem(error)
        plot.plot(
            x=charge,
            y=np.trapz(counts, charge)
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
            name = f"SPE model fit",
        )
        legend = pg.TextItem(
            f"SPE model fit gain : {gain - gain_error:.2f} < {gain:.2f} < {gain + gain_error:.2f} ADC/pe,\n likelihood :  {likelihood:.2f}",
            color=(200, 200, 200),          
        )
        legend.setPos(pedestal,np.max(counts)/2)
        font = QtGui.QFont()
        font.setPointSize(12)
        legend.setFont(font)
        legend.setTextWidth(500)
        plot.addItem(legend)
        
        label_style = {"color": "#EEE", "font-size": "12pt"}
        plot.setLabel("bottom", "Charge (ADC)", **label_style)
        plot.setLabel("left", "Events",**label_style)
        plot.setRange(xRange=[pedestal - 6 * pedestalWidth, np.quantile(charge.data[~charge.mask],0.84)])
        #ax.legend(fontsize=18)
        return win

    def plot_single_matplotlib(
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
        ax.set_xlim([pedestal - 6 * pedestalWidth, np.max(charge)])
        ax.legend(fontsize=18)
        return fig, ax

    def display(self, pixels_id: np.ndarray, package = "pyqtgraph", **kwargs) -> None:
        """
        Display and save the plot for each specified pixel ID.

        Args:
            pixels_id (np.ndarray): An array of pixel IDs.
            package (str): the package use to plot, can be matplotlib or pyqtgraph. Default to pyqtgraph
            **kwargs: Additional keyword arguments.
                figpath (str): The path to save the generated plot figures. Defaults to "/tmp/NectarGain_pid{os.getpid()}".
        """
        figpath = kwargs.get("figpath", f"{os.environ.get('NECTARCHAIN_FIGURES','/tmp')}/NectarGain_pid{os.getpid()}")
        self.log.debug(f"saving figures in {figpath}")
        os.makedirs(figpath, exist_ok=True)
        if package == "matplotlib" : 
            matplotlib.use('TkAgg') 
            for _id in pixels_id:
                index = np.argmax(self._results.pixels_id == _id)
                fig, ax = __class__.plot_single_matplotlib(
                    _id,
                    self._charge[index],
                    self._counts[index],
                    self._results.pp[index][0],
                    self._results.resolution[index][0],
                    self._results.high_gain[index][0],
                    self._results.high_gain[index][1:].mean(),
                    self._results.n[index][0],
                    self._results.pedestal[index][0],
                    self._results.pedestalWidth[index][0],
                    self._results.luminosity[index][0],
                    self._results.likelihood[index],
                )
                fig.savefig(f"{figpath}/fit_SPE_pixel{_id}.pdf")
                fig.clf()
                plt.close(fig)
                del fig, ax
        elif package =="pyqtgraph" :
            import pyqtgraph as pg
            import pyqtgraph.exporters
            for _id in pixels_id:
                index = np.argmax(self._results.pixels_id == _id)
                try : 
                    widget = None
                    widget = __class__.plot_single_pyqtgraph(
                        _id,
                        self._charge[index],
                        self._counts[index],
                        self._results.pp[index][0],
                        self._results.resolution[index][0],
                        self._results.high_gain[index][0],
                        self._results.high_gain[index][1:].mean(),
                        self._results.n[index][0],
                        self._results.pedestal[index][0],
                        self._results.pedestalWidth[index][0],
                        self._results.luminosity[index][0],
                        self._results.likelihood[index],
                    )
                    exporter = pg.exporters.ImageExporter(widget.getItem(0,0))
                    exporter.parameters()['width'] = 1000
                    exporter.export(f"{figpath}/fit_SPE_pixel{_id}.png")
                except Exception as e : 
                    log.warning(e,exc_info = True)
                finally : 
                    del widget



class SPEHHValgorithm(SPEnominalalgorithm):
    """class to perform fit of the SPE HHV signal with n and pp free"""

    parameters_file = Unicode("parameters_SPEHHV.yaml",
                                read_only = True,
                                help = "The name of the SPE fit parameters file",
    ).tag(config = True)
    tol = Float(1e40,
                help="The tolerance used for minuit",
                read_only = True,
    ).tag(config=True)


class SPEnominalStdalgorithm(SPEnominalalgorithm):
    """class to perform fit of the SPE signal with n and pp fixed"""

    parameters_file = Unicode("parameters_SPEnominalStd.yaml",
                                read_only = True,
                                help = "The name of the SPE fit parameters file",
    ).tag(config = True)

    def __init__(self, pixels_id : np.ndarray, charge: np.ndarray, counts: np.ndarray,config=None, parent=None, **kwargs) -> None:
        """
        Initializes a new instance of the FlatFieldSingleHHVStdSPEMaker class.

        Args:
            charge (np.ndarray): The charge data.
            counts (np.ndarray): The counts data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(pixels_id=pixels_id,charge=charge, counts = counts,config=config, parent=parent, **kwargs)
        self.__fix_parameters()

    def __fix_parameters(self) -> None:
        """
        Fixes the values of the n and pp parameters by setting their frozen attribute to True.
        """
        self.log.info("updating parameters by fixing pp and n")
        pp = self._parameters["pp"]
        pp.frozen = True
        n = self._parameters["n"]
        n.frozen = True

class SPEHHVStdalgorithm(SPEnominalStdalgorithm) : 
    parameters_file = Unicode(
        "parameters_SPEHHVStd.yaml",
        read_only = True,
        help = "The name of the SPE fit parameters file",
    ).tag(config = True)
    tol = Float(1e40,
        help="The tolerance used for minuit",
        read_only = True,
    ).tag(config=True)


class SPECombinedalgorithm(SPEnominalalgorithm):
    parameters_file = Unicode("parameters_SPECombined_fromHHVFit.yaml",
                                read_only = True,
                                help = "The name of the SPE fit parameters file",
    ).tag(config = True)

    tol = Float(1e5,
        help="The tolerance used for minuit",
        read_only = True,
    ).tag(config=True)

    SPE_result = Path(
        help="the path of the SPE result container computed with very high voltage data",
    ).tag(config = True)

    same_luminosity = Bool(
        help="if the luminosity is the same between high voltage and low voltage runs",
        default_value = False,
    ).tag(config = True)

    def __init__(self, pixels_id : np.ndarray, charge: np.ndarray, counts: np.ndarray,config=None, parent=None, **kwargs) -> None:
        """
        Initializes a new instance of the FlatFieldSingleHHVStdSPEMaker class.

        Args:
            charge (np.ndarray): The charge data.
            counts (np.ndarray): The counts data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(pixels_id=pixels_id,charge=charge, counts = counts,config=config, parent=parent, **kwargs)

        self.__fix_parameters()
        self._nectarGainSPEresult = SPEfitContainer.from_hdf5(self.SPE_result)
        if len(pixels_id[np.in1d(pixels_id, self._nectarGainSPEresult.pixels_id[self._nectarGainSPEresult.is_valid])]) == 0:
            self.log.warning(
                "The intersection between pixels id from the data and those valid from the SPE fit result is empty"
            )

    def __fix_parameters(self) -> None:
        """
        Fixes the parameters n, pp, res, and possibly luminosity.

        Args:
            same_luminosity (bool): Whether to fix the luminosity parameter.
        """
        self.log.info("updating parameters by fixing pp, n and res")
        pp = self._parameters["pp"]
        pp.frozen = True
        n = self._parameters["n"]
        n.frozen = True
        resolution = self._parameters["resolution"]
        resolution.frozen = True
        if self.same_luminosity:
            self.log.info("fixing luminosity")
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
            nectarGainSPEresult=self._nectarGainSPEresult,
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
        param = super(__class__,__class__)._update_parameters(parameters, charge, counts, **kwargs)
        luminosity = param["luminosity"]
        resolution = param["resolution"]
        pp = param["pp"]
        n = param["n"]

        index = np.where(pixel_id == nectarGainSPEresult.pixels_id)[0][0]

        resolution.value = nectarGainSPEresult.resolution[index]
        pp.value = nectarGainSPEresult.pp[index]
        n.value = nectarGainSPEresult.n[index]["n"]

        if luminosity.frozen:
            luminosity.value = nectarGainSPEresult.luminosity[index].value
        return param

    def run(
            self,
            pixels_id: np.ndarray = None,
            display: bool = True,
            **kwargs,
        ) -> np.ndarray :
        if pixels_id is None : 
            pixels_id = self._nectarGainSPEresult.pixels_id[self._nectarGainSPEresult.is_valid]
        else : 
            pixels_id = np.asarray(pixels_id)[np.in1d(pixels_id,self._nectarGainSPEresult.pixels_id[self._nectarGainSPEresult.is_valid])]
        return super().run(pixels_id=pixels_id,display = display,**kwargs)
