import sys
import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers


import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt
import astropy.units as u
from astropy.visualization import quantity_support
from astropy.table import QTable,Column
import os
from datetime import date
from pathlib import Path
import copy

from ctapipe_io_nectarcam import constants

from ....data.container import ChargesContainer,ChargesContainerIO

from ...chargesMakers import ChargesMaker

from .gainMakers import GainMaker


__all__ = ["PhotoStatisticMaker"]


class PhotoStatisticMaker(GainMaker):
    """
    The `PhotoStatisticMaker` class is a subclass of `GainMaker` and is used to calculate photo statistics for a given set of charge data. It provides methods to create an instance from charge containers or run numbers, as well as methods to calculate various statistics such as gain and standard deviation.

    Example Usage:
        # Create an instance of PhotoStatisticMaker using charge containers
        FFcharge = ChargesContainer(...)
        Pedcharge = ChargesContainer(...)
        coefCharge_FF_Ped = 0.5
        SPE_result = "path/to/SPE_results"
        photo_stat = PhotoStatisticMaker.create_from_chargeContainer(FFcharge, Pedcharge, coefCharge_FF_Ped, SPE_result)

        # Calculate and retrieve the gain values
        gain_hg = photo_stat.gainHG
        gain_lg = photo_stat.gainLG

        # Plot the correlation between photo statistic gain and SPE gain
        photo_stat_gain = np.array(...)
        SPE_gain = np.array(...)
        fig = PhotoStatisticMaker.plot_correlation(photo_stat_gain, SPE_gain)

    Methods:
        - `__init__(self, FFcharge_hg, FFcharge_lg, Pedcharge_hg, Pedcharge_lg, coefCharge_FF_Ped, SPE_resolution, *args, **kwargs)`: Constructor method to initialize the `PhotoStatisticMaker` instance with charge data and other parameters.
        - `create_from_chargeContainer(cls, FFcharge, Pedcharge, coefCharge_FF_Ped, SPE_result, **kwargs)`: Class method to create an instance of `PhotoStatisticMaker` from charge containers.
        - `create_from_run_numbers(cls, FFrun, Pedrun, SPE_result, **kwargs)`: Class method to create an instance of `PhotoStatisticMaker` from run numbers.
        - `__readSPE(SPEresults) -> tuple`: Static method to read SPE resolution from a file and return the resolution and pixel IDs.
        - `__get_charges_FF_Ped_reshaped(FFcharge, Pedcharge, SPE_resolution, SPE_pixels_id) -> dict`: Static method to reshape the charge data based on the intersection of pixel IDs and return a dictionary of reshaped data.
        - `__readFF(FFRun, **kwargs) -> dict`: Static method to read FF data from a file and return the FF charge data and coefficient.
        - `__readPed(PedRun, **kwargs) -> dict`: Static method to read Ped data from a file and return the Ped charge data.
        - `__check_shape(self) -> None`: Method to check the shape of the charge data arrays.
        - `make(self, **kwargs) -> None`: Method to run the photo statistic method and store the results.
        - `plot_correlation(photoStat_gain, SPE_gain) -> fig`: Static method to plot the correlation between photo statistic gain and SPE gain.

    Fields:
        - `SPE_resolution`: Property to get the SPE resolution.
        - `sigmaPedHG`: Property to get the standard deviation of Pedcharge_hg.
        - `sigmaChargeHG`: Property to get the standard deviation of FFcharge_hg - meanPedHG.
        - `meanPedHG`: Property to get the mean of Pedcharge_hg.
        - `meanChargeHG`: Property to get the mean of FFcharge_hg - meanPedHG.
        - `BHG`: Property to calculate the BHG value.
        - `gainHG`: Property to calculate the gain for high gain.
        - `sigmaPedLG`: Property to get the standard deviation of Pedcharge_lg.
        - `sigmaChargeLG`: Property to get the standard deviation of FFcharge_lg - meanPedLG.
        - `meanPedLG`: Property to get the mean of Pedcharge_lg.
        - `meanChargeLG`: Property to get the mean of FFcharge_lg - meanPedLG.
        - `BLG`: Property to calculate the BLG value.
        - `gainLG`: Property to calculate the gain for low gain.
    """
    _reduced_name = "PhotoStatistic"

#constructors
    def __init__(self,
                 FFcharge_hg : np.ndarray,
                 FFcharge_lg : np.ndarray,
                 Pedcharge_hg: np.ndarray,
                 Pedcharge_lg: np.ndarray,
                 coefCharge_FF_Ped : float,
                 SPE_resolution,
                 *args,
                 **kwargs
                 ) -> None:
        """
        Initializes the instance of the PhotoStatisticMaker class with charge data and other parameters.

        Args:
            FFcharge_hg (np.ndarray): Array of charge data for high gain in the FF (Flat Field) image.
            FFcharge_lg (np.ndarray): Array of charge data for low gain in the FF image.
            Pedcharge_hg (np.ndarray): Array of charge data for high gain in the Ped (Pedestal) image.
            Pedcharge_lg (np.ndarray): Array of charge data for low gain in the Ped image.
            coefCharge_FF_Ped (float): Coefficient to convert FF charge to Ped charge.
            SPE_resolution: Array-like of single photoelectron (SPE) resolutions for each pixel, or single value to use the same for each pixel.

        Raises:
            TypeError: If SPE_resolution is not provided in a valid format.

        Returns:
            None
        """
        super().__init__(*args,**kwargs)

        self.__coefCharge_FF_Ped = coefCharge_FF_Ped
    
        self.__FFcharge_hg = FFcharge_hg
        self.__FFcharge_lg = FFcharge_lg

        self.__Pedcharge_hg = Pedcharge_hg
        self.__Pedcharge_lg = Pedcharge_lg

        if isinstance(SPE_resolution,np.ndarray) and len(SPE_resolution) == self.npixels : 
            self.__SPE_resolution = SPE_resolution
        elif isinstance(SPE_resolution,list) and len(SPE_resolution) == self.npixels : 
            self.__SPE_resolution = np.array(SPE_resolution)
        elif isinstance(SPE_resolution,float) :
            self.__SPE_resolution = SPE_resolution * np.ones((self.npixels)) 
        else : 
            e = TypeError("SPE_resolution must be a float, a numpy.ndarray or list instance")
            raise e

        self.__check_shape()


    @classmethod
    def create_from_chargeContainer(cls, 
                                    FFcharge : ChargesContainer, 
                                    Pedcharge : ChargesContainer, 
                                    coefCharge_FF_Ped : float, 
                                    SPE_result, 
                                    **kwargs) : 
        """
        Create an instance of the PhotoStatisticMaker class from Pedestal and Flatfield runs stored in ChargesContainer.
    
        Args:
            FFcharge (ChargesContainer): Array of charge data for the FF image.
            Pedcharge (ChargesContainer): Array of charge data for the Ped image.
            coefCharge_FF_Ped (float): Coefficient to convert FF charge to Ped charge.
            SPE_result (str or Path): Path to the SPE result file (optional).
            **kwargs: Additional keyword arguments for initializing the PhotoStatisticMaker instance.
        
        Returns:
            PhotoStatisticMaker: An instance of the PhotoStatisticMaker class created from the ChargesContainer instances.
        """
        if isinstance(SPE_result , str) or isinstance(SPE_result , Path) : 
            SPE_resolution,SPE_pixels_id = __class__.__readSPE(SPE_result)
        else : 
            SPE_pixels_id = None

        kwargs_init =  __class__.__get_charges_FF_Ped_reshaped(FFcharge,
                                                            Pedcharge,
                                                            SPE_resolution,
                                                            SPE_pixels_id)

        kwargs.update(kwargs_init)
        return cls(coefCharge_FF_Ped = coefCharge_FF_Ped, **kwargs)

    @classmethod
    def create_from_run_numbers(cls, FFrun: int, Pedrun: int, SPE_result: str, **kwargs):
        """
        Create an instance of the PhotoStatisticMaker class by reading the FF (Flat Field) and Ped (Pedestal) charge data from run numbers.
    
        Args:
            FFrun (int): The run number for the FF charge data.
            Pedrun (int): The run number for the Ped charge data.
            SPE_result (str): The path to the SPE result file.
            **kwargs: Additional keyword arguments.
        
        Returns:
            PhotoStatisticMaker: An instance of the PhotoStatisticMaker class created from the FF and Ped charge data and the SPE result file.
        """
        FFkwargs = __class__.__readFF(FFrun, **kwargs)
        Pedkwargs = __class__.__readPed(Pedrun, **kwargs)
        kwargs.update(FFkwargs)
        kwargs.update(Pedkwargs)
        return cls.create_from_chargeContainer(SPE_result=SPE_result, **kwargs)

#methods
    @staticmethod
    def __readSPE(SPEresults) -> tuple: 
        """
        Reads the SPE resolution from a file and returns the resolution values and corresponding pixel IDs.

        Args:
            SPEresults (str): The file path to the SPE results file.

        Returns:
            tuple: A tuple containing the SPE resolution values and corresponding pixel IDs.
        """
        log.info(f'reading SPE resolution from {SPEresults}')
        table = QTable.read(SPEresults)
        table.sort('pixels_id')
        return table['resolution'][table['is_valid']].value,table['pixels_id'][table['is_valid']].value

    @staticmethod
    def __get_charges_FF_Ped_reshaped( FFcharge : ChargesContainer, Pedcharge : ChargesContainer, SPE_resolution : np.ndarray, SPE_pixels_id: np.ndarray)-> dict : 
        """
        Reshapes the FF (Flat Field) and Ped (Pedestal) charges based on the intersection of pixel IDs between the two charges.
        Selects the charges for the high-gain and low-gain channels and returns them along with the common pixel IDs.

        Args:
            FFcharge (ChargesContainer): The charges container for the Flat Field data.
            Pedcharge (ChargesContainer): The charges container for the Pedestal data.
            SPE_resolution (np.ndarray): An array containing the SPE resolutions.
            SPE_pixels_id (np.ndarray): An array containing the pixel IDs for the SPE data.

        Returns:
            dict: A dictionary containing the reshaped data, including the common pixel IDs, SPE resolution (if provided), and selected charges for the high-gain and low-gain channels.
        """
        log.info("reshape of SPE, Ped and FF data with intersection of pixel ids")
        out = {}

        FFped_intersection =  np.intersect1d(Pedcharge.pixels_id,FFcharge.pixels_id)
        if not(SPE_pixels_id is None) : 
            SPEFFPed_intersection = np.intersect1d(FFped_intersection,SPE_pixels_id)
            mask_SPE = np.array([SPE_pixels_id[i] in SPEFFPed_intersection for i in range(len(SPE_pixels_id))],dtype = bool)
            out["SPE_resolution"] = SPE_resolution[mask_SPE]

        out["pixels_id"] = SPEFFPed_intersection
        out["FFcharge_hg"] = ChargesMaker.select_charges_hg(FFcharge,SPEFFPed_intersection)
        out["FFcharge_lg"] = ChargesMaker.select_charges_lg(FFcharge,SPEFFPed_intersection)
        out["Pedcharge_hg"] = ChargesMaker.select_charges_hg(Pedcharge,SPEFFPed_intersection)
        out["Pedcharge_lg"] = ChargesMaker.select_charges_lg(Pedcharge,SPEFFPed_intersection)
        
        log.info(f"data have {len(SPEFFPed_intersection)} pixels in common")
        return out

    @staticmethod
    def __readFF(FFRun: int, **kwargs) -> dict:
        """
        Reads FF charge data from a FITS file.
        Args:
        - FFRun (int): The run number for the FF data.
        - kwargs (optional): Additional keyword arguments.
        Returns:
        - dict: A dictionary containing the FF charge data (`FFcharge`) and the coefficient for the FF charge (`coefCharge_FF_Ped`).
        """
        log.info('reading FF data')
        method = kwargs.get('method', 'FullWaveformSum')
        FFchargeExtractorWindowLength = kwargs.get('FFchargeExtractorWindowLength', None)
        if method != 'FullWaveformSum':
            if FFchargeExtractorWindowLength is None:
                e = Exception(f"we have to specify FFchargeExtractorWindowLength argument if charge extractor method is not FullwaveformSum")
                log.error(e, exc_info=True)
                raise e
            else:
                coefCharge_FF_Ped = FFchargeExtractorWindowLength / constants.N_SAMPLES
        else:
            coefCharge_FF_Ped = 1
        if isinstance(FFRun, int):
            try:
                FFcharge = ChargesContainerIO.load(f"{os.environ['NECTARCAMDATA']}/charges/{method}", FFRun)
                log.info(f'charges have ever been computed for FF run {FFRun}')
            except Exception as e:
                log.error("charge have not been yet computed")
                raise e
        else:
            e = TypeError("FFRun must be int")
            log.error(e, exc_info=True)
            raise e
        return {"FFcharge": FFcharge, "coefCharge_FF_Ped": coefCharge_FF_Ped}
    @staticmethod
    def __readPed(PedRun: int, **kwargs) -> dict:
        """
        Reads Ped charge data from a FITS file.
        Args:
        - PedRun (int): The run number for the Ped data.
        - kwargs (optional): Additional keyword arguments.
        Returns:
        - dict: A dictionary containing the Ped charge data (`Pedcharge`).
        """
        log.info('reading Ped data')
        method = 'FullWaveformSum'  # kwargs.get('method','std')
        if isinstance(PedRun, int):
            try:
                Pedcharge = ChargesContainerIO.load(f"{os.environ['NECTARCAMDATA']}/charges/{method}", PedRun)
                log.info(f'charges have ever been computed for Ped run {PedRun}')
            except Exception as e:
                log.error("charge have not been yet computed")
                raise e
        else:
            e = TypeError("PedRun must be int")
            log.error(e, exc_info=True)
            raise e
        return {"Pedcharge": Pedcharge}

    def __check_shape(self) -> None: 
        """
        Checks the shape of certain attributes and raises an exception if the shape is not as expected.
        """
        try : 
            self.__FFcharge_hg[0] * self.__FFcharge_lg[0] * self.__Pedcharge_hg[0] * self.__Pedcharge_lg[0] * self.__SPE_resolution * self._pixels_id
        except Exception as e : 
            log.error(e,exc_info = True)
            raise e

    def make(self, **kwargs) -> None:
        """
        Runs the photo statistic method and assigns values to the high_gain and low_gain keys in the _results dictionary.

        Args:
            **kwargs: Additional keyword arguments (not used in this method).

        Returns:
            None
        """
        log.info('running photo statistic method')
        self._results["high_gain"] = self.gainHG
        self._results["low_gain"] = self.gainLG
        # self._results["is_valid"] = self._SPEvalid


    def plot_correlation(photoStat_gain: np.ndarray, SPE_gain: np.ndarray) -> plt.Figure:
        """
        Plot the correlation between the photo statistic gain and the single photoelectron (SPE) gain.

        Args:
            photoStat_gain (np.ndarray): Array of photo statistic gain values.
            SPE_gain (np.ndarray): Array of SPE gain values.

        Returns:
            fig (plt.Figure): The figure object containing the scatter plot and the linear fit line.
        """

        # Create a mask to filter the data points based on certain criteria
        mask = (photoStat_gain > 20) * (SPE_gain > 0) * (photoStat_gain < 80)

        # Perform a linear regression analysis on the filtered data points
        a, b, r, p_value, std_err = linregress(photoStat_gain[mask], SPE_gain[mask], 'greater')

        # Generate a range of x-values for the linear fit line
        x = np.linspace(photoStat_gain[mask].min(), photoStat_gain[mask].max(), 1000)

        # Define a lambda function for the linear fit line
        y = lambda x: a * x + b

        with quantity_support():
            # Create a scatter plot of the filtered data points
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(photoStat_gain[mask], SPE_gain[mask], marker=".")

            # Plot the linear fit line using the x-values and the lambda function
            ax.plot(x, y(x), color='red',
                    label=f"linear fit,\n a = {a:.2e},\n b = {b:.2e},\n r = {r:.2e},\n p_value = {p_value:.2e},\n std_err = {std_err:.2e}")

            # Plot the line y = x
            ax.plot(x, x, color='black', label="y = x")

            ax.set_xlabel("Gain Photo stat (ADC)", size=15)
            ax.set_ylabel("Gain SPE fit (ADC)", size=15)
            ax.legend(fontsize=15)

        return fig

@property
def SPE_resolution(self) -> float:
    """
    Returns a deep copy of the SPE resolution.

    Returns:
        float: The SPE resolution.
    """
    return copy.deepcopy(self.__SPE_resolution)


@property
def sigmaPedHG(self) -> float:
    """
    Calculates and returns the standard deviation of Pedcharge_hg multiplied by the square root of coefCharge_FF_Ped.

    Returns:
        float: The standard deviation of Pedcharge_hg.
    """
    return np.std(self.__Pedcharge_hg, axis=0) * np.sqrt(self.__coefCharge_FF_Ped)


@property
def sigmaChargeHG(self) -> float:
    """
    Calculates and returns the standard deviation of FFcharge_hg minus meanPedHG.

    Returns:
        float: The standard deviation of FFcharge_hg minus meanPedHG.
    """
    return np.std(self.__FFcharge_hg - self.meanPedHG, axis=0)


@property
def meanPedHG(self) -> float:
    """
    Calculates and returns the mean of Pedcharge_hg multiplied by coefCharge_FF_Ped.

    Returns:
        float: The mean of Pedcharge_hg.
    """
    return np.mean(self.__Pedcharge_hg, axis=0) * self.__coefCharge_FF_Ped


@property
def meanChargeHG(self) -> float:
    """
    Calculates and returns the mean of FFcharge_hg minus meanPedHG.

    Returns:
        float: The mean of FFcharge_hg minus meanPedHG.
    """
    return np.mean(self.__FFcharge_hg - self.meanPedHG, axis=0)


@property
def BHG(self) -> float:
    """
    Calculates and returns the BHG value.

    Returns:
        float: The BHG value.
    """
    min_events = np.min((self.__FFcharge_hg.shape[0], self.__Pedcharge_hg.shape[0]))
    upper = (np.power(self.__FFcharge_hg.mean(axis=1)[:min_events] - self.__Pedcharge_hg.mean(axis=1)[:min_events] * self.__coefCharge_FF_Ped - self.meanChargeHG.mean(), 2)).mean(axis=0)
    lower = np.power(self.meanChargeHG.mean(), 2)
    return np.sqrt(upper / lower)


@property
def gainHG(self) -> float:
    """
    Calculates and returns the gain for high gain charge data.

    Returns:
        float: The gain for high gain charge data.
    """
    return ((np.power(self.sigmaChargeHG, 2) - np.power(self.sigmaPedHG, 2) - np.power(self.BHG * self.meanChargeHG, 2))
            / (self.meanChargeHG * (1 + np.power(self.SPE_resolution, 2))))


@property
def sigmaPedLG(self) -> float:
    """
    Calculates and returns the standard deviation of Pedcharge_lg multiplied by the square root of coefCharge_FF_Ped.

    Returns:
        float: The standard deviation of Pedcharge_lg.
    """
    return np.std(self.__Pedcharge_lg, axis=0) * np.sqrt(self.__coefCharge_FF_Ped)


@property
def sigmaChargeLG(self) -> float:
    """
    Calculates and returns the standard deviation of FFcharge_lg minus meanPedLG.

    Returns:
        float: The standard deviation of FFcharge_lg minus meanPedLG.
    """
    return np.std(self.__FFcharge_lg - self.meanPedLG, axis=0)


@property
def meanPedLG(self) -> float:
    """
    Calculates and returns the mean of Pedcharge_lg multiplied by coefCharge_FF_Ped.

    Returns:
        float: The mean of Pedcharge_lg.
    """
    return np.mean(self.__Pedcharge_lg, axis=0) * self.__coefCharge_FF_Ped


@property
def meanChargeLG(self) -> float:
    """
    Calculates and returns the mean of FFcharge_lg minus meanPedLG.

    Returns:
        float: The mean of FFcharge_lg minus meanPedLG.
    """
    return np.mean(self.__FFcharge_lg - self.meanPedLG, axis=0)


@property
def BLG(self) -> float:
    """
    Calculates and returns the BLG value.

    Returns:
        float: The BLG value.
    """
    min_events = np.min((self.__FFcharge_lg.shape[0], self.__Pedcharge_lg.shape[0]))
    upper = (np.power(self.__FFcharge_lg.mean(axis=1)[:min_events] - self.__Pedcharge_lg.mean(axis=1)[:min_events] * self.__coefCharge_FF_Ped - self.meanChargeLG.mean(), 2)).mean(axis=0)
    lower = np.power(self.meanChargeLG.mean(), 2)
    return np.sqrt(upper / lower)


@property
def gainLG(self) -> float:
    """
    Calculates and returns the gain for low gain charge data.

    Returns:
        float: The gain for low gain charge data.
    """
    return ((np.power(self.sigmaChargeLG, 2) - np.power(self.sigmaPedLG, 2) - np.power(self.BLG * self.meanChargeLG, 2))
            / (self.meanChargeLG * (1 + np.power(self.SPE_resolution, 2))))

