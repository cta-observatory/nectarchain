import copy
import logging
import os

import numpy as np
from astropy.visualization import quantity_support
from ctapipe.core import Component
from matplotlib import pyplot as plt
from scipy.stats import linregress

from ...data.container import ChargesContainer, GainContainer, SPEfitContainer
from ..component import ChargesComponent

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = ["PhotoStatisticAlgorithm"]


class PhotoStatisticAlgorithm(Component):
    def __init__(
        self,
        pixels_id: np.ndarray,
        FFcharge_hg: np.ndarray,
        FFcharge_lg: np.ndarray,
        Pedcharge_hg: np.ndarray,
        Pedcharge_lg: np.ndarray,
        coefCharge_FF_Ped: float,
        SPE_resolution: np.ndarray,
        SPE_high_gain: np.ndarray,
        config=None,
        parent=None,
        **kwargs,
    ) -> None:
        # constructors
        super().__init__(config=config, parent=parent, **kwargs)

        self._pixels_id = pixels_id
        self.__coefCharge_FF_Ped = coefCharge_FF_Ped

        self.__SPE_resolution = SPE_resolution
        self.__SPE_high_gain = SPE_high_gain

        self.__FFcharge_hg = FFcharge_hg
        self.__FFcharge_lg = FFcharge_lg

        self.__Pedcharge_hg = Pedcharge_hg
        self.__Pedcharge_lg = Pedcharge_lg

        self.__check_shape()

        self.__results = GainContainer(
            is_valid=np.zeros((self.npixels), dtype=bool),
            high_gain=np.zeros((self.npixels, 3)),
            low_gain=np.zeros((self.npixels, 3)),
            pixels_id=self._pixels_id,
            charge=np.zeros((self.npixels, 1)),
            charge_std=np.zeros((self.npixels, 1)),
        )

    @classmethod
    def create_from_chargesContainer(
        cls,
        FFcharge: ChargesContainer,
        Pedcharge: ChargesContainer,
        SPE_result: SPEfitContainer,
        coefCharge_FF_Ped: float,
        **kwargs,
    ):
        kwargs_init = __class__.__get_charges_FF_Ped_reshaped(
            FFcharge, Pedcharge, SPE_result
        )

        kwargs.update(kwargs_init)
        return cls(coefCharge_FF_Ped=coefCharge_FF_Ped, **kwargs)

    @staticmethod
    def __get_charges_FF_Ped_reshaped(
        FFcharge: ChargesContainer,
        Pedcharge: ChargesContainer,
        SPE_result: SPEfitContainer,
    ) -> dict:
        log.info("reshape of SPE, Ped and FF data with intersection of pixel ids")
        out = {}

        FFped_intersection = np.intersect1d(Pedcharge.pixels_id, FFcharge.pixels_id)
        SPEFFPed_intersection = np.intersect1d(
            FFped_intersection, SPE_result.pixels_id[SPE_result.is_valid]
        )
        mask_SPE = np.array(
            [
                SPE_result.pixels_id[i] in SPEFFPed_intersection
                for i in range(len(SPE_result.pixels_id))
            ],
            dtype=bool,
        )
        out["SPE_resolution"] = SPE_result.resolution[mask_SPE].T
        out["SPE_high_gain"] = SPE_result.high_gain[mask_SPE].T

        out["pixels_id"] = SPEFFPed_intersection
        out["FFcharge_hg"] = ChargesComponent.select_charges_hg(
            FFcharge, SPEFFPed_intersection
        )
        out["FFcharge_lg"] = ChargesComponent.select_charges_lg(
            FFcharge, SPEFFPed_intersection
        )
        out["Pedcharge_hg"] = ChargesComponent.select_charges_hg(
            Pedcharge, SPEFFPed_intersection
        )
        out["Pedcharge_lg"] = ChargesComponent.select_charges_lg(
            Pedcharge, SPEFFPed_intersection
        )

        log.info(f"data have {len(SPEFFPed_intersection)} pixels in common")
        return out

    def __check_shape(self) -> None:
        """Checks the shape of certain attributes and raises an exception if the shape
        is not as expected."""
        try:
            self.__FFcharge_hg[0] * self.__FFcharge_lg[0] * self.__Pedcharge_hg[
                0
            ] * self.__Pedcharge_lg[0] * self.__SPE_resolution[0] * self._pixels_id
        except Exception as e:
            log.error(e, exc_info=True)
            raise e

    def run(self, pixels_id: np.ndarray = None, **kwargs) -> None:
        log.info("running photo statistic method")
        if pixels_id is None:
            pixels_id = self._pixels_id
        mask = np.array(
            [pixel_id in pixels_id for pixel_id in self._pixels_id], dtype=bool
        )
        gainHG_err = self.gainHG_err * mask
        gainLG_err = self.gainLG_err * mask
        self._results.high_gain = np.array(
            (self.gainHG * mask, gainHG_err, gainHG_err)
        ).T
        self._results.low_gain = np.array(
            (self.gainLG * mask, gainLG_err, gainLG_err)
        ).T
        self._results.is_valid = mask
        log.info("Trying to write charges")
        self._results.charge = self.meanChargeHG
        self._results.charge_std = self.sigmaChargeHG

        figpath = kwargs.get("figpath", False)
        if figpath:
            os.makedirs(figpath, exist_ok=True)
            fig = __class__.plot_correlation(
                self._results.high_gain.T[0], self.__SPE_high_gain[0]
            )
            fig.savefig(f"{figpath}/plot_correlation_Photo_Stat_SPE.pdf")
            fig.clf()
            plt.close(fig)
        return 0

    @staticmethod
    def plot_correlation(
        photoStat_gain: np.ndarray, SPE_gain: np.ndarray
    ) -> plt.Figure:
        """Plot the correlation between the photo statistic gain and the single
        photoelectron (SPE) gain.

        Args:
            photoStat_gain (np.ndarray): Array of photo statistic gain values.
            SPE_gain (np.ndarray): Array of SPE gain values.

        Returns:
            fig (plt.Figure): The figure object containing the scatter plot
            and the linear fit line.
        """
        # matplotlib.use("TkAgg")
        # Create a mask to filter the data points based on certain criteria
        mask = (photoStat_gain > 20) * (SPE_gain > 0) * (photoStat_gain < 80)

        if not (np.max(mask)):
            log.debug("mask conditions are much strict, remove the mask")
            mask = np.ones(len(mask), dtype=bool)
        # Perform a linear regression analysis on the filtered data points
        a, b, r, p_value, std_err = linregress(
            photoStat_gain[mask], SPE_gain[mask], "greater"
        )

        # Generate a range of x-values for the linear fit line
        x = np.linspace(photoStat_gain[mask].min(), photoStat_gain[mask].max(), 1000)

        # Define a lambda function for the linear fit line
        def y(x):
            return a * x + b

        with quantity_support():
            # Create a scatter plot of the filtered data points
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.scatter(photoStat_gain[mask], SPE_gain[mask], marker=".")

            # Plot the linear fit line using the x-values and the lambda function
            ax.plot(
                x,
                y(x),
                color="red",
                label=f"linear fit,\n a = {a:.2e},\n b = {b:.2e},\n r = {r:.2e},\n\
                    p_value = {p_value:.2e},\n std_err = {std_err:.2e}",
            )

            # Plot the line y = x
            ax.plot(x, x, color="black", label="y = x")

            ax.set_xlabel("Gain Photo stat (ADC)", size=15)
            ax.set_ylabel("Gain SPE fit (ADC)", size=15)
            ax.legend(fontsize=15)

        return fig

    @property
    def SPE_resolution(self) -> float:
        """Returns a deep copy of the SPE resolution.

        Returns:
            float: The SPE resolution.
        """
        return copy.deepcopy(self.__SPE_resolution)

    @property
    def sigmaPedHG(self) -> float:
        """Calculates and returns the standard deviation of Pedcharge_hg multiplied by
        the square root of coefCharge_FF_Ped.

        Returns:
            float: The standard deviation of Pedcharge_hg.
        """
        return np.std(self.__Pedcharge_hg, axis=0) * np.sqrt(self.__coefCharge_FF_Ped)

    @property
    def sigmaChargeHG(self) -> float:
        """Calculates and returns the standard deviation of FFcharge_hg minus meanPedHG.

        Returns:
            float: The standard deviation of FFcharge_hg minus meanPedHG.
        """
        return np.std(self.__FFcharge_hg - self.meanPedHG, axis=0)

    @property
    def meanPedHG(self) -> float:
        """Calculates and returns the mean of Pedcharge_hg multiplied by
        coefCharge_FF_Ped.

        Returns:
            float: The mean of Pedcharge_hg.
        """
        return np.mean(self.__Pedcharge_hg, axis=0) * self.__coefCharge_FF_Ped

    @property
    def meanChargeHG(self) -> float:
        """Calculates and returns the mean of FFcharge_hg minus meanPedHG.

        Returns:
            float: The mean of FFcharge_hg minus meanPedHG.
        """
        return np.mean(self.__FFcharge_hg - self.meanPedHG, axis=0)

    @property
    def BHG(self) -> float:
        """Calculates and returns the BHG value.

        Returns:
            float: The BHG value.
        """
        min_events = np.min((self.__FFcharge_hg.shape[0], self.__Pedcharge_hg.shape[0]))
        upper = (
            np.power(
                self.__FFcharge_hg.mean(axis=1)[:min_events]
                - self.__Pedcharge_hg.mean(axis=1)[:min_events]
                * self.__coefCharge_FF_Ped
                - self.meanChargeHG.mean(),
                2,
            )
        ).mean(axis=0)
        lower = np.power(self.meanChargeHG.mean(), 2)
        return np.sqrt(upper / lower)

    @property
    def gainHG(self) -> float:
        """Calculates and returns the gain for high gain charge data.

        Returns:
            float: The gain for high gain charge data.
        """
        return (
            np.power(self.sigmaChargeHG, 2)
            - np.power(self.sigmaPedHG, 2)
            - np.power(self.BHG * self.meanChargeHG, 2)
        ) / (self.meanChargeHG * (1 + np.power(self.SPE_resolution[0], 2)))

    @property
    def gainHG_err(self) -> float:
        """Calculates and returns the gain for high gain charge data.

        Returns:
            float: The gain for high gain charge data.
        """
        return np.sqrt(
            np.power(
                self.gainHG
                * (-2 * self.SPE_resolution[0])
                * np.mean(self.SPE_resolution[1:], axis=0),
                2,
            )
        )

    @property
    def gainLG_err(self) -> float:
        """Calculates and returns the gain for high gain charge data.

        Returns:
            float: The gain for high gain charge data.
        """
        return np.sqrt(
            np.power(
                self.gainLG
                * (-2 * self.SPE_resolution[0])
                * np.mean(self.SPE_resolution[1:], axis=0),
                2,
            )
        )

    @property
    def sigmaPedLG(self) -> float:
        """Calculates and returns the standard deviation of Pedcharge_lg multiplied by
        the square root of coefCharge_FF_Ped.

        Returns:
            float: The standard deviation of Pedcharge_lg.
        """
        return np.std(self.__Pedcharge_lg, axis=0) * np.sqrt(self.__coefCharge_FF_Ped)

    @property
    def sigmaChargeLG(self) -> float:
        """Calculates and returns the standard deviation of FFcharge_lg minus meanPedLG.

        Returns:
            float: The standard deviation of FFcharge_lg minus meanPedLG.
        """
        return np.std(self.__FFcharge_lg - self.meanPedLG, axis=0)

    @property
    def meanPedLG(self) -> float:
        """Calculates and returns the mean of Pedcharge_lg multiplied by
        coefCharge_FF_Ped.

        Returns:
            float: The mean of Pedcharge_lg.
        """
        return np.mean(self.__Pedcharge_lg, axis=0) * self.__coefCharge_FF_Ped

    @property
    def meanChargeLG(self) -> float:
        """Calculates and returns the mean of FFcharge_lg minus meanPedLG.

        Returns:
            float: The mean of FFcharge_lg minus meanPedLG.
        """
        return np.mean(self.__FFcharge_lg - self.meanPedLG, axis=0)

    @property
    def BLG(self) -> float:
        """Calculates and returns the BLG value.

        Returns:
            float: The BLG value.
        """
        min_events = np.min((self.__FFcharge_lg.shape[0], self.__Pedcharge_lg.shape[0]))
        upper = (
            np.power(
                self.__FFcharge_lg.mean(axis=1)[:min_events]
                - self.__Pedcharge_lg.mean(axis=1)[:min_events]
                * self.__coefCharge_FF_Ped
                - self.meanChargeLG.mean(),
                2,
            )
        ).mean(axis=0)
        lower = np.power(self.meanChargeLG.mean(), 2)
        return np.sqrt(upper / lower)

    @property
    def gainLG(self) -> float:
        """Calculates and returns the gain for low gain charge data.

        Returns:
            float: The gain for low gain charge data.
        """
        return (
            np.power(self.sigmaChargeLG, 2)
            - np.power(self.sigmaPedLG, 2)
            - np.power(self.BLG * self.meanChargeLG, 2)
        ) / (self.meanChargeLG * (1 + np.power(self.SPE_resolution[0], 2)))

    @property
    def results(self):
        return copy.deepcopy(self.__results)

    @property
    def _results(self):
        return self.__results

    @property
    def npixels(self):
        return len(self._pixels_id)
