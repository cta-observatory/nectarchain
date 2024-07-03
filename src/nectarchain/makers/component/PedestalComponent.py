import copy
import logging

import numpy as np
import numpy.ma as ma
from ctapipe.containers import EventType
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from ctapipe_io_nectarcam.constants import N_GAINS, HIGH_GAIN, LOW_GAIN
from ctapipe.core.traits import Integer, Unicode, Float, Dict
from ...data.container import NectarCAMPedestalContainer, PedestalFlagBits
from ...utils import ComponentUtils
from .chargesComponent import ChargesComponent
from .core import NectarCAMComponent
from .waveformsComponent import WaveformsComponent

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = [
    "PedestalEstimationComponent",
]


class PedestalEstimationComponent(NectarCAMComponent):
    """
    Component that computes calibration pedestal coefficients from raw data.
    Waveforms can be filtered based on time, stanndard deviation of the waveforms
    or charge distribution within the sample.
    Use the `events_per_slice' parameter of `NectarCAMComponent' to reduce memory load.

    Parameters
    ----------
    ucts_tmin : int
        Minimum UCTS timestamp for events used in pedestal estimation
    ucts_tmax : int
        Maximum UCTS timestamp for events used in pedestal estimation
    filter_method : str
        The waveforms filter method to be used.
        Inplemented methods: WaveformsStdFilter (standard deviation of waveforms),
        ChargeDistributionFilter (charge distribution).
    wfs_std_threshold : float
        Threshold of waveforms standard deviation in ADC counts above which a
        waveform is excluded from pedestal computation.
    charge_sigma_high_thr : float
        Threshold in charge distribution (number of sigmas above mean) beyond which a
        waveform is excluded from pedestal computation.
    charge_sigma_low_thr : float
        Threshold in charge distribution (number of sigmas below mean) beyond which a waveform
        is excluded from pedestal computation.
    pixel_mask_nevents_min : int
        Minimum number of events below which the pixel is flagged as bad
    pixel_mask_mean_min : float
        Minimum value of pedestal mean below which the pixel is flagged as bad
    pixel_mask_mean_max : float
        Maximum value of pedestal mean above which the pixel is flagged as bad
    pixel_mask_std_sample_min : float
        Minimum value of pedestal standard deviation for all samples below which the pixel is flagged as bad
    pixel_mask_std_pixel_max : float
        Maximum value of pedestal standard deviation in a pixel above which the pixel is flagged as bad
    """

    ucts_tmin = Integer(
        None,
        help="Minimum UCTS timestamp for events used in pedestal estimation",
        allow_none=True,
    ).tag(config=True)

    ucts_tmax = Integer(
        None,
        help="Maximum UCTS timestamp for events used in pedestal estimation",
        allow_none=True,
    ).tag(config=True)

    filter_method = Unicode(
        None,
        help="The waveforms filter method to be used.\n"
        "Inplemented methods: WaveformsStdFilter (standard deviation of waveforms),\n"
        "                     ChargeDistributionFilter (charge distribution).",
        read_only=False,
        allow_none=True,
    ).tag(config=True)

    wfs_std_threshold = Float(
        4.0,
        help="Threshold of waveforms standard deviation in ADC counts above which a "
        "waveform is excluded from pedestal computation.",
    ).tag(config=True)

    charge_sigma_high_thr = Float(
        3.0,
        help="Threshold in charge distribution (number of sigmas above mean) beyond "
        "which a waveform is excluded from pedestal computation.",
    ).tag(config=True)

    charge_sigma_low_thr = Float(
        3.0,
        help="Threshold in charge distribution (number of sigmas below mean) beyond "
        "which a waveform is excluded from pedestal computation.",
    ).tag(config=True)

    pixel_mask_nevents_min = Integer(
        100,
        help="Minimum number of events below which the pixel is flagged as bad",
    ).tag(config=True)

    pixel_mask_mean_min = Float(
        200.,
        help="Minimum value of pedestal mean below which the pixel is flagged as bad",
    ).tag(config=True)

    pixel_mask_mean_max = Float(
        300.,
        help="Maximum value of pedestal mean above which the pixel is flagged as bad",
    ).tag(config=True)

    pixel_mask_std_sample_min = Float(
        0.5,
        # for run 3938 in dark room typical standard deviations in working pixel HG are 2.7, LT 2024 July 3
        help="Minimum value of pedestal standard deviation for all samples below which the pixel is flagged as bad",
    ).tag(config=True)

    pixel_mask_std_pixel_max = Float(
        4,
        # for run 3938 in dark room typical standard deviations in working pixel HG are 0.25, LT 2024 July 3
        help="Maximum value of pedestal standard deviation in a pixel above which the pixel is flagged as bad",
    ).tag(config=True)

    # I do not understand why but the ChargesComponents traits are not loaded
    # FIXME
    method = Unicode(
        default_value="FullWaveformSum",
        help="the charge extraction method",
    ).tag(config=True)

    extractor_kwargs = Dict(
        default_value={},
        help="The kwargs to be pass to the charge extractor method",
    ).tag(config=True)

    SubComponents = copy.deepcopy(NectarCAMComponent.SubComponents)
    SubComponents.default_value = ["WaveformsComponent", "ChargesComponent"]
    SubComponents.read_only = True

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        """
        Component that computes calibration pedestal coefficients from raw data.
        Waveforms can be filtered based on time, standard deviation of the waveforms
        or charge distribution within the sample.
        Use the `events_per_slice' parameter of `NectarCAMComponent' to
        reduce memory load.

        Parameters
        ----------
        ucts_tmin : int
            Minimum UCTS timestamp for events used in pedestal estimation
        ucts_tmax : int
            Maximum UCTS timestamp for events used in pedestal estimation
        filter_method : str
            The waveforms filter method to be used.
            Inplemented methods: WaveformsStdFilter (standard deviation of waveforms),
            ChargeDistributionFilter (charge distribution).
        wfs_std_threshold : float
            Threshold of waveforms standard deviation in ADC counts above which a
            waveform is excluded from pedestal computation.
        charge_sigma_high_thr : float
            Threshold in charge distribution (number of sigmas above mean) beyond which
            a waveform is excluded from pedestal computation.
        charge_sigma_low_thr : float
            Threshold in charge distribution (number of sigmas below mean) beyond which
            a waveform is excluded from pedestal computation.
        """

        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )

        # initialize members
        self._waveformsContainers = None
        self._chargesContainers = None
        self._wfs_mask = None
        self._ped_stats = {}

        # initialize waveforms component
        waveformsComponent_kwargs = {}
        waveformsComponent_configurable_traits = ComponentUtils.get_configurable_traits(
            WaveformsComponent
        )
        for key in kwargs.keys():
            if key in waveformsComponent_configurable_traits.keys():
                waveformsComponent_kwargs[key] = kwargs[key]
        self.waveformsComponent = WaveformsComponent(
            subarray=subarray,
            config=config,
            parent=parent,
            *args,
            **waveformsComponent_kwargs,
        )

    @staticmethod
    def calculate_stats(waveformsContainers, wfs_mask, statistics):
        """
        Calculate statistics for the pedestals from a waveforms container.

        Parameters
        ----------
        waveformsContainers : `~nectarchain.data.container.WaveformsContainer`
            Waveforms container
        wfs_mask : `numpy.ndarray`
            Mask to apply to exclude outliers with shape (n_pixels,n_samples)
        statistics : `list`
            Names of the statistics (numpy.ma attributes) to compute

        Returns
        ----------
        ped_stats : `dict`
            A dictionary containing 3D (n_chan,n_pixels,n_samples) arrays for each
            statistic.
        """

        ped_stats = {}

        for stat in statistics:
            # Calculate the statistic along axis = 0, that is over events
            ped_stat_hg = getattr(ma, stat)(
                ma.masked_array(waveformsContainers.wfs_hg, wfs_mask), axis=0
            )
            ped_stat_lg = getattr(ma, stat)(
                ma.masked_array(waveformsContainers.wfs_lg, wfs_mask), axis=0
            )

            # Create a 3D array for the statistic
            array_shape = np.append([N_GAINS], np.shape(waveformsContainers.wfs_hg[0]))
            ped_stat = np.zeros(array_shape)
            ped_stat[HIGH_GAIN] = ped_stat_hg
            ped_stat[LOW_GAIN] = ped_stat_lg

            # Store the result in the dictionary
            ped_stats[stat] = ped_stat

        return ped_stats

    def flag_bad_pixels(self, ped_stats, nevents):
        """
        Flag bad pixels based on pedestal properties

        Parameters
        ----------
        ped_stats : `dict`
            A dictionary containing 3D (n_chan,n_pixels,n_samples) arrays for each statistic
        nevents : `np.ndarray`
            An array that contains the numbber of events used to calculate the statistics
            for each pixel

        Returns
        pixel_mask : `np.ndarray`
            An array that contains the binary mask of shape (nchannels,npixels)
            to flag bad pixels
        ----------
        """

        # Initialize mask to False (all pixels are good)
        pixel_mask = np.int8(np.zeros(np.shape(ped_stats['mean'])[:2]))

        # Flag on number of events
        flag_nevents = np.int8(nevents < self.pixel_mask_nevents_min)
        # Bitwise OR
        pixel_mask = pixel_mask | flag_nevents[np.newaxis, :] * PedestalFlagBits.NEVENTS

        # Flag on mean pedestal value
        # Average over all samples for each channel/pixel
        ped_mean = np.mean(ped_stats['mean'], axis=2)
        # Apply thresholds
        flag_mean = np.int8(np.logical_or(ped_mean < self.pixel_mask_mean_min,
                                          ped_mean > self.pixel_mask_mean_max))
        # Bitwise OR
        pixel_mask = pixel_mask | flag_mean * PedestalFlagBits.MEAN_PEDESTAL

        # Flag on standard deviation per sample
        # all samples in channel/pixel below threshold
        flag_sample_std = np.int8(np.all(ped_stats['std'] < self.pixel_mask_std_sample_min,
                                         axis=2))
        # Bitwise OR
        pixel_mask = pixel_mask | flag_sample_std * PedestalFlagBits.STD_SAMPLE

        # Flag on standard deviation per pixel
        # Standard deviation of pedestal in channel/pixel above threshold
        flag_pixel_std = np.int8(
            np.std(ped_stats['mean'], axis=2) > self.pixel_mask_std_pixel_max)
        # Bitwise OR
        pixel_mask = pixel_mask | flag_pixel_std * PedestalFlagBits.STD_PIXEL

        # Return mask
        return pixel_mask

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        """
        Fill the waveform container looping over the events of type SKY_PEDESTAL.
        """

        if event.trigger.event_type == EventType.SKY_PEDESTAL:
            self.waveformsComponent(event=event, *args, **kwargs)
        else:
            pass

    def timestamp_mask(self, tmin, tmax):
        """
        Generates a mask to filter waveforms outside the required time interval

        Parameters
        ----------
        tmin : int
            Minimum time of the required interval
        tmax : int
            Maximum time of the required interval

        Returns
        ----------
        mask : `~numpy.ndarray`
            A boolean array of shape (n_events,n_pixels,n_samples) that identifies
            waveforms to be masked.
        """
        # Check if time filtering affects the data
        if (
            tmin > self._waveformsContainers.ucts_timestamp.min()
            or tmax < self._waveformsContainers.ucts_timestamp.max()
        ):
            log.info(
                f"Apply time interval selection: UCTS timestamps in "
                f"interval {tmin}-{tmax}"
            )
            # Define the mask on UCTS timestamps
            t_mask = (self._waveformsContainers.ucts_timestamp < tmin) | (
                self._waveformsContainers.ucts_timestamp > tmax
            )
            # Log information
            nonzero = np.count_nonzero(t_mask)
            log.info(
                f"{len(t_mask) - nonzero}/{len(t_mask)} waveforms pass time selection."
            )
            # Create waveforms mask to apply time selection
            new_mask = np.logical_or(self._wfs_mask, t_mask[:, np.newaxis, np.newaxis])
        else:
            log.info(
                f"The entire time interval will be used: UCTS timestamps "
                f"in {tmin}-{tmax}"
            )
            new_mask = self._wfs_mask

        # Return waveforms mask
        return new_mask

    def waveformsStdFilter_mask(self, threshold):
        """
        Generates a mask to filter waveforms that have a standard deviation above
        a threshold.
        This option is effective for dark room verification data.

        Parameters
        ----------
        threshold : float
            Waveform standard deviation (in ADC counts) above which the waveform is
            filtered out

        Returns
        ----------
        mask : `~numpy.ndarray`
            A boolean array of shape (n_events,n_pixels,n_samples) that identifies
            waveforms to be masked
        """

        # Log
        log.info(
            f"apply {self.filter_method} method with threshold = {threshold} ADC counts"
        )

        # For each event and pixel calculate the waveform std
        wfs_std = np.std(self._waveformsContainers.wfs_hg, axis=2)

        # Mask events/pixels that exceed the threshold
        std_mask = wfs_std > threshold
        # Log information
        # number of masked waveforms
        nonzero = np.count_nonzero(std_mask)
        # number of total waveforms
        tot_wfs = self._waveformsContainers.nevents * self._waveformsContainers.npixels
        # fraction of waveforms masked (%)
        frac = 100 * nonzero / tot_wfs
        log.info(
            f"{frac:.2f}% of the waveforms filtered out based on standard deviation."
        )

        # Create waveforms mask to apply time selection
        new_mask = np.logical_or(self._wfs_mask, std_mask[:, :, np.newaxis])

        return new_mask

    def chargeDistributionFilter_mask(self, sigma_low, sigma_high):
        """
        Generates a mask to filter waveforms that have a charge in the tails of the
        distribution.
        This option is useful for data with NSB.

        Parameters
        ----------
        sigma_low : float
            Number of standard deviation below mean charge beyond which waveforms are
            filtered out
        sigma_high : float
            Number of standard deviation above mean charge beyond which waveforms are
            filtered out

        Returns
        ----------
        mask : `~numpy.ndarray`
            A boolean array of shape (n_events,n_pixels,n_samples) that identifies
            waveforms to be masked
        """

        # Log
        log.info(
            f"apply {self.filter_method} method to filter waveforms with charge "
            f"outside the interval -{sigma_low}-{sigma_high} standard deviations "
            f"around the mean value"
        )

        # Check if waveforms and charges have the same shape
        wfs_shape = np.shape(self._waveformsContainers.wfs_hg)
        charges_shape = np.shape(self._chargesContainers.charges_hg)
        if wfs_shape[0] == charges_shape[0] and wfs_shape[1] == charges_shape[1]:
            # For each event and pixel calculate the charge mean and std over all events
            # taking into account the mask already calculated
            charge_array = ma.masked_array(
                self._chargesContainers.charges_hg, np.any(self._wfs_mask, axis=2)
            )
            charge_mean = ma.mean(charge_array, axis=0)
            charge_std = ma.std(charge_array, axis=0)

            # Mask events/pixels that are outside the core of the distribution
            low_threshold = charge_mean - sigma_low * charge_std
            low_mask = self._chargesContainers.charges_hg < low_threshold[np.newaxis, :]
            high_threshold = charge_mean + sigma_high * charge_std
            high_mask = (
                self._chargesContainers.charges_hg > high_threshold[np.newaxis, :]
            )
            charge_mask = np.logical_or(low_mask, high_mask)
            # Log information
            # number of masked waveforms
            nonzero = np.count_nonzero(charge_mask)
            # number of total waveforms
            tot_wfs = (
                self._waveformsContainers.nevents * self._waveformsContainers.npixels
            )
            # fraction of waveforms masked (%)
            frac = 100 * nonzero / tot_wfs
            log.info(
                f"{frac:.2f}% of the waveforms filtered out based on charge "
                f"distribution."
            )

            # Create waveforms mask to apply time selection
            new_mask = np.logical_or(self._wfs_mask, charge_mask[:, :, np.newaxis])
        else:
            log.warning(
                "Waveforms and charges have incompatible shapes. No filtering applied."
            )
            new_mask = self._wfs_mask

        return new_mask

    def finish(self, *args, **kwargs):
        """
        Finish the component by filtering the waveforms and calculating the pedestal
        quantities.
        """

        # Use only pedestal type events
        waveformsContainers = self.waveformsComponent.finish()
        self._waveformsContainers = waveformsContainers.containers[
            EventType.SKY_PEDESTAL
        ]
        # log.info('JPL: waveformsContainers=',waveformsContainers.containers[
        # EventType.SKY_PEDESTAL].nsamples)

        # If we want to filter based on charges distribution
        # make sure that the charge distribution container is filled
        if (
            self.filter_method == "ChargeDistributionFilter"
            and self._chargesContainers is None
        ):
            log.debug("Compute charges from waveforms")
            chargesComponent_kwargs = {}
            chargesComponent_configurable_traits = (
                ComponentUtils.get_configurable_traits(ChargesComponent)
            )
            for key in kwargs.keys():
                if key in chargesComponent_configurable_traits.keys():
                    chargesComponent_kwargs[key] = kwargs[key]
            self._chargesContainers = ChargesComponent.create_from_waveforms(
                waveformsContainer=self._waveformsContainers,
                subarray=self.subarray,
                config=self.config,
                parent=self.parent,
                *args,
                **chargesComponent_kwargs,
            )

        # Check if waveforms container is empty
        if self._waveformsContainers is None:
            log.warning("Waveforms container is none, pedestals cannot be evaluated")
            # container with no results
            return None
        elif (
            self._waveformsContainers.nevents is None
            or self._waveformsContainers.nevents == 0
        ):
            log.warning("Waveforms container is empty, pedestals cannot be evaluated")
            # container with no results
            return None
        else:
            # Build mask to filter the waveforms
            # Mask based on the high gain channel that is most sensitive to signals
            # Initialize empty mask
            self._wfs_mask = np.zeros(
                np.shape(self._waveformsContainers.wfs_hg), dtype=bool
            )

            # Time selection
            # set the minimum time
            tmin = np.maximum(
                self.ucts_tmin or self._waveformsContainers.ucts_timestamp.min(),
                self._waveformsContainers.ucts_timestamp.min(),
            )
            # set the maximum time
            tmax = np.minimum(
                self.ucts_tmax or self._waveformsContainers.ucts_timestamp.max(),
                self._waveformsContainers.ucts_timestamp.max(),
            )
            # Add time selection to mask
            self._wfs_mask = self.timestamp_mask(tmin, tmax)

            # Filter Waveforms
            if self.filter_method is None:
                log.info("no filtering applied to waveforms")
            elif self.filter_method == "WaveformsStdFilter":
                self._wfs_mask = self.waveformsStdFilter_mask(self.wfs_std_threshold)
            elif self.filter_method == "ChargeDistributionFilter":
                self._wfs_mask = self.chargeDistributionFilter_mask(
                    self.charge_sigma_low_thr, self.charge_sigma_high_thr
                )
            else:
                log.warning(
                    f"required filtering method {self.filter_method} not available"
                )
                log.warning("no filtering applied to waveforms")

            # compute statistics for the pedestals
            # the statistic names must be valid numpy.ma attributes
            statistics = ["mean", "std"]
            self._ped_stats = self.calculate_stats(
                self._waveformsContainers, self._wfs_mask, statistics
            )

            # calculate the number of events per pixel used to compute the quantitites
            # start wit total number of events
            nevents = np.ones(len(self._waveformsContainers.pixels_id))
            nevents *= self._waveformsContainers.nevents
            # subtract masked events
            # use the first sample for each event/pixel
            # assumes that a waveform is either fully masked or not
            nevents -= np.sum(self._wfs_mask[:, :, 0], axis=0)

            # calculate on the flight which pixels should be flagged as bad
            pixel_mask = self.flag_bad_pixels(self._ped_stats, nevents)

            # Fill and return output container
            output = NectarCAMPedestalContainer(
                nsamples=self._waveformsContainers.nsamples,
                nevents=nevents,
                pixels_id=self._waveformsContainers.pixels_id,
                ucts_timestamp_min=np.uint64(tmin),
                ucts_timestamp_max=np.uint64(tmax),
                pedestal_mean_hg=self._ped_stats['mean'][HIGH_GAIN],
                pedestal_mean_lg=self._ped_stats['mean'][LOW_GAIN],
                pedestal_std_hg=self._ped_stats['std'][HIGH_GAIN],
                pedestal_std_lg=self._ped_stats['std'][LOW_GAIN],
                pixel_mask=pixel_mask,
            )

            return output
