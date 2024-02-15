import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy
import numpy.ma as ma

from .core import NectarCAMComponent
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe_io_nectarcam.constants import N_GAINS, HIGH_GAIN, LOW_GAIN
from ctapipe.core.traits import Integer, Unicode, Float
from ...data.container import NectarCAMPedestalContainer, merge_map_ArrayDataContainer
from ...utils import ComponentUtils
from .waveformsComponent import WaveformsComponent

import numpy as np

__all__ = ["PedestalEstimationComponent", ]


class PedestalEstimationComponent(NectarCAMComponent):
    """
    Component that computes calibration pedestal coefficients from raw data.
    """

    ucts_tmin = Integer(None,
                        help="Minimum UCTS timestamp for events used in pedestal estimation",
                        read_only=False,
                        allow_none=True, ).tag(config=True)

    ucts_tmax = Integer(None,
                        help="Maximum UCTS timestamp for events used in pedestal estimation",
                        read_only=False,
                        allow_none=True, ).tag(config=True)

    filter_method = Unicode(
        None,
        help="The waveforms filter method to be used.\n"
             "Inplemented methods: WaveformsStdFilter (standard deviation of waveforms),\n"
             "                     ChargeDistributionFilter (charge distribution).",
        read_only=False,
        allow_none=True,
    ).tag(config=True)

    wfs_std_threshold = Float(
        4,
        help="Threshold of waveforms standard deviation in ADC counts above which a waveform is excluded from pedestal computation.",
        read_only=False,
    )

    SubComponents = copy.deepcopy(NectarCAMComponent)
    SubComponents.default_value = ["WaveformsComponent",
                                   # f"{PedestalFilterAlgorithm.default_value}",
                                   ]
    SubComponents.read_only = True

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):

        waveformsComponent_kwargs = {}
        other_kwargs = {}
        waveformsComponent_configurable_traits = ComponentUtils.get_configurable_traits(
            WaveformsComponent)

        for key in kwargs.keys():
            if key in waveformsComponent_configurable_traits.keys():
                waveformsComponent_kwargs[key] = kwargs[key]
            else:
                other_kwargs[key] = kwargs[key]

        super().__init__(subarray=subarray, config=config, parent=parent, *args, **kwargs)

        self.waveformsComponent = WaveformsComponent(subarray=subarray, config=config,
                                                     parent=parent, *args,
                                                     **waveformsComponent_kwargs, )

        # initialize members
        self._waveformsContainers = None
        self._wfs_mask = None
        self._ped_stats = {}

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
            A dictionary containing 3D (n_chan,n_pixels,n_samples) arrays for each statistic
        """

        ped_stats = {}

        for stat in statistics:
            # Calculate the statistic along axis = 0, that is over events
            ped_stat_hg = getattr(ma, stat)(
                ma.masked_array(waveformsContainers.wfs_hg, wfs_mask), axis=0)
            ped_stat_lg = getattr(ma, stat)(
                ma.masked_array(waveformsContainers.wfs_lg, wfs_mask), axis=0)

            # Create a 3D array for the statistic
            array_shape = np.append([N_GAINS], np.shape(waveformsContainers.wfs_hg[0]))
            ped_stat = np.zeros(array_shape)
            ped_stat[HIGH_GAIN] = ped_stat_hg
            ped_stat[LOW_GAIN] = ped_stat_lg

            # Store the result in the dictionary
            ped_stats[stat] = ped_stat

        return ped_stats

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):

        # fill the waveform container looping over the events
        self.waveformsComponent(event=event, *args, **kwargs)

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
            A boolean array of shape (n_events,n_pixels,n_samples) that identifies waveforms to be masked
        """
        # Check if time filtering affects the data
        if tmin > self._waveformsContainers.ucts_timestamp.min() or tmax < self._waveformsContainers.ucts_timestamp.max():
            log.info(
                f"Apply time interval selection: UCTS timestamps in interval {tmin}-{tmax}")
            # Define the mask on UCTS timestamps
            t_mask = ((self._waveformsContainers.ucts_timestamp < tmin) | (
                    self._waveformsContainers.ucts_timestamp > tmax))
            # Log information
            nonzero = np.count_nonzero(t_mask)
            log.info(f"{len(t_mask) - nonzero}/{len(t_mask)} waveforms pass time selection.")
            # Create waveforms mask to apply time selection
            new_mask = np.logical_or(self._wfs_mask, t_mask[:, np.newaxis, np.newaxis])
        else:
            log.info(
                f"The entire time interval will be used: UCTS timestamps in {tmin}-{tmax}")
            new_mask = self._wfs_mask

        # Return waveforms mask
        return new_mask

    def waveformsStdFilter_mask(self, threshold):
        """
        Generates a mask to filter waveforms that have a standard deviation above a threshold.
        This option is effective for dark room verification data.

        Parameters
        ----------
        threshold : float
            Waveform standard deviation (in ADC counts) above which the waveform is filtered out

        Returns
        ----------
        mask : `~numpy.ndarray`
            A boolean array of shape (n_events,n_pixels,n_samples) that identifies waveforms to be masked
        """

        # Log
        log.info(f"apply {self.filter_method} method with threshold = {threshold} ADC counts")

        # For each event and pixel calculate the waveform std
        wfs_std = np.std(self._waveformsContainers.wfs_hg, axis=2)

        # Mask events/pixels that exceed the threshold
        std_mask = (wfs_std > threshold)
        # Log information
        # number of masked waveforms
        nonzero = np.count_nonzero(std_mask)
        # number of total waveforms
        tot_wfs = self._waveformsContainers.nevents * self._waveformsContainers.npixels
        # fraction of waveforms masked (%)
        frac = 100 * nonzero / tot_wfs
        log.info(f"{frac:.2f}% of the waveforms filtered out based on standard deviation.")

        # Create waveforms mask to apply time selection
        new_mask = np.logical_or(self._wfs_mask, std_mask[:, :, np.newaxis])

        return new_mask

    def finish(self, *args, **kwargs):

        # Make sure that waveforms container is properly filled
        is_empty = False
        if self._waveformsContainers is None:
            self._waveformsContainers = self.waveformsComponent.finish(*args, **kwargs)
            is_empty = self._waveformsContainers.is_empty()
            if is_empty:
                log.warning("empty waveforms container, pedestals cannot be evaluated")
                # container with no results
                return None
            else:
                # container merging
                self._waveformsContainers = merge_map_ArrayDataContainer(
                    self._waveformsContainers)

        if not is_empty:
            # Build mask to filter the waveforms
            # Mask based on the high gain channel that is most sensitive to signals
            # Initialize empty mask
            self._wfs_mask = np.zeros(np.shape(self._waveformsContainers.wfs_hg),
                                      dtype=bool)

            # Time selection
            # set the minimum time
            print('time filter')
            tmin = np.maximum(
                self.ucts_tmin or self._waveformsContainers.ucts_timestamp.min(),
                self._waveformsContainers.ucts_timestamp.min())
            # set the maximum time
            tmax = np.minimum(
                self.ucts_tmax or self._waveformsContainers.ucts_timestamp.max(),
                self._waveformsContainers.ucts_timestamp.max())
            # Add time selection to mask
            self._wfs_mask = self.timestamp_mask(tmin, tmax)

            # Filter Waveforms
            if self.filter_method is None:
                log.info('no filtering applied to waveforms')
            elif self.filter_method == 'WaveformsStdFilter':
                self._wfs_mask = self.waveformsStdFilter_mask(self.wfs_std_threshold)
            elif self.filter_method == 'ChargeDistributionFilter':
                log.info(f"method {self.filter_method} not implemented yet")
            else:
                log.warning(
                    f"required filtering method {self.filter_method} not available")
                log.warning("no filtering applied to waveforms")

            # compute statistics for the pedestals
            # the statistic names must be valid numpy.ma attributes
            statistics = ['mean', 'median', 'std']
            self._ped_stats = self.calculate_stats(self._waveformsContainers,
                                                   self._wfs_mask,
                                                   statistics)

            # Fill and return output container
            # metadata = {} # store information about filtering method and params
            output = NectarCAMPedestalContainer(
                nsamples=self._waveformsContainers.nsamples,
                nevents=self._waveformsContainers.nevents,
                pixels_id=self._waveformsContainers.pixels_id,
                ucts_timestamp_min=np.uint64(tmin),
                ucts_timestamp_max=np.uint64(tmax),
                pedestal_mean_hg=self._ped_stats['mean'][HIGH_GAIN],
                pedestal_mean_lg=self._ped_stats['mean'][LOW_GAIN],
                pedestal_median_hg=self._ped_stats['median'][HIGH_GAIN],
                pedestal_median_lg=self._ped_stats['median'][LOW_GAIN],
                pedestal_std_hg=self._ped_stats['std'][HIGH_GAIN],
                pedestal_std_lg=self._ped_stats['std'][LOW_GAIN], )

            return output
