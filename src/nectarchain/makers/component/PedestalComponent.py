import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy
import numpy.ma as ma

from .core import NectarCAMComponent
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe_io_nectarcam.constants import (
    N_GAINS, HIGH_GAIN, LOW_GAIN,
)
from ctapipe.core.traits import Dict, Integer
from ...data.container import NectarCAMPedestalContainer, merge_map_ArrayDataContainer
from ...utils import ComponentUtils
from .waveformsComponent import WaveformsComponent

import numpy as np

__all__ = [
    "PedestalEstimationComponent",
]


class PedestalEstimationComponent(NectarCAMComponent):
    """
    Component that computes calibration pedestal coefficients from raw data.
    """

    ucts_tmin = Integer(
        "ucts_tmin",
        help="Minimum UCTS timestamp for events used in pedestal estimation",
        default=-np.inf,
        read_only=True,
    ).tag(config=True)

    ucts_tmax = Integer(
        "ucts_tmax",
        help="Maximum UCTS timestamp for events used in pedestal estimation",
        default=np.inf,
        read_only=True,
    ).tag(config=True)
    # PedestalFilterAlgorithm = Unicode(
    #     "PedestalWaveformStdFilter",
    #     help="The waveform filter method",
    #     read_only=True,
    # ).tag(config=True)

    # not implemented yet, placeholder
    filter_kwargs = Dict(
        default_value={},
        help="The kwargs to be pass to the waveform filter method",
    ).tag(config=True)

    # add parameters for min max time

    SubComponents = copy.deepcopy(NectarCAMComponent)
    SubComponents.default_value = [
        "WaveformsComponent",
        # f"{PedestalFilterAlgorithm.default_value}",
    ]
    SubComponents.read_only = True

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        waveformsComponent_kwargs = {}
        self._PedestalFilterAlgorithm_kwargs = {}
        other_kwargs = {}
        waveformsComponent_configurable_traits = ComponentUtils.get_configurable_traits(
            WaveformsComponent
        )
        # pedestalFilterAlgorithm_configurable_traits = ComponentUtils.get_configurable_traits(
        #     eval(self.PedestalFilterAlgorithm)
        # )

        for key in kwargs.keys():
            if key in waveformsComponent_configurable_traits.keys():
                waveformsComponent_kwargs[key] = kwargs[key]
            # elif key in waveformsComponent_configurable_traits.keys():
            #     self._SPEfitalgorithm_kwargs[key] = kwargs[key]
            else:
                other_kwargs[key] = kwargs[key]

        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )

        self.waveformsComponent = WaveformsComponent(
            subarray=subarray,
            config=config,
            parent=parent,
            *args,
            **waveformsComponent_kwargs,
        )

        # initialize containers
        self._waveformsContainers = None
        self._wfs_mask = None
        self._ped_stats = {}

    @staticmethod
    def calculate_stats(waveformsContainers, wfs_mask, statistics):
        """
        Calculate statistics for the pedestals from a waveforms container.

        Parameters
        ----------
        waveformsContainers : `~nectarchain.data.container.WaveFormsContainer`
            Waveforms container
        wfs_mask : `numpy.ndarray`
            Mask to apply to exclude outliers with shape (n_pixels,n_samples)
        statistics : `list`
            Names of the statistics (numpy attributes) to compute

        Returns
        ----------
        statistics : `dict`
            A dictionary containing 3D (n_chan,n_pixels,n_samples) arrays for each statistic
        """

        ped_stats = {}

        for stat in statistics:
            # Calculate the statistic along axis = 0, that is over events
            ped_stat_hg = getattr(np, stat)(
                ma.masked_array(waveformsContainers.wfs_hg, wfs_mask), axis=0)
            ped_stat_lg = getattr(np, stat)(
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

    def finish(self, *args, **kwargs):

        # Make sure that Waveforms container is properly filled
        is_empty = False
        if self._waveformsContainers is None:
            self._waveformsContainers = self.waveformsComponent.finish(*args, **kwargs)
            is_empty = self._waveformsContainers.is_empty()
            if is_empty:
                log.warning("empty waveforms container, pedestals cannot be evaluated")
            else:
                # container merging
                self._waveformsContainers = merge_map_ArrayDataContainer(
                    self._waveformsContainers
                )

        if not is_empty:
            # Check that there are events for the computation of the pedestals
            # Build mask to filter the waveforms
            # mask based on the high gain channel that is most sensitive to signals
            self._wfs_mask = np.zeros(np.shape(self._waveformsContainers.wfs_hg), dtype=bool)

            # apply time filter
            # log.info(
            #     f"Mask events outside the UCTS timestamp range {self.ucts_tmin}-{self.ucts_tmax}")
            # t_mask = np.where((self._waveformsContainers.ucts_timestamps < self.ucts_tmin) |
            #                   (self._waveformsContainers.ucts_timestamps > self.ucts_tmax))

            # compute statistics for the pedestals
            # the statistic names must be valid numpy attributes
            statistics = ['mean', 'median', 'std']
            self._ped_stats = self.calculate_stats(self._waveformsContainers, self._wfs_mask,
                                                   statistics)

            # Fill and return output container

            # metadata = {} # store information about filtering method and params

            output = NectarCAMPedestalContainer(
                nsamples=self._waveformsContainers.nsamples,
                nevents=self._waveformsContainers.nevents,
                pixels_id=self._waveformsContainers.pixels_id,
                ucts_timestamp_min=self._waveformsContainers.ucts_timestamp.min(),
                ucts_timestamp_max=self._waveformsContainers.ucts_timestamp.max(),
                pedestal_mean_hg=self._ped_stats['mean'][HIGH_GAIN],
                pedestal_mean_lg=self._ped_stats['mean'][LOW_GAIN],
                pedestal_median_hg=self._ped_stats['median'][HIGH_GAIN],
                pedestal_median_lg=self._ped_stats['median'][LOW_GAIN],
                pedestal_std_hg=self._ped_stats['std'][HIGH_GAIN],
                pedestal_std_lg=self._ped_stats['std'][LOW_GAIN],
            )

        return output
