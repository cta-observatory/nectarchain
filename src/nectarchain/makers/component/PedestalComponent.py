import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy
import astropy.units as u
import numpy.ma as ma

from .core import NectarCAMComponent
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from ctapipe_io_nectarcam.constants import (
    N_GAINS, N_PIXELS, N_SAMPLES,
    HIGH_GAIN, LOW_GAIN,
)
from ctapipe_io_nectarcam import time_from_unix_tai_ns
from ctapipe.core.traits import Dict, Unicode
from ctapipe.containers import PedestalContainer
from ...data.container import merge_map_ArrayDataContainer
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
    # PedestalFilterAlgorithm = Unicode(
    #     "PedestalWaveformStdFilter",
    #     help="The waveform filter method",
    #     read_only=True,
    # ).tag(config=True)

    #not implemented yet, placeholder
    filter_kwargs = Dict(
        default_value={},
        help="The kwargs to be pass to the waveform filter method",
    ).tag(config=True)

    # add parameters for min max time

    SubComponents = copy.deepcopy(NectarCAMComponent)
    SubComponents.default_value = [
        "WaveformsComponent",
        #f"{PedestalFilterAlgorithm.default_value}",
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
        self._waveformsContainers = None

        self._wfs_mask = None

    @staticmethod
    def calculate_stats(waveformsContainers,wfs_mask,statistics):
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
                ma.masked_array(waveformsContainers.wfs_hg,wfs_mask),axis=0)
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

        self.waveformsComponent(event=event, *args, **kwargs)

        is_empty = False
        if self._waveformsContainers is None:
            self._waveformsContainers = self.waveformsComponent.finish(*args, **kwargs)
            is_empty = self._waveformsContainers.is_empty()
            if is_empty:
                log.warning("empty waveformsContainer in output")
            else:
                self._waveformsContainers = merge_map_ArrayDataContainer(
                    self._waveformsContainers
                )

        if not (is_empty):
            # change this into something that creates a real mask
            # one mask for both HG and LG or separate? mask only on hg which is more sensitive
            self._wfs_mask = np.zeros(np.shape(self._waveformsContainers.wfs_hg),dtype=bool)
        else:
            pass

    def finish(self, *args, **kwargs):

        # compute statistics for the pedestals
        # the statistic names must be valid numpy attributes
        statistics = ['mean', 'median', 'std']
        #print(np.shape(self._waveformsContainers.wfs_hg))
        ped_stats = self.calculate_stats(self._waveformsContainers,self._wfs_mask,statistics)

        metadata = {} # store information about filtering method and params

        output = PedestalContainer(
            n_events = self._waveformsContainers.nevents,
            sample_time= np.nan, # TODO : does this need to be filled?
            sample_time_min= time_from_unix_tai_ns(self._waveformsContainers.ucts_timestamp.min()).value * u.s,
            sample_time_max = time_from_unix_tai_ns(self._waveformsContainers.ucts_timestamp.max()).value * u.s,
            charge_mean = ped_stats['mean'],
            charge_median = ped_stats['median'],
            charge_std = ped_stats['std'],
            meta = metadata,
        )

        return output
