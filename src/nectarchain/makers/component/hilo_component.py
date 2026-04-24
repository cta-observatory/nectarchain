import copy

import numpy as np
from ctapipe.containers import EventType
from ctapipe.core.traits import Path
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from ...data.container import (
    PhotostatContainer,
    SPEfitContainer,
    merge_map_ArrayDataContainer,
)
from ...makers.component import ChargesComponent, GainNectarCAMComponent
from ...utils import ComponentUtils, ContainerUtils
from ...utils.constants import GAIN_DEFAULT, GAIN_LINEAR_RANGE, HILO_DEFAULT

# logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
# log = logging.getLogger(__name__)
# log.handlers = logging.getLogger("__main__").handlers

GAIN_CONTAINER_CLASSES = [PhotostatContainer, SPEfitContainer]


class HiLoComponent(GainNectarCAMComponent):
    """
    Component that computes the HiLo ratio.
    """

    gain_file = Path(
        default_value=None,
        help="Path to h5 file with gain calibration coefficients",
        allow_none=True,
    ).tag(config=True)

    SubComponents = copy.deepcopy(GainNectarCAMComponent.SubComponents)
    SubComponents.default_value = [
        "ChargesComponent",
    ]
    SubComponents.read_only = True

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        chargesComponent_kwargs = {}
        other_kwargs = {}
        chargesComponent_configurable_traits = ComponentUtils.get_configurable_traits(
            ChargesComponent
        )
        for key in kwargs.keys():
            if key in chargesComponent_configurable_traits.keys():
                chargesComponent_kwargs[key] = kwargs[key]
            else:
                other_kwargs[key] = kwargs[key]

        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **other_kwargs
        )

        self.chargesComponent = ChargesComponent(
            subarray=subarray,
            config=config,
            parent=parent,
            *args,
            **chargesComponent_kwargs,
        )
        self._chargesContainer = None
        self.log.debug(f"{kwargs.keys()}")

        self._init_gain_container()

    def _init_gain_container(self):
        self.__gain_container = None

        if self.gain_file is None:
            raise ValueError("Need to provide a gain_file to compute HiLo ratio")

        try:
            self.__gain_container = ContainerUtils.get_container_from_hdf5(
                self.gain_file,
                GAIN_CONTAINER_CLASSES,
            )
            ContainerUtils.add_missing_pixels_to_container(
                self.__gain_container, pad_value=GAIN_DEFAULT
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize gain container from {self.gain_file}"
            ) from e

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        # For now only flat-field events, to be updated for e.g. white-target
        if event.trigger.event_type == EventType.FLATFIELD:
            self.chargesComponent(event=event, *args, **kwargs)

    def finish(self, *args, **kwargs):
        if self._chargesContainer is None:
            chargesContainers = self.chargesComponent.finish(*args, **kwargs)
            self._chargesContainer = merge_map_ArrayDataContainer(chargesContainers)

        self._compute_low_gain()

        return self.__gain_container

    def _compute_low_gain(self):
        ContainerUtils.add_missing_pixels_to_container(self._chargesContainer)
        charges_hg = self._chargesContainer["charges_hg"]
        charges_lg = self._chargesContainer["charges_lg"]
        high_gain = self.__gain_container["high_gain"][:, 0]

        # Mask the linear regime between low gain and high gain
        charges_hg_pe = charges_hg / high_gain
        mask_linearity = np.logical_and(
            charges_hg_pe > np.min(GAIN_LINEAR_RANGE),
            charges_hg_pe < np.max(GAIN_LINEAR_RANGE),
        )

        # Add failsafe to not divide by 0
        mask = np.logical_and(mask_linearity, charges_lg > 0)

        # Compute HiLo ratio (per pixel) for each event
        hilo_ratio_all_events = np.divide(
            charges_hg,
            charges_lg,
            where=mask,
            out=np.full_like(charges_hg, np.nan, dtype=float),
        )

        # Compute HiLo ratio (per pixel) averaged over all events
        hilo_ratio = np.nanmean(hilo_ratio_all_events, axis=0)

        # Set default values if all events were masked
        hilo_ratio = np.where(np.isnan(hilo_ratio), HILO_DEFAULT, hilo_ratio)

        # Fill gain container
        self.__gain_container["low_gain"][:, 0] = high_gain / hilo_ratio
