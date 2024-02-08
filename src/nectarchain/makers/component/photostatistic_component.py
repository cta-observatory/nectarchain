import logging

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

import copy

from ctapipe.containers import EventType
from ctapipe.core.traits import List, Path, Unicode
from ctapipe_io_nectarcam.constants import N_SAMPLES
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from ...data.container import SPEfitContainer, merge_map_ArrayDataContainer
from ...utils import ComponentUtils
from .chargesComponent import ChargesComponent
from .gainComponent import GainNectarCAMComponent
from .photostatistic_algorithm import PhotoStatisticAlgorithm

__all__ = ["PhotoStatisticNectarCAMComponent"]


class PhotoStatisticNectarCAMComponent(GainNectarCAMComponent):
    SPE_result = Path(
        help="the path of the SPE result container computed with very high voltage data",
    ).tag(config=True)
    PhotoStatAlgorithm = Unicode(
        "PhotoStatisticAlgorithm",
        help="The photo-statitic algorithm to be used",
        read_only=True,
    ).tag(config=True)

    SubComponents = copy.deepcopy(GainNectarCAMComponent.SubComponents)
    SubComponents.default_value = [
        "ChargesComponent",
        f"{PhotoStatAlgorithm.default_value}",
    ]
    SubComponents.read_only = True

    asked_pixels_id = List(
        default_value=None,
        allow_none=True,
        help="The pixels id where we want to perform the SPE fit",
    ).tag(config=True)

    # constructor
    def __init__(self, subarray, config=None, parent=None, *args, **kwargs) -> None:
        chargesComponent_kwargs = {}
        self._PhotoStatAlgorithm_kwargs = {}
        other_kwargs = {}
        chargesComponent_configurable_traits = ComponentUtils.get_configurable_traits(
            ChargesComponent
        )
        PhotoStatAlgorithm_configurable_traits = ComponentUtils.get_configurable_traits(
            eval(self.PhotoStatAlgorithm)
        )

        for key in kwargs.keys():
            if key in chargesComponent_configurable_traits.keys():
                chargesComponent_kwargs[key] = kwargs[key]
            elif key in PhotoStatAlgorithm_configurable_traits.keys():
                self._PhotoStatAlgorithm_kwargs[key] = kwargs[key]
            else:
                other_kwargs[key] = kwargs[key]

        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **other_kwargs
        )
        self.FF_chargesComponent = ChargesComponent(
            subarray=subarray,
            config=config,
            parent=parent,
            *args,
            **chargesComponent_kwargs,
        )
        self._FF_chargesContainers = None
        self.Ped_chargesComponent = ChargesComponent(
            subarray=subarray,
            config=config,
            parent=parent,
            *args,
            **chargesComponent_kwargs,
        )
        self._Ped_chargesContainers = None

        self.__coefCharge_FF_Ped = (
            int(
                chargesComponent_kwargs.get("extractor_kwargs").get(
                    "window_width", N_SAMPLES
                )
            )
            / N_SAMPLES
        )

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        if event.trigger.event_type == EventType.FLATFIELD:
            self.FF_chargesComponent(event=event, *args, **kwargs)
        elif event.trigger.event_type in [
            EventType.SKY_PEDESTAL,
            EventType.DARK_PEDESTAL,
            EventType.ELECTRONIC_PEDESTAL,
        ]:
            self.Ped_chargesComponent(event=event, *args, **kwargs)
        else:
            self.log.warning(
                f"event {event.index.event_id} is event type {event.trigger.event_type} which is not used here"
            )

    def finish(self, *args, **kwargs):
        if self._FF_chargesContainers is None:
            self._FF_chargesContainers = self.FF_chargesComponent.finish(
                *args, **kwargs
            )
            self._FF_chargesContainers = merge_map_ArrayDataContainer(
                self._FF_chargesContainers
            )
        if self._Ped_chargesContainers is None:
            self._Ped_chargesContainers = self.Ped_chargesComponent.finish(
                *args, **kwargs
            )
            self._Ped_chargesContainers = merge_map_ArrayDataContainer(
                self._Ped_chargesContainers
            )
        nectarGainSPEresult = next(SPEfitContainer.from_hdf5(self.SPE_result))
        photo_stat = eval(self.PhotoStatAlgorithm).create_from_chargesContainer(
            FFcharge=self._FF_chargesContainers,
            Pedcharge=self._Ped_chargesContainers,
            SPE_result=nectarGainSPEresult,
            coefCharge_FF_Ped=self.__coefCharge_FF_Ped,
            parent=self,
            **self._PhotoStatAlgorithm_kwargs,
        )
        fit_output = photo_stat.run(pixels_id=self.asked_pixels_id, *args, **kwargs)
        return photo_stat.results
