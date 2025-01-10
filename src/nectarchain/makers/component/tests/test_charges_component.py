from argparse import ArgumentError

import numpy as np
import pytest

from nectarchain.data.container import ChargesContainer, ChargesContainers
from nectarchain.makers.component import ArrayDataComponent, ChargesComponent
from nectarchain.makers.component.tests.test_core_component import (
    TestArrayDataComponent,
)
from nectarchain.makers.core import BaseNectarCAMCalibrationTool


class TestChargesComponent(TestArrayDataComponent):
    METHOD = "GlobalPeakWindowSum"
    EXTRACTOR_KWARGS = {"window_width": 12, "window_shift": 4}

    @pytest.fixture
    def instance(self, eventsource):
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        return ChargesComponent(
            subarray=eventsource.subarray,
            parent=parent,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            method=self.METHOD,
        )

    @pytest.fixture
    def charges_container_1(self):
        return ChargesContainer(
            run_number=1,
            nevents=2,
            npixels=3,
            camera="the camera",
            pixels_id=np.array([1, 2, 3]),
            broken_pixels_hg=np.zeros((2, 3), dtype=bool),
            broken_pixels_lg=np.zeros((2, 3), dtype=bool),
            ucts_timestamp=np.array([1, 2]),
            ucts_busy_counter=np.array([1, 2]),
            ucts_event_counter=np.array([1, 2]),
            event_type=np.array([1, 1]),
            event_id=np.array([2, 4]),
            trig_pattern_all=np.array([2, 5, 4]),
            trig_pattern=np.array([2, 3]),
            multiplicity=np.array([2, 2]),
            charges_hg=np.array([[100, 110, 120], [210, 220, 230]]),
            charges_lg=np.array([[10, 11, 12], [21, 22, 23]]),
            peak_hg=np.array([[1, 2, 3], [4, 5, 6]]),
            peak_lg=np.array([[1, 2, 3], [4, 5, 6]]),
        )

    def test_init(self, instance):
        assert issubclass(instance.__class__, ArrayDataComponent)
        assert instance.method == self.METHOD
        assert instance.extractor_kwargs == self.EXTRACTOR_KWARGS
        assert isinstance(instance._charges_hg, dict)
        assert isinstance(instance._charges_lg, dict)
        assert isinstance(instance._peak_hg, dict)
        assert isinstance(instance._peak_lg, dict)

    def test_init_trigger_type(self, instance):
        instance._init_trigger_type(self.TRIGGER)
        assert self.TRIGGER in instance.trigger_list
        assert f"{self.TRIGGER.name}" in instance._charges_hg
        assert f"{self.TRIGGER.name}" in instance._charges_lg
        assert f"{self.TRIGGER.name}" in instance._peak_hg
        assert f"{self.TRIGGER.name}" in instance._peak_lg

    def test_call(self, instance, event):
        instance(event)
        name = ChargesComponent._get_name_trigger(event.trigger.event_type)
        assert np.array(instance._charges_hg[f"{name}"]).shape == (1, self.NPIXELS)
        assert np.array(instance._charges_lg[f"{name}"]).shape == (1, self.NPIXELS)
        assert np.array(instance._peak_hg[f"{name}"]).shape == (1, self.NPIXELS)
        assert np.array(instance._peak_lg[f"{name}"]).shape == (1, self.NPIXELS)

        assert np.all(
            instance.charges_hg(event.trigger.event_type)
            == instance._charges_hg[f"{name}"]
        )
        assert np.all(
            instance.charges_lg(event.trigger.event_type)
            == instance._charges_lg[f"{name}"]
        )
        assert np.all(
            instance.peak_hg(event.trigger.event_type)
            == np.uint16(instance._peak_hg[f"{name}"])
        )
        assert np.all(
            instance.peak_lg(event.trigger.event_type)
            == np.uint16(instance._peak_lg[f"{name}"])
        )

    def test_finish(self, instance, event):
        instance(event)
        output = instance.finish()
        output.validate()
        assert isinstance(output, ChargesContainers)

    def test_sort(self, charges_container_1):
        with pytest.raises(ArgumentError):
            ChargesComponent.sort(charges_container_1, method="blabla")
        output = ChargesComponent.sort(charges_container_1)
        assert np.all(output.event_id == np.sort(charges_container_1.event_id))


"""make_histo

_get_extractor_kwargs_from_method_and_kwargs
_get_imageExtractor
sort
select_charges_hg
select_charges_lg
_create_from_waveforms_looping_eventType
create_from_waveforms
compute_charges
_histo
histo_hg
histo_lg
"""
