import copy
from argparse import ArgumentError

import numpy as np
import pytest
from ctapipe.containers import EventType
from ctapipe.image import GlobalPeakWindowSum
from ctapipe_io_nectarcam import constants

from nectarchain.data.container import (
    ChargesContainer,
    ChargesContainers,
    WaveformsContainer,
    WaveformsContainers,
)
from nectarchain.makers.component import ChargesComponent
from nectarchain.makers.component.tests.test_core_component import (
    BaseTestArrayDataComponent,
)
from nectarchain.makers.core import BaseNectarCAMCalibrationTool


class TestChargesComponent(BaseTestArrayDataComponent):
    METHOD = "GlobalPeakWindowSum"
    EXTRACTOR_KWARGS = {"window_width": 3, "window_shift": 1}

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

    @pytest.fixture
    def waveforms_container_1(self):
        return WaveformsContainer(
            nsamples=10,
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
            wfs_hg=1000 * np.random.rand(2, 3, 10),
            wfs_lg=np.random.rand(2, 3, 10),
        )

    @pytest.fixture
    def waveforms_containers(self, waveforms_container_1):
        wfs = WaveformsContainers()
        wfs.containers[EventType.FLATFIELD] = copy.deepcopy(waveforms_container_1)
        wfs.containers[EventType.SKY_PEDESTAL] = copy.deepcopy(waveforms_container_1)
        return wfs

    def test_init(self, instance):
        assert isinstance(instance, ChargesComponent)
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

    def test_get_extractor_kwargs_from_method_and_kwargs(self):
        with pytest.raises(Exception):
            ChargesComponent._get_extractor_kwargs_from_method_and_kwargs(
                method="blabla", kwargs={}
            )
        output = ChargesComponent._get_extractor_kwargs_from_method_and_kwargs(
            method="FullWaveformSum", kwargs={}
        )
        assert output == {}
        output = ChargesComponent._get_extractor_kwargs_from_method_and_kwargs(
            self.METHOD, self.EXTRACTOR_KWARGS
        )
        check = {"apply_integration_correction": False}
        check.update(self.EXTRACTOR_KWARGS)
        assert output == check

    def test_get_name_trigger(self, instance):
        with pytest.raises(ArgumentError):
            ChargesComponent._get_imageExtractor("blabla", instance.subarray)
        output = ChargesComponent._get_imageExtractor(self.METHOD, instance.subarray)
        assert isinstance(output, GlobalPeakWindowSum)

    def test_select_charges_hg(self, charges_container_1):
        output = ChargesComponent.select_charges_hg(
            charges_container_1, pixel_id=np.array([1])
        )
        assert np.all(
            output
            == np.array(
                [
                    [charges_container_1.charges_hg.T[0][0]],
                    [charges_container_1.charges_hg.T[0][1]],
                ]
            )
        )

    def test_select_charges_lg(self, charges_container_1):
        output = ChargesComponent.select_charges_lg(
            charges_container_1, pixel_id=np.array([1])
        )
        assert np.all(
            output
            == np.array(
                [
                    [charges_container_1.charges_lg.T[0][0]],
                    [charges_container_1.charges_lg.T[0][1]],
                ]
            )
        )

    def test_create_from_waveforms_looping_eventType(
        self, waveforms_containers, instance
    ):
        output = ChargesComponent._create_from_waveforms_looping_eventType(
            waveforms_containers,
            subarray=instance.subarray,
            method=self.METHOD,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
        )
        assert isinstance(output, ChargesContainers)
        for field, _ in waveforms_containers.containers.items():
            assert isinstance(output.containers[field], ChargesContainer)

    def test_create_from_waveforms(self, waveforms_container_1, instance):
        output = ChargesComponent.create_from_waveforms(
            waveforms_container_1,
            instance.subarray,
            self.METHOD,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
        )
        assert isinstance(output, ChargesContainer)

    def test_compute_charges(self, waveforms_container_1, instance):
        with pytest.raises(ArgumentError):
            output1, output2 = ChargesComponent.compute_charges(
                waveforms_container_1,
                channel=1000,
                subarray=instance.subarray,
                method=self.METHOD,
                extractor_kwargs=self.EXTRACTOR_KWARGS,
            )
        output1, output2 = ChargesComponent.compute_charges(
            waveforms_container_1,
            channel=constants.HIGH_GAIN,
            subarray=instance.subarray,
            method="FullWaveformSum",
        )
        assert output1.shape == (2, 3)
        assert output2.shape == (2, 3)
        assert np.isclose(
            output1.mean(), 10 * np.mean(1000 * np.random.rand(2, 3, 10)), rtol=2000
        )

        output1, output2 = ChargesComponent.compute_charges(
            waveforms_container_1,
            channel=constants.LOW_GAIN,
            subarray=instance.subarray,
            method="FullWaveformSum",
        )
        assert output1.shape == (2, 3)
        assert output2.shape == (2, 3)
        assert np.isclose(
            output1.mean(), 10 * np.mean(np.random.rand(2, 3, 10)), rtol=2
        )

    def test_histo_hg(self, charges_container_1):
        output = ChargesComponent.histo_hg(
            charges_container_1, n_bins=2, autoscale=False
        )
        assert isinstance(output, np.ndarray)
        assert output.shape == (
            2,
            charges_container_1.npixels,
            charges_container_1.nevents,
        )
        output = ChargesComponent.histo_hg(
            charges_container_1,
            autoscale=True,
        )
        assert isinstance(output, np.ma.masked_array)
        assert output.shape[:2] == (
            2,
            charges_container_1.npixels,
        )
