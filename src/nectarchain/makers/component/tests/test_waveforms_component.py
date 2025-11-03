from argparse import ArgumentError

import numpy as np
import pytest

from nectarchain.data.container import WaveformsContainer, WaveformsContainers
from nectarchain.makers.component import WaveformsComponent
from nectarchain.makers.component.tests.test_core_component import (
    BaseTestArrayDataComponent,
)
from nectarchain.makers.core import BaseNectarCAMCalibrationTool


class TestWaveformsComponent(BaseTestArrayDataComponent):
    @pytest.fixture
    def instance(self, eventsource):
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        return WaveformsComponent(subarray=eventsource.subarray, parent=parent)

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

    def test_init(self, instance):
        assert isinstance(instance, WaveformsComponent)
        assert (
            instance.geometry == instance.subarray.tel[instance.tel_id].camera.geometry
        )
        assert isinstance(instance._wfs_lg, dict)
        assert isinstance(instance._wfs_hg, dict)

    def test_init_trigger_type(self, instance):
        instance._init_trigger_type(self.TRIGGER)
        assert self.TRIGGER in instance.trigger_list
        assert f"{self.TRIGGER.name}" in instance._wfs_lg
        assert f"{self.TRIGGER.name}" in instance._wfs_hg

    def test_call(self, instance, event):
        instance(event)
        name = WaveformsComponent._get_name_trigger(event.trigger.event_type)
        assert np.array(instance._wfs_lg[f"{name}"]).shape == (
            1,
            self.NPIXELS,
            self.NSAMPLES,
        )
        assert np.array(instance._wfs_hg[f"{name}"]).shape == (
            1,
            self.NPIXELS,
            self.NSAMPLES,
        )

        assert np.all(
            instance.wfs_lg(event.trigger.event_type) == instance._wfs_lg[f"{name}"]
        )
        assert np.all(
            instance.wfs_hg(event.trigger.event_type) == instance._wfs_hg[f"{name}"]
        )

    def test_finish(self, instance, event):
        instance(event)
        output = instance.finish()
        output.validate()
        assert isinstance(output, WaveformsContainers)

    def tes_create_from_events_list(self):
        pass

    def test_sort(self, waveforms_container_1):
        with pytest.raises(ArgumentError):
            WaveformsComponent.sort(waveforms_container_1, method="blabla")
        output = WaveformsComponent.sort(waveforms_container_1)
        assert np.all(output.event_id == np.sort(waveforms_container_1.event_id))

    def test_select_waveforms_hg(self, waveforms_container_1):
        output = WaveformsComponent.select_waveforms_hg(
            waveforms_container_1, pixel_id=np.array([1])
        )
        assert np.all(
            output
            == np.array(
                [
                    [waveforms_container_1.wfs_hg[0, 0, :]],
                    [waveforms_container_1.wfs_hg[1, 0, :]],
                ]
            )
        )

    def test_select_waveforms_lg(self, waveforms_container_1):
        output = WaveformsComponent.select_waveforms_lg(
            waveforms_container_1, pixel_id=np.array([1])
        )
        assert np.all(
            output
            == np.array(
                [
                    [waveforms_container_1.wfs_lg[0, 0, :]],
                    [waveforms_container_1.wfs_lg[1, 0, :]],
                ]
            )
        )
