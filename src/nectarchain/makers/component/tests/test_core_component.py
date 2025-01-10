from unittest.mock import patch

import numpy as np
import pytest
from ctapipe.containers import EventType
from ctapipe.utils import get_dataset_path

from nectarchain.data.container import ArrayDataContainer, NectarCAMPedestalContainer
from nectarchain.makers.component import ArrayDataComponent, get_valid_component
from nectarchain.makers.core import BaseNectarCAMCalibrationTool


def test_get_valid_component():
    assert isinstance(get_valid_component(), dict)


class BaseTestArrayDataComponent:
    RUN_NUMBER = 3938
    RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
    NPIXELS = 1834
    PIXEL_ID = 121
    NSAMPLES = 60
    TRIGGER = EventType.FLATFIELD

    @pytest.fixture
    def container_1(self):
        return ArrayDataContainer(
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
        )

    @pytest.fixture
    def container_2(self):
        return ArrayDataContainer(
            run_number=1,
            nevents=2,
            npixels=3,
            camera="the camera",
            pixels_id=np.array([1, 2, 3]),
            broken_pixels_hg=np.ones((2, 3), dtype=bool),
            broken_pixels_lg=np.ones((2, 3), dtype=bool),
            ucts_timestamp=np.array([3, 4]),
            ucts_busy_counter=np.array([1, 2]),
            ucts_event_counter=np.array([1, 2]),
            event_type=np.array([1, 1]),
            event_id=np.array([6, 8]),
            trig_pattern_all=np.array([2, 5, 4]),
            trig_pattern=np.array([2, 3]),
            multiplicity=np.array([2, 2]),
        )

    @pytest.fixture
    def container_3(self):
        return ArrayDataContainer(
            run_number=1,
            nevents=2,
            npixels=2,
            camera="the camera",
            pixels_id=np.array([1, 2]),
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
        )

    @pytest.fixture
    def eventsource(self):
        return BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, max_events=1, run_file=self.RUN_FILE
        )

    @pytest.fixture
    def event(self, eventsource):
        return next(eventsource.__iter__())


class TestArrayDataComponent(BaseTestArrayDataComponent):
    @pytest.fixture
    @patch.multiple(ArrayDataComponent, __abstractmethods__=set())
    def instance(self, eventsource):
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        return ArrayDataComponent(subarray=eventsource.subarray, parent=parent)

    def test_init(self, instance):
        assert instance._pixels_id[100] == self.PIXEL_ID
        assert instance._run_number == self.RUN_NUMBER
        assert instance._npixels == self.NPIXELS
        assert instance.pixels_id[100] == self.PIXEL_ID
        assert instance.run_number == self.RUN_NUMBER
        assert instance.npixels == self.NPIXELS

        assert isinstance(instance.trigger_list, list)
        assert instance._nsamples == self.NSAMPLES
        assert isinstance(instance._ucts_timestamp, dict)
        assert isinstance(instance._ucts_busy_counter, dict)
        assert isinstance(instance._ucts_event_counter, dict)
        assert isinstance(instance._event_type, dict)
        assert isinstance(instance._event_id, dict)
        assert isinstance(instance._trig_pattern_all, dict)
        assert isinstance(instance._broken_pixels_hg, dict)
        assert isinstance(instance._broken_pixels_lg, dict)

    def test_init_trigger_type(self, instance):
        instance._init_trigger_type(self.TRIGGER)
        assert self.TRIGGER in instance.trigger_list
        assert f"{self.TRIGGER.name}" in instance._ucts_busy_counter
        assert f"{self.TRIGGER.name}" in instance._ucts_timestamp
        assert f"{self.TRIGGER.name}" in instance._ucts_event_counter
        assert f"{self.TRIGGER.name}" in instance._event_type
        assert f"{self.TRIGGER.name}" in instance._event_id
        assert f"{self.TRIGGER.name}" in instance._trig_pattern_all
        assert f"{self.TRIGGER.name}" in instance._broken_pixels_hg
        assert f"{self.TRIGGER.name}" in instance._broken_pixels_lg

    def test_compute_broken_pixels(self):
        wfs_hg = np.empty((10, 10, 10))
        wfs_lg = np.empty((10, 10, 10))

        a, b = ArrayDataComponent._compute_broken_pixels(wfs_hg, wfs_lg)
        assert a.shape == (10, 10)
        assert b.shape == (10, 10)
        assert a.dtype == bool
        assert b.dtype == bool

    def test_compute_broken_pixels_event(self, instance, event):
        pixels_id = np.array([3])
        a, b = instance._compute_broken_pixels_event(event, pixels_id)
        assert a.shape == (len(pixels_id),)
        assert b.shape == (len(pixels_id),)
        assert a.dtype == bool
        assert b.dtype == bool

    def test_get_name_trigger(self):
        output = ArrayDataComponent._get_name_trigger(EventType.FLATFIELD)
        assert output == "FLATFIELD"
        output = ArrayDataComponent._get_name_trigger(None)
        assert output == "None"

    def test_call(self, instance, event):
        a, b = instance(event, return_wfs=True)
        name = ArrayDataComponent._get_name_trigger(event.trigger.event_type)
        assert np.array(instance._ucts_timestamp[f"{name}"]).shape == (1,)
        assert np.array(instance._ucts_busy_counter[f"{name}"]).shape == (1,)
        assert np.array(instance._ucts_event_counter[f"{name}"]).shape == (1,)
        assert np.array(instance._event_type[f"{name}"]).shape == (1,)
        assert np.array(instance._event_id[f"{name}"]).shape == (1,)
        assert np.array(instance._trig_pattern_all[f"{name}"]).shape == (1, 1855, 4)
        assert np.array(instance._broken_pixels_hg[f"{name}"]).shape == (
            1,
            self.NPIXELS,
        )
        assert np.array(instance._broken_pixels_lg[f"{name}"]).shape == (
            1,
            self.NPIXELS,
        )
        assert a.shape == (self.NPIXELS, self.NSAMPLES)
        assert b.shape == (self.NPIXELS, self.NSAMPLES)

        assert (
            instance.ucts_timestamp(event.trigger.event_type)
            == instance._ucts_timestamp[f"{name}"]
        )
        assert (
            instance.ucts_busy_counter(event.trigger.event_type)
            == instance._ucts_busy_counter[f"{name}"]
        )
        assert (
            instance.ucts_event_counter(event.trigger.event_type)
            == instance._ucts_event_counter[f"{name}"]
        )
        assert (
            instance.event_type(event.trigger.event_type)
            == instance._event_type[f"{name}"]
        )
        assert (
            instance.event_id(event.trigger.event_type) == instance._event_id[f"{name}"]
        )
        assert np.all(
            instance.trig_pattern_all(event.trigger.event_type)
            == np.array(
                instance._trig_pattern_all[f"{name}"],
                dtype=ArrayDataContainer.fields["trig_pattern_all"].dtype,
            )
        )
        assert np.all(
            instance.broken_pixels_hg(event.trigger.event_type)[0]
            == instance._broken_pixels_hg[f"{name}"][0]
        )
        assert np.all(
            instance.broken_pixels_lg(event.trigger.event_type)[0]
            == instance._broken_pixels_lg[f"{name}"][0]
        )

        assert np.all(
            instance.trig_pattern(event.trigger.event_type)
            == instance.trig_pattern_all(event.trigger.event_type).any(axis=2)
        )
        assert np.all(
            instance.multiplicity(event.trigger.event_type)
            == np.count_nonzero(instance.trig_pattern(event.trigger.event_type), axis=1)
        )

    def test_select_container_array_field(self, container_1):
        res = ArrayDataComponent.select_container_array_field(
            container_1, np.array([1, 2, 4]), field="broken_pixels_hg"
        )
        assert res.shape == (2, 2)

    def test_merge_along_slices(self, container_1, container_2):
        pass

    def test_merge_error_type(self, container_1):
        container = NectarCAMPedestalContainer()
        with pytest.raises(Exception):
            ArrayDataComponent.merge(container_1, container)

    def test_merge_error_pixels(self, container_1, container_3):
        with pytest.raises(Exception):
            ArrayDataComponent.merge(container_1, container_3)

    def test_merge_error_runnumber(self, container_1, container_2):
        container_2.run_number = 20
        with pytest.raises(Exception):
            ArrayDataComponent.merge(container_1, container_2)

    def test_merge(self, container_1, container_2):
        merged_container = ArrayDataComponent.merge(container_1, container_2)
        assert isinstance(merged_container, ArrayDataContainer)
        assert np.all(
            merged_container.broken_pixels_hg
            == np.concatenate(
                [container_1.broken_pixels_hg, container_2.broken_pixels_hg]
            )
        )
        assert merged_container.nevents == container_1.nevents + container_2.nevents
        assert np.all(merged_container.event_id == np.array([2, 4, 6, 8]))
