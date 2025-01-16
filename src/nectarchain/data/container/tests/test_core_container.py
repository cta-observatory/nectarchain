import numpy as np
from ctapipe.containers import EventType, Field
from ctapipe.io import HDF5TableWriter

from nectarchain.data.container import (
    ArrayDataContainer,
    NectarCAMContainer,
    TriggerMapContainer,
)
from nectarchain.data.container.core import get_array_keys, merge_map_ArrayDataContainer


class Container_test(NectarCAMContainer):
    field1 = Field(type=np.ndarray, dtype=np.uint16, ndim=1, description="filed1")
    field2 = Field(type=np.uint16, description="field2")
    field3 = Field(type=np.ndarray, dtype=np.uint8, ndim=2, description="field3")


def create_fake_NectarCAMContainer():
    container = Container_test(
        field1=np.array([1, 2, 3], dtype=np.uint16),
        field2=np.uint16(5),
        field3=np.array([[4, 5, 6], [3, 4, 7]], dtype=np.uint8),
    )
    container.validate()
    return container


class TestGetArrayKeys:
    def test_get_array_keys(self):
        container = create_fake_NectarCAMContainer()
        keys = get_array_keys(container)
        assert ["field1", "field3"] == keys


class TestNectarCAMContainer:
    def test_create_NectarCAMContainer(self):
        test = create_fake_NectarCAMContainer()
        assert isinstance(test, NectarCAMContainer)


def create_fake_ArrayDataContainer():
    nevents = TestArrayDataContainer.nevents
    npixels = TestArrayDataContainer.npixels
    rng = np.random.default_rng()
    test = ArrayDataContainer(
        pixels_id=np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16),
        nevents=np.uint64(nevents),
        npixels=np.uint16(npixels),
        run_number=np.uint16(TestArrayDataContainer.run_number),
        camera="TEST",
        broken_pixels_hg=rng.integers(
            low=0, high=1.1, size=(nevents, npixels), dtype=bool
        ),
        broken_pixels_lg=rng.integers(
            low=0, high=1.1, size=(nevents, npixels), dtype=bool
        ),
        ucts_timestamp=rng.integers(low=0, high=100, size=(nevents), dtype=np.uint64),
        ucts_busy_counter=rng.integers(
            low=0, high=100, size=(nevents), dtype=np.uint32
        ),
        ucts_event_counter=rng.integers(
            low=0, high=100, size=(nevents), dtype=np.uint32
        ),
        event_type=rng.integers(low=0, high=1, size=(nevents), dtype=np.uint8),
        event_id=rng.integers(low=0, high=1000, size=(nevents), dtype=np.uint32),
        trig_pattern_all=rng.integers(
            low=0, high=1, size=(nevents, npixels, 4), dtype=bool
        ),
        trig_pattern=rng.integers(low=0, high=1, size=(nevents, npixels), dtype=bool),
        multiplicity=rng.integers(low=0, high=1, size=(nevents), dtype=np.uint16),
    )
    test.validate()
    return test


class TestArrayDataContainer:
    # THESE TESTs ARE ALSO COVERING THE NectarCAMContainer class#

    nevents = 10
    npixels = 40
    run_number = 1234

    def test_create_ArrayDataContainer(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16)
        nevents = np.uint64(TestArrayDataContainer.nevents)
        npixels = np.uint16(TestArrayDataContainer.npixels)
        run_number = np.uint16(TestArrayDataContainer.run_number)
        arrayDataContainer = ArrayDataContainer(
            run_number=run_number,
            pixels_id=pixels_id,
            nevents=nevents,
            npixels=npixels,
        )
        arrayDataContainer.validate()
        assert arrayDataContainer.run_number == run_number
        assert arrayDataContainer.pixels_id.tolist() == pixels_id.tolist()
        assert arrayDataContainer.nevents == nevents
        assert arrayDataContainer.npixels == npixels

    def test_write_ArrayDataContainer(self, tmp_path="/tmp"):
        arrayDataContainer = create_fake_ArrayDataContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="ArrayDataContainer", containers=arrayDataContainer)
        writer.close()

    # Tests that a ChargeContainer object can be loaded from a file and the object is
    # correctly initialized.
    def test_from_hdf5(self, tmp_path="/tmp"):
        arrayDataContainer = create_fake_ArrayDataContainer()
        tmp_path += f"/{np.random.randn(1)[0]}.h5"
        writer = HDF5TableWriter(
            filename=tmp_path,
            mode="w",
            group_name="data",
        )
        writer.write(table_name="ArrayDataContainer_0", containers=arrayDataContainer)
        writer.close()
        loaded_arrayDataContainer = next(ArrayDataContainer.from_hdf5(tmp_path))
        assert isinstance(loaded_arrayDataContainer, ArrayDataContainer)
        assert loaded_arrayDataContainer.run_number == arrayDataContainer.run_number
        assert (
            loaded_arrayDataContainer.pixels_id.tolist()
            == arrayDataContainer.pixels_id.tolist()
        )
        assert loaded_arrayDataContainer.nevents == arrayDataContainer.nevents
        assert loaded_arrayDataContainer.npixels == arrayDataContainer.npixels

    def test_access_properties(self):
        pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10])
        nevents = 40
        npixels = 10
        run_number = 1234
        arrayDataContainer = ArrayDataContainer(
            run_number=run_number,
            pixels_id=pixels_id,
            nevents=nevents,
            npixels=npixels,
        )
        assert arrayDataContainer.run_number == run_number
        assert arrayDataContainer.pixels_id.tolist() == pixels_id.tolist()
        assert arrayDataContainer.npixels == npixels
        assert arrayDataContainer.nevents == nevents


class TestTriggerMapContainer:
    def test_create_TriggerMapContainer(self):
        triggerMapContainer = TriggerMapContainer()
        triggerMapContainer.containers[
            EventType.FLATFIELD
        ] = create_fake_NectarCAMContainer()
        triggerMapContainer.containers[
            EventType.MUON
        ] = create_fake_NectarCAMContainer()
        assert isinstance(triggerMapContainer, TriggerMapContainer)

    def test_merge_map_ArrayDataContainer(self):
        triggerMapContainer = TriggerMapContainer()
        arrayDataContainer1 = create_fake_ArrayDataContainer()
        arrayDataContainer2 = create_fake_ArrayDataContainer()

        triggerMapContainer.containers["1"] = arrayDataContainer1
        triggerMapContainer.containers["2"] = arrayDataContainer2

        merged = merge_map_ArrayDataContainer(triggerMapContainer)

        assert (
            merged.nevents == arrayDataContainer1.nevents + arrayDataContainer2.nevents
        )
