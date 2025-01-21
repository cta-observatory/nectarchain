from unittest.mock import patch

import numpy as np
import pytest
from ctapipe.utils import get_dataset_path

from nectarchain.data.container import SPEfitContainer
from nectarchain.makers.component import (
    ChargesComponent,
    FlatFieldCombinedSPEStdNectarCAMComponent,
    FlatFieldSingleHHVSPENectarCAMComponent,
    FlatFieldSingleHHVSPEStdNectarCAMComponent,
    FlatFieldSingleNominalSPENectarCAMComponent,
    FlatFieldSingleNominalSPEStdNectarCAMComponent,
)
from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm
from nectarchain.makers.core import BaseNectarCAMCalibrationTool


class TestFlatFieldSingleNominalSPENectarCAMComponent:
    RUN_NUMBER = 3938
    RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
    NPIXELS = 1834
    METHOD = "GlobalPeakWindowSum"
    EXTRACTOR_KWARGS = {"window_width": 3, "window_shift": 1}
    MULTIPROC = True
    NPROC = 2
    CHUNKSIZE = 1
    ASKED_PIXELS_ID = [50, 100, 200, 300]
    CLASS = FlatFieldSingleNominalSPENectarCAMComponent

    @pytest.fixture
    def eventsource(self):
        return BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, run_file=self.RUN_FILE
        )

    @pytest.fixture
    def event(self, eventsource):
        return next(eventsource.__iter__())

    @pytest.fixture
    def instance(self, eventsource):
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        return self.CLASS(
            subarray=eventsource.subarray,
            parent=parent,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            method=self.METHOD,
            multiproc=self.MULTIPROC,
            chunksize=self.CHUNKSIZE,
            nproc=self.NPROC,
            asked_pixels_id=self.ASKED_PIXELS_ID,
        )

    def test_init(self, instance):
        assert isinstance(instance, self.CLASS)
        assert isinstance(instance._SPEfitalgorithm_kwargs, dict)
        assert instance._SPEfitalgorithm_kwargs == {
            "multiproc": self.MULTIPROC,
            "chunksize": self.CHUNKSIZE,
            "nproc": self.NPROC,
        }
        assert isinstance(instance.chargesComponent, ChargesComponent)
        assert instance._chargesContainers is None

    def test_call(self, instance, event):
        instance(event)
        assert len(instance.chargesComponent.trigger_list) == 1

    @pytest.mark.skip(reason="test multiproc make GitHub worker be killed")
    def test_finish_multiproc(self):
        SPEalgorithm.window_length.default_value = 2
        # We need to re-instance objects because otherwise, en exception is raised :
        # ReferenceError('weakly-referenced object no longer exists')
        # in component init in ctapipe
        eventsource = BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, run_file=self.RUN_FILE
        )
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        instance = self.CLASS(
            subarray=eventsource.subarray,
            parent=parent,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            method=self.METHOD,
            multiproc=self.MULTIPROC,
            chunksize=self.CHUNKSIZE,
            nproc=self.NPROC,
            asked_pixels_id=self.ASKED_PIXELS_ID,
        )
        for event in eventsource:
            instance(event)
        output = instance.finish(
            tol=100,
        )
        assert isinstance(output, SPEfitContainer)
        assert isinstance(output.is_valid, np.ndarray)

    def test_finish_singleproc(self):
        SPEalgorithm.window_length.default_value = 2
        eventsource = BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, run_file=self.RUN_FILE
        )
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        instance = self.CLASS(
            subarray=eventsource.subarray,
            parent=parent,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            method=self.METHOD,
            multiproc=False,
            chunksize=self.CHUNKSIZE,
            nproc=self.NPROC,
            asked_pixels_id=self.ASKED_PIXELS_ID,
        )
        for event in eventsource:
            instance(event)
        output = instance.finish(
            tol=100,
        )
        assert isinstance(output, SPEfitContainer)
        assert isinstance(output.is_valid, np.ndarray)

    def test_finish_empty(self, instance):
        output = instance.finish()
        assert output is None


class TestFlatFieldSingleNominalSPEStdNectarCAMComponent(
    TestFlatFieldSingleNominalSPENectarCAMComponent
):
    CLASS = FlatFieldSingleNominalSPEStdNectarCAMComponent


class TestFlatFieldSingleHHVSPENectarCAMComponent(
    TestFlatFieldSingleNominalSPENectarCAMComponent
):
    CLASS = FlatFieldSingleHHVSPENectarCAMComponent


class TestFlatFieldSingleHHVSPEStdNectarCAMComponent(
    TestFlatFieldSingleNominalSPENectarCAMComponent
):
    CLASS = FlatFieldSingleHHVSPEStdNectarCAMComponent


class TestFlatFieldCombinedSPEStdNectarCAMComponent(
    TestFlatFieldSingleNominalSPENectarCAMComponent
):
    CLASS = FlatFieldCombinedSPEStdNectarCAMComponent
    SPE_RESULT = "./tmp/"

    @pytest.fixture
    def instance(self, eventsource):
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        return self.CLASS(
            subarray=eventsource.subarray,
            parent=parent,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            method=self.METHOD,
            multiproc=self.MULTIPROC,
            chunksize=self.CHUNKSIZE,
            nproc=self.NPROC,
            asked_pixels_id=self.ASKED_PIXELS_ID,
            SPE_result=self.SPE_RESULT,
        )

    def test_init(self, instance):
        assert isinstance(instance, self.CLASS)
        assert isinstance(instance._SPEfitalgorithm_kwargs, dict)
        assert instance._SPEfitalgorithm_kwargs == {
            "SPE_result": self.SPE_RESULT,
            "multiproc": self.MULTIPROC,
            "chunksize": self.CHUNKSIZE,
            "nproc": self.NPROC,
        }
        assert isinstance(instance.chargesComponent, ChargesComponent)
        assert instance._chargesContainers is None

    @pytest.mark.skip(reason="test multiproc make GitHub worker be killed")
    def test_finish_multiproc(self):
        # We need to re-instance objects because otherwise, en exception is raised :
        # ReferenceError('weakly-referenced object no longer exists')
        # in component init in ctapipe
        eventsource = BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, run_file=self.RUN_FILE
        )
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        instance = self.CLASS(
            subarray=eventsource.subarray,
            parent=parent,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            method=self.METHOD,
            multiproc=self.MULTIPROC,
            chunksize=self.CHUNKSIZE,
            nproc=self.NPROC,
            asked_pixels_id=self.ASKED_PIXELS_ID,
            SPE_result=self.SPE_RESULT,
        )
        spe_fit_container = SPEfitContainer(
            likelihood=np.random.randn(self.NPIXELS, 3),
            p_value=np.random.randn(self.NPIXELS, 3),
            pedestal=60 + 5 * np.random.randn(self.NPIXELS, 3),
            pedestalWidth=5 + 0.1 * np.random.randn(self.NPIXELS, 3),
            resolution=np.random.rand(self.NPIXELS, 3),
            luminosity=np.random.rand(self.NPIXELS, 3),
            mean=60 + 5 * np.random.randn(self.NPIXELS, 3),
            n=np.random.rand(self.NPIXELS, 3),
            pp=np.random.rand(self.NPIXELS, 3),
            is_valid=np.ones((self.NPIXELS), dtype=bool),
            high_gain=np.random.randn(self.NPIXELS, 3) * 5 + 300,
            low_gain=np.random.randn(self.NPIXELS, 3) * 1 + 60,
            pixels_id=instance.pixels_id,
        )
        with patch(
            "nectarchain.data.container.SPEfitContainer.from_hdf5",
            return_value=(spe_fit_container for i in range(2)),
        ):
            SPEalgorithm.window_length.default_value = 2
            for event in eventsource:
                instance(event)
            output = instance.finish(
                tol=100,
            )
        assert isinstance(output, SPEfitContainer)
        assert isinstance(output.is_valid, np.ndarray)

    def test_finish_singleproc(self):
        eventsource = BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, run_file=self.RUN_FILE
        )
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        instance = self.CLASS(
            subarray=eventsource.subarray,
            parent=parent,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            method=self.METHOD,
            multiproc=False,
            chunksize=self.CHUNKSIZE,
            nproc=self.NPROC,
            asked_pixels_id=self.ASKED_PIXELS_ID,
            SPE_result=self.SPE_RESULT,
        )
        spe_fit_container = SPEfitContainer(
            likelihood=np.random.randn(self.NPIXELS, 3),
            p_value=np.random.randn(self.NPIXELS, 3),
            pedestal=60 + 5 * np.random.randn(self.NPIXELS, 3),
            pedestalWidth=5 + 0.1 * np.random.randn(self.NPIXELS, 3),
            resolution=np.random.rand(self.NPIXELS, 3),
            luminosity=np.random.rand(self.NPIXELS, 3),
            mean=60 + 5 * np.random.randn(self.NPIXELS, 3),
            n=np.random.rand(self.NPIXELS, 3),
            pp=np.random.rand(self.NPIXELS, 3),
            is_valid=np.ones((self.NPIXELS), dtype=bool),
            high_gain=np.random.randn(self.NPIXELS, 3) * 5 + 300,
            low_gain=np.random.randn(self.NPIXELS, 3) * 1 + 60,
            pixels_id=instance.pixels_id,
        )
        with patch(
            "nectarchain.data.container.SPEfitContainer.from_hdf5",
            return_value=(spe_fit_container for i in range(2)),
        ):
            SPEalgorithm.window_length.default_value = 2
            for event in eventsource:
                instance(event)
            output = instance.finish(
                tol=100,
            )
        assert isinstance(output, SPEfitContainer)
        assert isinstance(output.is_valid, np.ndarray)
