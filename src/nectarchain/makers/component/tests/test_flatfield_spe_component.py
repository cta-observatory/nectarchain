import numpy as np
import pytest
from ctapipe.utils import get_dataset_path

from nectarchain.data.container import SPEfitContainer
from nectarchain.makers.component import (
    ChargesComponent,
    FlatFieldSingleNominalSPENectarCAMComponent,
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
        return FlatFieldSingleNominalSPENectarCAMComponent(
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
        assert isinstance(instance, FlatFieldSingleNominalSPENectarCAMComponent)
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

    def test_finish_multiproc(self):
        SPEalgorithm.window_length.default_value = 2
        eventsource = BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, run_file=self.RUN_FILE
        )
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        instance = FlatFieldSingleNominalSPENectarCAMComponent(
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
        instance = FlatFieldSingleNominalSPENectarCAMComponent(
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
