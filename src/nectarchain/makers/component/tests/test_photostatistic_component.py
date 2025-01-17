import os
from unittest.mock import patch

import numpy as np
import pytest
from ctapipe.containers import EventType
from ctapipe.utils import get_dataset_path
from ctapipe_io_nectarcam.constants import N_SAMPLES
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer  # noqa : F401

from nectarchain.data.container import GainContainer, SPEfitContainer
from nectarchain.makers.component import PhotoStatisticNectarCAMComponent
from nectarchain.makers.component.charges_component import ChargesComponent
from nectarchain.makers.core import BaseNectarCAMCalibrationTool


class TestPhotoStatisticNectarCAMComponent:
    RUN_NUMBER = 3938
    RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
    NPIXELS = 1834
    METHOD = "GlobalPeakWindowSum"
    EXTRACTOR_KWARGS = {"window_width": 3, "window_shift": 1}
    ASKED_PIXELS_ID = [50, 100, 200, 300]
    SPE_RESULT = "./tmp/"

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
        return PhotoStatisticNectarCAMComponent(
            subarray=eventsource.subarray,
            parent=parent,
            SPE_result=self.SPE_RESULT,
            method=self.METHOD,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            asked_pixels_id=self.ASKED_PIXELS_ID,
        )

    def test_init(self, instance):
        assert isinstance(instance, PhotoStatisticNectarCAMComponent)
        assert isinstance(instance._PhotoStatAlgorithm_kwargs, dict)
        assert isinstance(instance.coefCharge_FF_Ped, float)
        assert (
            instance.coefCharge_FF_Ped
            == self.EXTRACTOR_KWARGS["window_width"] / N_SAMPLES
        )
        assert isinstance(instance.Ped_chargesComponent, ChargesComponent)
        assert isinstance(instance.FF_chargesComponent, ChargesComponent)
        assert instance._Ped_chargesContainers is None
        assert instance._FF_chargesContainers is None

    def test_call(self, instance, event):
        with patch.object(event.trigger, "event_type", new=EventType.FLATFIELD):
            instance(event)
        assert len(instance.FF_chargesComponent.trigger_list) == 1
        with patch.object(event.trigger, "event_type", new=EventType.SKY_PEDESTAL):
            instance(event)
        assert len(instance.Ped_chargesComponent.trigger_list) == 1
        assert len(instance.FF_chargesComponent.trigger_list) == 1
        with patch.object(
            event.trigger, "event_type", new=EventType.ELECTRONIC_PEDESTAL
        ):
            instance(event)
        assert len(instance.Ped_chargesComponent.trigger_list) == 2
        assert len(instance.FF_chargesComponent.trigger_list) == 1
        with patch.object(event.trigger, "event_type", new=EventType.DARK_PEDESTAL):
            instance(event)
        assert len(instance.Ped_chargesComponent.trigger_list) == 3
        assert len(instance.FF_chargesComponent.trigger_list) == 1
        with patch.object(event.trigger, "event_type", new=EventType.UNKNOWN):
            instance(event)
        assert len(instance.Ped_chargesComponent.trigger_list) == 3
        assert len(instance.FF_chargesComponent.trigger_list) == 1

    def test_finish(self):
        eventsource = BaseNectarCAMCalibrationTool.load_run(
            run_number=self.RUN_NUMBER, run_file=self.RUN_FILE
        )
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS
        instance = PhotoStatisticNectarCAMComponent(
            subarray=eventsource.subarray,
            parent=parent,
            SPE_result=self.SPE_RESULT,
            method=self.METHOD,
            extractor_kwargs=self.EXTRACTOR_KWARGS,
            asked_pixels_id=self.ASKED_PIXELS_ID,
        )
        for _ in range(10):
            event = next(eventsource.__iter__())
            with patch.object(event.trigger, "event_type", new=EventType.FLATFIELD):
                instance(event)
        for _ in range(10):
            event = next(eventsource.__iter__())
            with patch.object(event.trigger, "event_type", new=EventType.SKY_PEDESTAL):
                instance(event)
        spe_fit_container = SPEfitContainer(
            likelihood=np.random.randn(self.NPIXELS, 3),
            p_value=np.random.randn(self.NPIXELS, 3),
            pedestal=np.random.randn(self.NPIXELS, 3),
            pedestalWidth=np.random.randn(self.NPIXELS, 3),
            resolution=np.random.randn(self.NPIXELS, 3),
            luminosity=np.random.randn(self.NPIXELS, 3),
            mean=60 * np.random.randn(self.NPIXELS, 3),
            n=np.random.randn(self.NPIXELS, 3),
            pp=np.random.randn(self.NPIXELS, 3),
            is_valid=np.ones((self.NPIXELS), dtype=bool),
            high_gain=np.random.randn(self.NPIXELS, 3) * 300,
            low_gain=np.random.randn(self.NPIXELS, 3) * 60,
            pixels_id=instance.pixels_id,
        )
        with patch(
            "nectarchain.data.container.SPEfitContainer.from_hdf5",
            return_value=(spe_fit_container for i in range(2)),
        ):
            output = instance.finish(figpath="/tmp/photostat_test/")
        assert isinstance(output, GainContainer)
        assert isinstance(output.is_valid, np.ndarray)
        assert os.path.exists("/tmp/photostat_test/plot_correlation_Photo_Stat_SPE.pdf")
