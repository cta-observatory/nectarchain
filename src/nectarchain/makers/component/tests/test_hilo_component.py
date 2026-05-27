import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from ctapipe.containers import EventType
from ctapipe_io_nectarcam.constants import N_PIXELS, PIXEL_INDEX

from nectarchain.data.container import ChargesContainer, GainContainer
from nectarchain.makers.component import ChargesComponent, HiLoComponent
from nectarchain.makers.component.tests.test_core_component import (
    BaseTestArrayDataComponent,
)
from nectarchain.makers.core import BaseNectarCAMCalibrationTool
from nectarchain.utils.constants import HILO_DEFAULT


class TestHiLoComponent(BaseTestArrayDataComponent):
    tmpdir = tempfile.TemporaryDirectory().name
    GAIN_FILE = Path(tmpdir) / "gain.h5"

    @pytest.fixture
    def instance(self, eventsource):
        parent = BaseNectarCAMCalibrationTool()
        parent._event_source = eventsource
        parent.run_number = self.RUN_NUMBER
        parent.npixels = self.NPIXELS

        gain_container = GainContainer(
            high_gain=np.ones((self.NPIXELS, 3)),
            low_gain=np.ones((self.NPIXELS, 3)),
            pixels_id=np.arange(self.NPIXELS),
            is_valid=np.ones(self.NPIXELS, dtype=bool),
        )

        with patch(
            (
                "nectarchain.makers.component.hilo_component.ContainerUtils."
                "get_container_from_hdf5"
            ),
            return_value=gain_container,
        ):
            yield HiLoComponent(
                subarray=eventsource.subarray,
                parent=parent,
                gain_file=self.GAIN_FILE,
            )

    @pytest.fixture
    def charges_container_test(self):
        """
        Known HiLo ratio = 10
        """
        return ChargesContainer(
            run_number=1,
            nevents=2,
            npixels=3,
            camera="the camera",
            pixels_id=np.array([0, 1, 2]),
            charges_hg=np.array(
                [
                    [100, 110, 120],
                    [200, 220, 240],
                ]
            ),
            charges_lg=np.array(
                [
                    [10, 11, 12],
                    [20, 22, 24],
                ]
            ),
        )

    @pytest.fixture
    def gain_container_test(self):
        return GainContainer(
            high_gain=np.full((N_PIXELS, 3), 5.0),
            low_gain=np.zeros((N_PIXELS, 3)),
            pixels_id=PIXEL_INDEX,
            is_valid=np.ones(N_PIXELS, dtype=bool),
        )

    def test_init(self, instance):
        assert isinstance(instance, HiLoComponent)
        assert instance.gain_file == self.GAIN_FILE
        assert isinstance(instance.chargesComponent, ChargesComponent)
        assert instance._chargesContainer is None

    def test_init_gain_container(self, instance):
        instance._init_gain_container()

        gain_container = instance._HiLoComponent__gain_container

        assert isinstance(gain_container, GainContainer)
        assert "high_gain" in gain_container.keys()
        assert "low_gain" in gain_container.keys()

    def test_call(self, instance, event):
        with patch.object(event.trigger, "event_type", new=EventType.FLATFIELD):
            instance(event)
        assert len(instance.chargesComponent.trigger_list) == 1

    def test_compute_low_gain(
        self, instance, charges_container_test, gain_container_test
    ):
        instance._chargesContainer = charges_container_test
        instance._HiLoComponent__gain_container = gain_container_test

        instance._compute_low_gain()

        assert np.all(instance._HiLoComponent__gain_container["low_gain"][:3, 0] == 0.5)
        assert np.all(
            instance._HiLoComponent__gain_container["low_gain"][3:, 0]
            == gain_container_test["high_gain"][3:, 0] / HILO_DEFAULT
        )

    def test_finish(self, instance, charges_container_test):
        instance._chargesContainer = charges_container_test

        with patch.object(
            instance,
            "_compute_low_gain",
            wraps=instance._compute_low_gain,
        ) as mock_compute_low_gain:
            output = instance.finish()
            mock_compute_low_gain.assert_called_once()

        assert isinstance(output, GainContainer)
        assert output["is_valid"].shape == (N_PIXELS,)
        assert output["high_gain"].shape == (N_PIXELS, 3)
        assert output["low_gain"].shape == (N_PIXELS, 3)
        assert np.all(output["pixels_id"] == PIXEL_INDEX)
