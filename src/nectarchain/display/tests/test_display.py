import numpy as np
import pytest
from ctapipe.utils import get_dataset_path

from nectarchain.data.container import WaveformsContainer
from nectarchain.display.display import ContainerDisplay
from nectarchain.makers.core import BaseNectarCAMCalibrationTool

# Use pixel IDs from the real camera geometry for a NectarCAM run
_RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
_eventsource = BaseNectarCAMCalibrationTool.load_run(
    3938, max_events=1, run_file=_RUN_FILE
)
_tel_id = list(_eventsource.subarray.tel_ids)[0]
_geom = _eventsource.subarray.tel[_tel_id].camera.geometry
REAL_PIXEL_IDS = _geom.pix_id.astype(np.uint16)
N_PIXELS_REAL = len(REAL_PIXEL_IDS)


class MockGeometry:
    class PixId:
        value = REAL_PIXEL_IDS

    pix_id = PixId()

    def rotate(self, rotation):
        pass


class TestContainerDisplay:
    def test_display_invalid_container(self):
        with pytest.raises(Exception, match="container can't be displayed"):
            ContainerDisplay.display(object(), evt=0, geometry=MockGeometry())

    def test_plot_waveform(self):
        c = WaveformsContainer(
            wfs_hg=np.zeros((1, 3, 60), dtype=np.uint16),
            wfs_lg=np.zeros((1, 3, 60), dtype=np.uint16),
            nsamples=np.uint8(60),
            pixels_id=np.array([1, 2, 3], dtype=np.uint16),
            run_number=np.uint16(1),
            nevents=np.uint64(1),
            npixels=np.uint16(3),
        )
        c.validate()
        fig, ax = ContainerDisplay.plot_waveform(c, evt=0)
        assert fig is not None and ax is not None
