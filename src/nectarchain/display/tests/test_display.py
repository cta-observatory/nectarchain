import numpy as np
import pytest
from ctapipe.utils import get_dataset_path

from nectarchain.data.container import WaveformsContainer
from nectarchain.display.display import ContainerDisplay
from nectarchain.makers.core import BaseNectarCAMCalibrationTool

# Load the real camera geometry from a NectarCAM test dataset
_RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
_eventsource = BaseNectarCAMCalibrationTool.load_run(
    3938, max_events=1, run_file=_RUN_FILE
)
_tel_id = list(_eventsource.subarray.tel_ids)[0]
GEOMETRY = _eventsource.subarray.tel[_tel_id].camera.geometry


class TestContainerDisplay:
    def test_display_invalid_container(self):
        with pytest.raises(Exception, match="container can't be displayed"):
            ContainerDisplay.display(object(), evt=0, geometry=GEOMETRY)

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
