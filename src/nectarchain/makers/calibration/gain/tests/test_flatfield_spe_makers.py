import tempfile
from unittest.mock import patch

import numpy as np
from ctapipe.utils import get_dataset_path

from nectarchain.data import SPEfitContainer
from nectarchain.makers.calibration import (
    FlatFieldSPECombinedStdNectarCAMCalibrationTool,
    FlatFieldSPEHHVNectarCAMCalibrationTool,
    FlatFieldSPEHHVStdNectarCAMCalibrationTool,
    FlatFieldSPENominalNectarCAMCalibrationTool,
    FlatFieldSPENominalStdNectarCAMCalibrationTool,
)
from nectarchain.makers.component.spe.spe_algorithm import SPEalgorithm


class TestFlatFieldSPENominalNectarCAMCalibrationTool:
    RUN_NUMBER = 3938
    RUN_FILE = get_dataset_path("NectarCAM.Run3938.30events.fits.fz")
    NPIXELS = 1834
    METHOD = "GlobalPeakWindowSum"
    EXTRACTOR_KWARGS = {"window_width": 3, "window_shift": 1}
    MULTIPROC = True
    NPROC = 2
    CHUNKSIZE = 1
    ASKED_PIXELS_ID = [50, 100, 200, 300]
    EVENTS_PER_SLICE = 11
    CLASS = [
        FlatFieldSPENominalNectarCAMCalibrationTool,
        FlatFieldSPENominalStdNectarCAMCalibrationTool,
        FlatFieldSPEHHVNectarCAMCalibrationTool,
        FlatFieldSPEHHVStdNectarCAMCalibrationTool,
    ]

    def test_core(self):
        for _class in self.CLASS:
            tool = _class(
                run_number=self.RUN_NUMBER,
                run_file=self.RUN_FILE,
                npixels=self.NPIXELS,
                extractor_kwargs=self.EXTRACTOR_KWARGS,
                method=self.METHOD,
                multiproc=self.MULTIPROC,
                chunksize=self.CHUNKSIZE,
                nproc=self.NPROC,
                asked_pixels_id=self.ASKED_PIXELS_ID,
                reload_events=True,
                events_per_slice=self.EVENTS_PER_SLICE,
            )
            assert isinstance(tool, _class)
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = (
                tmpdirname
                + f"/{np.random.random()}"
                + "_testFlatFieldSPENominalStdNectarCAMCalibrationTool.h5"
            )
            tool = FlatFieldSPENominalStdNectarCAMCalibrationTool(
                run_number=self.RUN_NUMBER,
                output_path=outfile,
                run_file=self.RUN_FILE,
                npixels=self.NPIXELS,
                extractor_kwargs=self.EXTRACTOR_KWARGS,
                method=self.METHOD,
                multiproc=self.MULTIPROC,
                chunksize=self.CHUNKSIZE,
                nproc=self.NPROC,
                asked_pixels_id=self.ASKED_PIXELS_ID,
                reload_events=True,
                events_per_slice=self.EVENTS_PER_SLICE,
            )
            tool.setup()
            tool.start()
            output = tool.finish(return_output_component=True, tol=100)
        assert isinstance(output[0], SPEfitContainer)


class TestFlatFieldSPECombinedStdNectarCAMCalibrationTool(
    TestFlatFieldSPENominalNectarCAMCalibrationTool
):
    SPE_RESULT = "./tmp/run1234.h5"

    def test_core(self):
        SPEalgorithm.window_length.default_value = 1
        with tempfile.TemporaryDirectory() as tmpdirname:
            outfile = (
                tmpdirname
                + f"/{np.random.random()}"
                + "_testFlatFieldSPENominalStdNectarCAMCalibrationTool.h5"
            )
            tool = FlatFieldSPECombinedStdNectarCAMCalibrationTool(
                run_number=self.RUN_NUMBER,
                output_path=outfile,
                run_file=self.RUN_FILE,
                npixels=self.NPIXELS,
                extractor_kwargs=self.EXTRACTOR_KWARGS,
                method=self.METHOD,
                multiproc=self.MULTIPROC,
                chunksize=self.CHUNKSIZE,
                nproc=self.NPROC,
                asked_pixels_id=self.ASKED_PIXELS_ID,
                reload_events=True,
                events_per_slice=self.EVENTS_PER_SLICE,
                SPE_result=self.SPE_RESULT,
            )
            tool.setup()
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
                pixels_id=tool.event_source.nectarcam_service.pixel_ids,
            )
            with patch(
                "nectarchain.data.container.SPEfitContainer.from_hdf5",
                return_value=(spe_fit_container for i in range(10)),
            ):
                tool.start()
                output = tool.finish(return_output_component=True, tol=100)
        assert isinstance(output[0], SPEfitContainer)
