import logging
import tempfile
import numpy as np
import pytest
from ctapipe.containers import EventType
from pathlib import Path
from nectarchain.data.container import ChargesContainer, ChargesContainers
from nectarchain.makers import ChargesNectarCAMCalibrationTool
from ctapipe.utils import get_dataset_path

"This test file test the overall workflow of the ChargesNectarCAMCalibrationTool, adapted from the test of the WaveformsNectarCAMCalibrationTool. There are lot of code duplication but I don't care for now. Do you ?"


class TestChargesNectarCAMCalibrationTool : 
    RUNS = {
    "Run number": [3938, 5288],
    "Run file": [
        get_dataset_path("NectarCAM.Run3938.30events.fits.fz"),
        get_dataset_path("NectarCAM.Run5288.0001.fits.fz"),
    ],
    "nevents" : [30,13],
    "N pixels": [1834, 1848],
    "eventType" : EventType.SKY_PEDESTAL,
    "charges_lg_min" : [957,973],
    "charges_lg_max" : [2059,2113],
    "charges_lg_mean" : [19067,19306],
    "charges_lg_std" : [2273,2376],
    "charges_hg_min" : [963,972],
    "charges_hg_max" : [2057,2109],
    "charges_hg_mean" : [19106,19275],
    "charges_hg_std" : [2139,2357],
    "expected_ucts_timestamp_min" : [1674462932637854793, 1715007113924900896],
    "expected_ucts_timestamp_max" : [1674462932695877994, 1715007123524920096],
}
    OVERWRITE = True
    METHOD = "LocalPeakWindowSum"
    EXTRACTOR_KWARGS = {"window_shift": 4, "window_width": 8}
    
    def general_structure_testing(self,output : ChargesContainer,nevents:int,n_pixels:int,run_number:int) :
        assert isinstance(output.pixels_id, np.ndarray)
        assert output.pixels_id.dtype == np.uint16
        assert np.shape(output.pixels_id) == (n_pixels,)
        assert output.run_number == run_number
        assert output.camera == 'NectarCam-003'
        assert output.npixels == n_pixels
        assert isinstance(output.ucts_busy_counter, np.ndarray)
        assert output.ucts_busy_counter.dtype == np.uint32
        assert isinstance(output.ucts_event_counter, np.ndarray)
        assert output.ucts_event_counter.dtype == np.uint32
        assert isinstance(output.event_type, np.ndarray)
        assert output.event_type.dtype == np.uint8
        assert np.all(output.event_type == self.RUNS["eventType"].value)
        assert isinstance(output.trig_pattern, np.ndarray)
        assert output.trig_pattern.dtype == bool
        assert isinstance(output.trig_pattern_all, np.ndarray)
        assert output.trig_pattern_all.dtype == bool
        assert isinstance(output.multiplicity, np.ndarray)
        assert output.multiplicity.dtype == np.uint16
        assert isinstance(output.ucts_timestamp,np.ndarray)
        assert output.ucts_timestamp.dtype == np.uint64
        
        assert isinstance(output.event_id,np.ndarray)
        assert output.event_id.dtype == np.uint32
        assert isinstance(output.broken_pixels_hg,np.ndarray)
        assert output.broken_pixels_hg.dtype == bool
        assert output.broken_pixels_hg.shape == (nevents,n_pixels)
        assert isinstance(output.broken_pixels_lg,np.ndarray)
        assert output.broken_pixels_lg.dtype == bool
        assert output.broken_pixels_lg.shape == (nevents,n_pixels)
        assert output.charges_hg.shape == (nevents, n_pixels)
        assert output.charges_lg.shape == (nevents, n_pixels)
        assert isinstance(output.charges_hg, np.ndarray)
        assert isinstance(output.charges_lg, np.ndarray)
        assert output.charges_hg.dtype == np.uint16
        assert output.charges_lg.dtype == np.uint16
        assert output.peak_hg.shape == (nevents, n_pixels)
        assert output.peak_lg.shape == (nevents, n_pixels)
        assert isinstance(output.peak_hg, np.ndarray)
        assert isinstance(output.peak_lg, np.ndarray)
        assert output.peak_hg.dtype == np.uint16
        assert output.peak_lg.dtype == np.uint16
        assert isinstance(output.method,str)
        assert output.method == self.METHOD


    def test_base(self):
        """
        Test basic functionality, including IO on disk
        """

        events_per_slice =[None,None,10,11,8]
        max_events = [None,10,None,None,10]


        for _max_events,_events_per_slice in zip(max_events,events_per_slice):
            for i, run_number in enumerate(self.RUNS["Run number"]):
                run_file = self.RUNS["Run file"][i]
                n_pixels = self.RUNS["N pixels"][i]
                with tempfile.TemporaryDirectory() as tmpdirname:
                    outfile = tmpdirname + "/charges.h5"

                    # run tool
                    tool = ChargesNectarCAMCalibrationTool(
                        run_number=run_number,
                        run_file=run_file,
                        max_events=_max_events,
                        events_per_slice=_events_per_slice,
                        log_level=0,
                        output_path=outfile,
                        overwrite=self.OVERWRITE,
                        method = self.METHOD,
                        extractor_kwargs = self.EXTRACTOR_KWARGS,
                    )

                    tool.setup()
                    nevents = len(tool.event_source)
                    assert nevents == self.RUNS["nevents"][i] if _max_events is None else _max_events
                    tool.start()
                    output_containers = tool.finish(return_output_component=True)[0]
                    assert isinstance(output_containers, ChargesContainers)
                    output = output_containers.containers[self.RUNS["eventType"]]
                    assert isinstance(output, ChargesContainer)
                    # Check output in memory
                    if _events_per_slice is not None and nevents%_events_per_slice == 0 :
                        assert output.nevents is None
                    else : 
                        if _events_per_slice is None : 
                            assert output.nevents == nevents #nevents has been validated before
                        else :
                            assert output.nevents == nevents%_events_per_slice
                            
                        self.general_structure_testing(
                            output,
                            nevents if _events_per_slice is None else nevents%_events_per_slice,
                            n_pixels,
                            run_number
                            )
                        
                        
                        if _events_per_slice is None and _max_events is None:
                            #content only checked for the full run
                            assert np.min(output.ucts_timestamp) == np.uint64(
                            self.RUNS["expected_ucts_timestamp_min"][i]
                            )
                            assert np.max(output.ucts_timestamp) == np.uint64(
                                self.RUNS["expected_ucts_timestamp_max"][i]
                            )
                            assert output.charges_lg.min() == self.RUNS["charges_lg_min"][i]
                            assert output.charges_lg.max() == self.RUNS["charges_lg_max"][i]
                            assert int(10*output.charges_lg.mean()) == self.RUNS["charges_lg_mean"][i]
                            assert int(10*output.charges_lg.std()) ==self.RUNS["charges_lg_std"][i]
                            assert output.charges_hg.min() == self.RUNS["charges_hg_min"][i]
                            assert output.charges_hg.max() == self.RUNS["charges_hg_max"][i]
                            assert int(10*output.charges_hg.mean()) == self.RUNS["charges_hg_mean"][i]
                            assert int(10*output.charges_hg.std()) ==self.RUNS["charges_hg_std"][i]

                    # Check output on disk
                    assert Path(outfile).exists()
                    
                    chargesContainers =  ChargesContainers.from_hdf5(outfile)
                    ncontainers = 0
                    for container in chargesContainers : 
                        ncontainers +=1
                        assert isinstance(container, ChargesContainers)
                        output = container.containers[self.RUNS["eventType"]]
                        if _events_per_slice is None : 
                            expected_nevents = nevents
                        else : 
                            if  nevents%_events_per_slice == 0 : 
                                expected_nevents = _events_per_slice
                            else : 
                                if ncontainers == 1 : 
                                    expected_nevents = nevents%_events_per_slice
                                else :
                                    expected_nevents = _events_per_slice
                        self.general_structure_testing(
                            output,
                            expected_nevents,
                            n_pixels,
                            run_number
                            )
                    assert ncontainers == 1 if _events_per_slice is None else round(nevents / _events_per_slice)
                            
                            
