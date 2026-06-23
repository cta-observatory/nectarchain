"""
Load IRAP data into nectarchain charge and waveform files
Example of tool use below
Place IRAP .npy or .h5 files in NECTARCAMDATA/testbenchrun/RUN#####/
Tool will convert these into nectarchain readable waveform and charge container objects
available in the usual directories
There is a required dummy subarray h5 file in the same directory, this is needed to write the
correct data that NC is expecting.
"""

import fnmatch
import logging
import os

import numpy as np
from ctapipe.containers import EventType
from ctapipe.instrument import SubarrayDescription
from ctapipe.io import HDF5TableWriter

from nectarchain.data.container import (
    ChargesContainers,
    WaveformsContainer,
    WaveformsContainers,
)
from nectarchain.makers.component import ChargesComponent
from nectarchain.makers.extractor.utils import CtapipeExtractor

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers
log.setLevel(logging.INFO)


def find_all(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, pattern):
            result.append(filename)
    return result


class TestBenchNectarchainConverterTool:
    name = "TestBenchNectarchainConverterTool"

    def __init__(
        self,
        run_number,
        subarray=SubarrayDescription.from_hdf("dummy_subarray_description.h5"),
        events_per_slice=100,
        max_events=499,
        method="LocalPeakWindowSum",
        extractor_kwargs={"window_width": 16, "window_shift": 6},
    ):
        self.events_per_slice = events_per_slice
        self.max_events = max_events
        self.run_number = run_number
        self.subarray = subarray
        self.method = method
        self.extractor_kwargs = extractor_kwargs

    def extract_file_details(self):
        file_dir = os.path.join(
            os.environ["NECTARCAMDATA"], f"testbenchrun/RUN_{self.run_number}/"
        )
        files = find_all("*.npy", file_dir)
        drawers = []
        nfiles = []
        evtstotal1 = 0
        for file in files:
            split_file = file.split("_")
            drawer, fileno, evtstotal = (
                split_file[2],
                int(split_file[3].split(".")[0]),
                int(split_file[1]),
            )
            evtstotal1 = evtstotal
            drawers.append(drawer)
            nfiles.append(fileno)
        noffiles = np.max(np.array([nfiles]))
        nofdrawers = len(set(drawers))
        return evtstotal1, nofdrawers, noffiles

    def newloadpulses(self, nevts, npix, file):
        file_dir = os.path.join(
            os.environ["NECTARCAMDATA"], f"testbenchrun/RUN_{self.run_number}/"
        )
        data = np.load(file_dir + file)
        nevt = len(data[:, 0])
        nbin = int((len(data[0, :]) - 7) / 14)
        nshift = 7
        dupulse = np.zeros([7, 2, nevt, nbin])
        for idu in range(7):
            for ilghg in range(2):
                nmin = nshift + 2 * idu * nbin + ilghg * nbin
                nmax = nmin + nbin
                for ievt in range(nevt):
                    dupulse[idu, ilghg, ievt, :] = data[ievt, nmin:nmax]
        wfs_lg = dupulse[:, 0, :, :].astype(np.uint16).reshape((nevts, npix, 48))
        wfs_hg = dupulse[:, 1, :, :].astype(np.uint16).reshape((nevts, npix, 48))
        return wfs_lg, wfs_hg

    def create_irap_waveformContainer(self, wfs):
        nsamples = 48
        wfs_lg = wfs[0]
        wfs_hg = wfs[1]
        nevents = wfs_lg.shape[0]
        npix = wfs_lg.shape[1]
        rng = np.random.default_rng()
        waveform_container = WaveformsContainer(
            pixels_id=np.array([i for i in range(npix)], dtype=np.uint16),
            nevents=np.uint64(nevents),
            npixels=np.uint16(npix),
            nsamples=np.uint8(nsamples),
            wfs_hg=wfs_hg,
            wfs_lg=wfs_lg,
            run_number=np.uint16(self.run_number),
            camera="IRAP-TB",
            broken_pixels_hg=np.full(shape=(nevents, npix), fill_value=0, dtype=bool),
            broken_pixels_lg=np.full(shape=(nevents, npix), fill_value=0, dtype=bool),
            ucts_timestamp=rng.integers(
                low=0, high=nevents, size=(nevents), dtype=np.uint64
            ),
            ucts_busy_counter=rng.integers(
                low=0, high=nevents, size=(nevents), dtype=np.uint32
            ),
            ucts_event_counter=rng.integers(
                low=0, high=nevents, size=(nevents), dtype=np.uint32
            ),
            event_type=np.full(shape=(nevents), fill_value=1, dtype=np.uint8),
            event_id=np.linspace(0, nevents, nevents, dtype=np.uint32),
            trig_pattern_all=rng.integers(
                low=0, high=1, size=(nevents, npix, 4), dtype=bool
            ),
            trig_pattern=rng.integers(low=0, high=1, size=(nevents, npix), dtype=bool),
            multiplicity=rng.integers(low=0, high=1, size=(nevents), dtype=np.uint16),
        )
        waveform_container.validate()
        return waveform_container

    def create_waveformscontainers(self, wfs):
        waveform_1 = self.create_irap_waveformContainer(wfs)
        waveform_2 = self.create_irap_waveformContainer(wfs)
        waveformscontainers = WaveformsContainers()
        waveformscontainers.containers[EventType.FLATFIELD] = waveform_1
        waveformscontainers.containers[EventType.SKY_PEDESTAL] = waveform_2
        return waveformscontainers

    def create_chargescontainers(self, waveoformscontainers):
        chargesContainers = ChargesContainers()
        for key in waveoformscontainers.containers.keys():
            chargesContainers.containers[key] = ChargesComponent.create_from_waveforms(
                waveformsContainer=waveoformscontainers.containers[key],
                subarray=self.subarray,
                tel_id=0,
                method=self.method,
                extractor_kwargs=self.extractor_kwargs,
            )
        return chargesContainers

    def write_containers(self, initcontainers, file_i):
        if isinstance(initcontainers, WaveformsContainers):
            filename = (
                f"runs/waveforms/WaveformsNectarCAMCalibration_run"
                f"{initcontainers.containers[EventType.FLATFIELD].run_number:05d}"
                f"_maxevents{self.max_events}.h5"
            )
            path = os.path.join(os.environ["NECTARCAMDATA"], filename)

            writer = HDF5TableWriter(
                filename=path,
                mode="a",
                group_name=f"data/WaveformsContainer_{file_i}",
            )
            for key, container in initcontainers.containers.items():
                writer.write(table_name=f"{key.name}", containers=container)
            writer.close()

        if isinstance(initcontainers, ChargesContainers):
            str_extractor_kwargs = CtapipeExtractor.get_extractor_kwargs_str(
                method=self.method,
                extractor_kwargs=self.extractor_kwargs,
            )
            filename = (
                f"runs/charges/ChargesNectarCAMCalibration_run"
                f"{initcontainers.containers[EventType.FLATFIELD].run_number:05d}"
                f"_maxevents{self.max_events}"
                f"_{self.method}_{str_extractor_kwargs}.h5"
            )

            path = os.path.join(os.environ["NECTARCAMDATA"], filename)

            writer = HDF5TableWriter(
                filename=path,
                mode="a",
                group_name=f"data/ChargesContainer_{file_i}",
            )
            for key, container in initcontainers.containers.items():
                writer.write(table_name=f"{key.name}", containers=container)
            writer.close()

    def run_tool(self):
        evtspfile, nofdrawers, noffiles = self.extract_file_details()
        if self.max_events is None:
            self.max_events = evtspfile * noffiles

        if self.events_per_slice > evtspfile:
            raise Exception(
                "events per slice must be equal to or less than the events per test bench file"
            )
        if self.events_per_slice > self.max_events:
            raise Exception("events per slice must be less than or equal to max events")

        # determine number of files that need to be written
        if self.max_events >= noffiles * evtspfile:
            self.max_events = noffiles * evtspfile
        self.max_events = self.max_events - (self.max_events % self.events_per_slice)
        nfilestowrite = int(self.max_events / self.events_per_slice)

        input_buffer = []
        written_total = 0
        output_file_i = 0

        for fileno in range(noffiles):
            log.info(f"loading IRAP file {fileno}")
            all_drawer_wfs = []
            for d in range(nofdrawers):
                file = f"Nevts_{evtspfile}_Drawer{d}_{fileno+1:04d}.npy"
                # wfs for a single drawer of seven pixels and they are less than the
                # required number that we need per slice.
                # go through all drawers and append to some file
                wfs_lg, wfs_hg = self.newloadpulses(evtspfile, 7, file)
                all_drawer_wfs.append(np.array([wfs_lg, wfs_hg]))

            all_drawer_wfs = np.array(all_drawer_wfs).transpose(1, 2, 0, 3, 4)
            all_drawer_wfs = all_drawer_wfs.reshape(2, evtspfile, nofdrawers * 7, 48)
            input_buffer.append(all_drawer_wfs)
            combined = np.concatenate(input_buffer, axis=1)

            while (
                combined.shape[1] >= self.events_per_slice
                and written_total < self.max_events
            ):
                log.info(f"writing nectarchain file {output_file_i}")
                to_write = combined[:, : self.events_per_slice]
                wfscons = self.create_waveformscontainers(to_write)
                charscons = self.create_chargescontainers(wfscons)
                self.write_containers(wfscons, output_file_i)
                self.write_containers(charscons, output_file_i)
                output_file_i += 1
                written_total += self.events_per_slice
                combined = combined[:, self.events_per_slice :]

            input_buffer = [combined]

            if written_total >= self.max_events:
                break

        if input_buffer[0].shape[1] > 0 and written_total < self.max_events:
            remainder = input_buffer[0][:, : self.max_events - written_total]
            wfscons = self.create_waveformscontainers(remainder)
            charscons = self.create_chargescontainers(wfscons)
            self.write_containers(wfscons, output_file_i)
            self.write_containers(charscons, output_file_i)


tool = TestBenchNectarchainConverterTool(
    run_number="00620",
    events_per_slice=100,
    max_events=500,
)

tool.run_tool()
