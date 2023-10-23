import glob
import logging
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from ctapipe.instrument import CameraGeometry, SubarrayDescription
from ctapipe.visualization import CameraDisplay
from ctapipe_io_nectarcam import NectarCAMEventSource
from matplotlib import pyplot as plt

from .utils import DataManagement

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["WaveformsContainer", "WaveformsContainers"]


class WaveformsContainer:
    """class used to load run and load waveforms from r0 data"""

    TEL_ID = 0
    CAMERA = CameraGeometry.from_name("NectarCam-003")

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        return obj

    def __init__(
        self,
        run_number: int,
        max_events: int = None,
        nevents: int = -1,
        run_file=None,
        init_arrays: bool = False,
    ):
        """construtor

        Args:
            run_number (int): id of the run to be loaded
            maxevents (int, optional): max of events to be loaded.
            Defaults to 0, to load everythings.
            nevents (int, optional) : number of events in run if known
            (parameter used to save computing time)
            merge_file (optional) : if True will load all fits.fz files of
            the run, else merge_file can be integer to select the run fits.fz
            file according to this number

        """

        self.__run_number = run_number
        self.__run_file = run_file
        self.__max_events = max_events

        self.__reader = WaveformsContainer.load_run(
            run_number, max_events, run_file=run_file
        )

        # from reader members
        self.__npixels = self.__reader.camera_config.num_pixels
        self.__nsamples = self.__reader.camera_config.num_samples
        self.__geometry = self.__reader.subarray.tel[WaveformsContainer.TEL_ID].camera
        self.__subarray = self.__reader.subarray
        self.__pixels_id = self.__reader.camera_config.expected_pixels_id

        # set camera properties
        log.info(f"N pixels : {self.npixels}")

        # run properties
        if nevents != -1:
            self.__nevents = (
                nevents if max_events is None else min(max_events, nevents)
            )  # self.check_events()
            self.__reader = None
        else:
            self.__nevents = self.check_events()

        log.info(f"N_events : {self.nevents}")

        if init_arrays:
            self.__init_arrays()

    def __init_arrays(self, **kwargs):
        log.debug("creation of the EventSource reader")
        self.__reader = WaveformsContainer.load_run(
            self.__run_number, self.__max_events, run_file=self.__run_file
        )

        log.debug("create wfs, ucts, event properties and triger pattern arrays")
        # define zeros members which will be filled therafter
        self.wfs_hg = np.zeros(
            (self.nevents, self.npixels, self.nsamples), dtype=np.uint16
        )
        self.wfs_lg = np.zeros(
            (self.nevents, self.npixels, self.nsamples), dtype=np.uint16
        )
        self.ucts_timestamp = np.zeros((self.nevents), dtype=np.uint64)
        self.ucts_busy_counter = np.zeros((self.nevents), dtype=np.uint32)
        self.ucts_event_counter = np.zeros((self.nevents), dtype=np.uint32)
        self.event_type = np.zeros((self.nevents), dtype=np.uint8)
        self.event_id = np.zeros((self.nevents), dtype=np.uint32)
        self.trig_pattern_all = np.zeros(
            (self.nevents, self.CAMERA.n_pixels, 4), dtype=bool
        )
        # self.trig_pattern = np.zeros((self.nevents,self.npixels),dtype = bool)
        # self.multiplicity = np.zeros((self.nevents,self.npixels),dtype = np.uint16)

        self.__broken_pixels_hg = np.zeros((self.npixels), dtype=bool)
        self.__broken_pixels_lg = np.zeros((self.npixels), dtype=bool)

    def __compute_broken_pixels(self, **kwargs):
        log.warning("computation of broken pixels is not yet implemented")
        self.__broken_pixels_hg = np.zeros((self.npixels), dtype=bool)
        self.__broken_pixels_lg = np.zeros((self.npixels), dtype=bool)

    @staticmethod
    def load_run(run_number: int, max_events: int = None, run_file=None):
        """Static method to load from $NECTARCAMDATA directory data for
        specified run with max_events

        Args:
            run_number (int): run_id
            maxevents (int, optional): max of events to be loaded.
            Defaults to 0, to load everythings.
            merge_file (optional) : if True will load all fits.fz files of
            the run, else merge_file can be integer to select the run
            fits.fz file according to this number
        Returns:
            List[ctapipe_io_nectarcam.NectarCAMEventSource]: List of
            EventSource for each run files
        """
        if run_file is None:
            generic_filename, _ = DataManagement.findrun(run_number)
            log.info(f"{str(generic_filename)} will be loaded")
            eventsource = NectarCAMEventSource(
                input_url=generic_filename, max_events=max_events
            )
        else:
            log.info(f"{run_file} will be loaded")
            eventsource = NectarCAMEventSource(
                input_url=run_file, max_events=max_events
            )
        return eventsource

    def check_events(self):
        """method to check triggertype of events and counting
         number of events in all readers it prints the trigger type when
         trigger type change over events (to not write one line by events)

        Returns:
            int : number of events
        """
        log.info("checking trigger type")
        has_changed = True
        previous_trigger = None
        n_events = 0
        for i, event in enumerate(self.__reader):
            if previous_trigger is not None:
                has_changed = previous_trigger.value != event.trigger.event_type.value
            previous_trigger = event.trigger.event_type
            if has_changed:
                log.info(
                    f"event {i} is {previous_trigger} : {previous_trigger.value} "
                    f"trigger type"
                )
            n_events += 1
        return n_events

    def load_wfs(self, start=None, end=None, compute_trigger_patern=False):
        """mathod to extract waveforms data from the EventSource

        Args:
            compute_trigger_patern (bool, optional): To recompute on
            our side the trigger patern. Defaults to False.
        """
        if not (hasattr(self, "wfs_hg")):
            self.__init_arrays()

        wfs_hg_tmp = np.zeros((self.npixels, self.nsamples), dtype=np.uint16)
        wfs_lg_tmp = np.zeros((self.npixels, self.nsamples), dtype=np.uint16)

        # n_traited_events = 0
        event_start, event_end = start, end
        log.info(f" Loading Events ID from {event_start} to {event_end}")

        for i, event in enumerate(self.__reader):
            if i % 5000 == 0:
                log.info(f"reading event number {i}")

            if (
                event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.event_id
                >= event_start
                and event.nectarcam.tel[WaveformsContainer.TEL_ID].evt.event_id
                < event_end
            ):
                self.event_id[i] = np.uint16(event.index.event_id)
                self.ucts_timestamp[i] = event.nectarcam.tel[
                    WaveformsContainer.TEL_ID
                ].evt.ucts_timestamp
                self.event_type[i] = event.trigger.event_type.value
                self.ucts_busy_counter[i] = event.nectarcam.tel[
                    WaveformsContainer.TEL_ID
                ].evt.ucts_busy_counter
                self.ucts_event_counter[i] = event.nectarcam.tel[
                    WaveformsContainer.TEL_ID
                ].evt.ucts_event_counter

                self.trig_pattern_all[i] = event.nectarcam.tel[
                    WaveformsContainer.TEL_ID
                ].evt.trigger_pattern.T

                for pix in range(self.npixels):
                    wfs_lg_tmp[pix] = event.r0.tel[0].waveform[1, self.pixels_id[pix]]
                    wfs_hg_tmp[pix] = event.r0.tel[0].waveform[0, self.pixels_id[pix]]

                self.wfs_hg[i] = wfs_hg_tmp
                self.wfs_lg[i] = wfs_lg_tmp

        self.__compute_broken_pixels()

        # if compute_trigger_patern and np.max(self.trig_pattern) == 0:
        #    self.compute_trigger_patern()

    def write(self, path: str, start=None, end=None, block=None, **kwargs):
        """method to write in an output FITS file the
        WaveformsContainer. Two files are created, one FITS
        representing the data and one HDF5 file representing
        the subarray configuration

        Args:
            path (str): the directory where you want to save data
            start (int): Start of event_id. Default is None
            end (int): End of event_is. Default is None
            block (int): Block number of SPE-WT scan. Default is None
        """
        suffix = kwargs.get("suffix", "")
        if suffix != "":
            suffix = f"_{suffix}"

        log.info(f"saving in {path}")
        os.makedirs(path, exist_ok=True)

        hdr = fits.Header()
        hdr["RUN"] = self.__run_number
        hdr["NEVENTS"] = self.nevents
        hdr["NPIXELS"] = self.npixels
        hdr["NSAMPLES"] = self.nsamples
        hdr["SUBARRAY"] = self.subarray.name

        self.subarray.to_hdf(
            f"{Path(path)}/subarray_run{self.run_number}{suffix}.hdf5",
            overwrite=kwargs.get("overwrite", False),
        )

        hdr["COMMENT"] = (
            f"The waveforms containeur for run {self.__run_number} : "
            f"primary is the pixels id, 2nd HDU : high gain waveforms, "
            f"3rd HDU : low gain waveforms, 4th HDU :  event properties "
            f"and 5th HDU trigger paterns."
        )

        primary_hdu = fits.PrimaryHDU(self.pixels_id, header=hdr)

        wfs_hg_hdu = fits.ImageHDU(self.wfs_hg[start:end], name="HG Waveforms")
        wfs_lg_hdu = fits.ImageHDU(self.wfs_lg[start:end], name="LG Waveforms")

        col1 = fits.Column(array=self.event_id[start:end], name="event_id", format="1J")
        col2 = fits.Column(
            array=self.event_type[start:end], name="event_type", format="1I"
        )
        col3 = fits.Column(
            array=self.ucts_timestamp[start:end], name="ucts_timestamp", format="1K"
        )
        col4 = fits.Column(
            array=self.ucts_busy_counter[start:end],
            name="ucts_busy_counter",
            format="1J",
        )
        col5 = fits.Column(
            array=self.ucts_event_counter[start:end],
            name="ucts_event_counter",
            format="1J",
        )
        col6 = fits.Column(
            array=self.multiplicity[start:end], name="multiplicity", format="1I"
        )

        coldefs = fits.ColDefs([col1, col2, col3, col4, col5, col6])
        event_properties = fits.BinTableHDU.from_columns(
            coldefs, name="event properties"
        )

        col1 = fits.Column(
            array=self.trig_pattern_all[start:end],
            name="trig_pattern_all",
            format=f"{4 * self.CAMERA.n_pixels}L",
            dim=f"({self.CAMERA.n_pixels},4)",
        )
        col2 = fits.Column(
            array=self.trig_pattern[start:end],
            name="trig_pattern",
            format=f"{self.CAMERA.n_pixels}L",
        )
        coldefs = fits.ColDefs([col1, col2])
        trigger_patern = fits.BinTableHDU.from_columns(coldefs, name="trigger patern")

        hdul = fits.HDUList(
            [primary_hdu, wfs_hg_hdu, wfs_lg_hdu, event_properties, trigger_patern]
        )
        try:
            hdul.writeto(
                Path(path)
                / f"waveforms_run{self.run_number}{suffix}_bloc_{block}.fits",
                overwrite=kwargs.get("overwrite", False),
            )
            log.info(
                f"run saved in {Path(path)}/waveforms_run{self.run_number}{suffix}.fits"
            )
        except OSError as e:
            log.warning(e)
        except Exception as e:
            log.error(e, exc_info=True)
            raise e

    @staticmethod
    def load(path: str):
        """load WaveformsContainer from FITS file previously
        written with WaveformsContainer.write() method
        Note : 2 files are loaded, the FITS one representing
        the waveforms data and a HDF5 file representing the subarray
        configuration. This second file has to be next to the FITS file.

        Args:
            path (str): path of the FITS file

        Returns:
            WaveformsContainer: WaveformsContainer instance
        """
        log.info(f"loading from {path}")
        with fits.open(Path(path)) as hdul:
            cls = WaveformsContainer.__new__(WaveformsContainer)

            cls.__run_number = hdul[0].header["RUN"]
            cls.__nevents = hdul[0].header["NEVENTS"]
            cls.__npixels = hdul[0].header["NPIXELS"]
            cls.__nsamples = hdul[0].header["NSAMPLES"]

            cls.__subarray = SubarrayDescription.from_hdf(
                Path(path.replace("waveforms_", "subarray_").replace("fits", "hdf5"))
            )

            cls.__pixels_id = hdul[0].data
            cls.wfs_hg = hdul[1].data
            cls.wfs_lg = hdul[2].data

            table_prop = hdul[3].data
            cls.event_id = table_prop["event_id"]
            cls.event_type = table_prop["event_type"]
            cls.ucts_timestamp = table_prop["ucts_timestamp"]
            cls.ucts_busy_counter = table_prop["ucts_busy_counter"]
            cls.ucts_event_counter = table_prop["ucts_event_counter"]

            table_trigger = hdul[4].data
            cls.trig_pattern_all = table_trigger["trig_pattern_all"]

        cls.__compute_broken_pixels()

        return cls

    def compute_trigger_patern(self):
        """(preliminary) function to compute the trigger patern"""
        # mean.shape nevents * npixels
        mean, std = np.mean(self.wfs_hg, axis=2), np.std(self.wfs_hg, axis=2)
        self.trig_pattern = self.wfs_hg.max(axis=2) > (mean + 3 * std)

    # methods used to display
    def display(self, evt, cmap="gnuplot2"):
        """plot camera display

        Args:
            evt (int): event index
            cmap (str, optional): colormap. Defaults to 'gnuplot2'.

        Returns:
            CameraDisplay: thoe cameraDisplay plot
        """
        image = self.wfs_hg.sum(axis=2)
        disp = CameraDisplay(
            geometry=WaveformsContainer.CAMERA, image=image[evt], cmap=cmap
        )
        disp.add_colorbar()
        return disp

    def plot_waveform_hg(self, evt, **kwargs):
        """plot the waveform of the evt

        Args:
            evt (int): the event index

        Returns:
            tuple: the figure and axes
        """
        if "figure" in kwargs.keys() and "ax" in kwargs.keys():
            fig = kwargs.get("figure")
            ax = kwargs.get("ax")
        else:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.wfs_hg[evt].T)
        return fig, ax

    def select_waveforms_hg(self, pixel_id: np.ndarray):
        """method to extract waveforms HG from a list of pixel ids
        The output is the waveforms HG with a shape following the
        size of the input pixel_id argument
        Pixel in pixel_id which are not present in the
        WaveformsContaineur pixels_id are skipped
        Args:
            pixel_id (np.ndarray): array of pixel ids you want to
            extract the waveforms
        Returns:
            (np.ndarray): waveforms array in the order of specified pixel_id
        """
        mask_contain_pixels_id = np.array(
            [pixel in self.pixels_id for pixel in pixel_id], dtype=bool
        )
        for pixel in pixel_id[~mask_contain_pixels_id]:
            log.warning(
                f"You asked for pixel_id {pixel} but it is not present "
                f"in this WaveformsContainer, skip this one"
            )
        res = np.array(
            [
                self.wfs_hg[:, np.where(self.pixels_id == pixel)[0][0], :]
                for pixel in pixel_id[mask_contain_pixels_id]
            ]
        )
        res = res.transpose(res.shape[1], res.shape[0], res.shape[2])
        return res

    def select_waveforms_lg(self, pixel_id: np.ndarray):
        """method to extract waveforms LG from a list of pixel ids
        The output is the waveforms LG with a shape following the
        size of the input pixel_id argument
        Pixel in pixel_id which are not present in the
        WaveformsContaineur pixels_id are skipped

        Args:
            pixel_id (np.ndarray): array of pixel ids you want to
            extract the waveforms

        Returns:
            (np.ndarray): waveforms array in the order of specified pixel_id
        """
        mask_contain_pixels_id = np.array(
            [pixel in self.pixels_id for pixel in pixel_id], dtype=bool
        )
        for pixel in pixel_id[~mask_contain_pixels_id]:
            log.warning(
                f"You asked for pixel_id {pixel} but it is not present "
                f"in this WaveformsContainer, skip this one"
            )
        res = np.array(
            [
                self.wfs_lg[:, np.where(self.pixels_id == pixel)[0][0], :]
                for pixel in pixel_id[mask_contain_pixels_id]
            ]
        )
        res = res.transpose(res.shape[1], res.shape[0], res.shape[2])
        return res

    @property
    def _run_file(self):
        return self.__run_file

    @property
    def _max_events(self):
        return self.__max_events

    @property
    def reader(self):
        return self.__reader

    @property
    def npixels(self):
        return self.__npixels

    @property
    def nsamples(self):
        return self.__nsamples

    @property
    def geometry(self):
        return self.__geometry

    @property
    def subarray(self):
        return self.__subarray

    @property
    def pixels_id(self):
        return self.__pixels_id

    @property
    def nevents(self):
        return self.__nevents

    @property
    def run_number(self):
        return self.__run_number

    @property
    def broken_pixels_hg(self):
        return self.__broken_pixels_hg

    @property
    def broken_pixels_lg(self):
        return self.__broken_pixels_lg

    # physical properties
    @property
    def multiplicity(self):
        return np.uint16(np.count_nonzero(self.trig_pattern, axis=1))

    @property
    def trig_pattern(self):
        return self.trig_pattern_all.any(axis=1)


class WaveformsContainers:
    """This class is to be used for computing waveforms of a
    run treating run files one by one"""

    def __new__(cls, *args, **kwargs):
        """base class constructor

        Returns:
            WaveformsContainers : the new object
        """
        obj = object.__new__(cls)
        return obj

    def __init__(
        self, run_number: int, max_events: int = None, init_arrays: bool = False
    ):
        """initialize the waveformsContainer list inside the main object

        Args:
            run_number (int): the run number
            max_events (int, optional): max events we want to read. Defaults to None.
        """
        log.info("Initialization of WaveformsContainers")
        _, filenames = DataManagement.findrun(run_number)
        self.waveformsContainer = []
        self.__nWaveformsContainer = 0
        for i, file in enumerate(filenames):
            self.waveformsContainer.append(
                WaveformsContainer(
                    run_number,
                    max_events=max_events,
                    run_file=file,
                    init_arrays=init_arrays,
                )
            )
            self.__nWaveformsContainer += 1
            if not (max_events is None):
                max_events -= self.waveformsContainer[i].nevents
            log.info(f"WaveformsContainer number {i} is created")
            if not (max_events is None) and max_events <= 0:
                break

    def load_wfs(self, compute_trigger_patern=False, index: int = None):
        """load waveforms from the Eventsource reader

        Args:
            compute_trigger_patern (bool, optional): to compute
            manually the trigger patern. Defaults to False.
            index (int, optional): to only load waveforms from
            EventSource for the waveformsContainer at index, this parameters
            can be used to load, write or perform some work step by step.
            Thus all the event are not nessessary loaded at the same time
            In such case memory can be saved. Defaults to None.

        Raises:
            IndexError: the index is > nWaveformsContainer or < 0
        """
        if index is None:
            for i in range(self.__nWaveformsContainer):
                self.waveformsContainer[i].load_wfs(
                    compute_trigger_patern=compute_trigger_patern
                )
        else:
            if index < self.__nWaveformsContainer and index >= 0:
                self.waveformsContainer[index].load_wfs(
                    compute_trigger_patern=compute_trigger_patern
                )
            else:
                raise IndexError(
                    f"index must be >= 0 and < {self.__nWaveformsContainer}"
                )

    def write(self, path: str, index: int = None, **kwargs):
        """method to write on disk all the waveformsContainer in the list

        Args:
            path (str): path where to save the waveforms
            index (int, optional): index of the waveforms list you want
            to write. Defaults to None.

        Raises:
            IndexError: the index is > nWaveformsContainer or < 0
        """
        if index is None:
            for i in range(self.__nWaveformsContainer):
                self.waveformsContainer[i].write(path, suffix=f"{i:04d}", **kwargs)
        else:
            if index < self.__nWaveformsContainer and index >= 0:
                self.waveformsContainer[index].write(
                    path, suffix=f"{index:04d}", **kwargs
                )
            else:
                raise IndexError(
                    f"index must be >= 0 and < {self.__nWaveformsContainer}"
                )

    @staticmethod
    def load(path: str):
        """method to load the waveforms list from splited fits files

        Args:
            path (str): path where data should be, it contain filename
            without extension

        Raises:
            e: File not found

        Returns:
            WaveformsContainers:
        """
        log.info(f"loading from {path}")
        files = glob.glob(f"{path}_*.fits")
        if len(files) == 0:
            e = FileNotFoundError(f"no files found corresponding to {path}_*.fits")
            log.error(e)
            raise e
        else:
            cls = WaveformsContainers.__new__(WaveformsContainers)
            cls.waveformsContainer = []
            cls.__nWaveformsContainer = len(files)
            for file in files:
                cls.waveformsContainer.append(WaveformsContainer.load(file))
            return cls

    @property
    def nWaveformsContainer(self):
        """getter giving the number of waveformsContainer into the
        waveformsContainers instance

        Returns:
            int: the number of waveformsContainer
        """
        return self.__nWaveformsContainer

    @property
    def nevents(self):
        """getter giving the number of events in the waveformsContainers instance"""
        return np.sum(
            [
                self.waveformsContainer[i].nevents
                for i in range(self.__nWaveformsContainer)
            ]
        )

    def append(self, waveformsContainer: WaveformsContainer):
        """method to append WaveformsContainer to the waveforms list

        Args:
            waveformsContainer (WaveformsContainer): the waveformsContainer to stack
        """
        self.waveformsContainer.append(waveformsContainer)
        self.__nWaveformsContainer += 1
