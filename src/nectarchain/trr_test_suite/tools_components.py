import os
import pathlib

import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from ctapipe.containers import EventType, Field
from ctapipe.core.traits import ComponentNameList, Integer
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer

from nectarchain.data.container import NectarCAMContainer
from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent
from nectarchain.utils.constants import GAIN_DEFAULT


# overriding so we can have maxevents in the path
def _init_output_path(self):
    """Initializes the output path for the NectarCAMCalibrationTool.

    If `max_events` is `None`, the output file name will be in the format\
        `{self.name}_run{self.run_number}.h5`. Otherwise, the file name will\
            be in the format\
                `{self.name}_run{self.run_number}_maxevents{self.max_events}.h5`.

    The output path is constructed by joining the `NECTARCAMDATA` environment variable\
        (or `/tmp` if not set) with the `tests` subdirectory and the generated\
            file name.
    """

    if self.max_events is None:
        filename = f"{self.name}_run{self.run_number}.h5"
    else:
        filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}.h5"
    self.output_path = pathlib.Path(
        f"{os.environ.get('NECTARCAMDATA','/tmp')}/tests/{filename}"
    )


EventsLoopNectarCAMCalibrationTool._init_output_path = _init_output_path


class LinearityTestTool(EventsLoopNectarCAMCalibrationTool):
    """This class, `LinearityTestTool`, is a subclass of
    `EventsLoopNectarCAMCalibrationTool`. It is responsible for performing a linearity
    test on NectarCAM data. The class has a `componentsList` attribute that specifies
    the list of NectarCAM components to be applied.

    The `finish` method is the main functionality of this class. It reads the charge\
        data from the output file, calculates the mean charge, standard deviation,\
            and standard error for both the high gain and low gain channels, and\
                returns these values. This information can be used to assess\
                    the linearity of the NectarCAM system.
    """

    name = "LinearityTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        output = super().finish(return_output_component=True, *args, **kwargs)

        charge_container = output[0].containers[EventType.FLATFIELD]

        mean_charge = [0, 0]  # per channel
        std_charge = [0, 0]
        std_err = [0, 0]

        charge_hg = charge_container["charges_hg"]
        charge_lg = charge_container["charges_lg"]
        npixels = charge_container["npixels"]

        charge_pe_hg = np.array(charge_hg) / GAIN_DEFAULT
        charge_pe_lg = np.array(charge_lg) / GAIN_DEFAULT

        for channel, charge in enumerate([charge_pe_hg, charge_pe_lg]):
            pix_mean_charge = np.mean(charge, axis=0)  # in pe

            pix_std_charge = np.std(charge, axis=0)

            # average of all pixels
            mean_charge[channel] = np.mean(pix_mean_charge)

            std_charge[channel] = np.mean(pix_std_charge)
            # for the charge resolution
            std_err[channel] = np.std(pix_std_charge)

        return mean_charge, std_charge, std_err, npixels


class TimingResolutionTestTool(EventsLoopNectarCAMCalibrationTool):
    """This class, `TimingResolutionTestTool`, is a subclass of
    `EventsLoopNectarCAMCalibrationTool` and is used to perform timing resolution tests
    on NectarCAM data. It reads the output data from the `ToMContainer` dataset and
    processes the charge, timing, and event information to calculate the timing
    resolution and mean charge in photoelectrons.

    The `finish()` method is the main entry point for this tool. It reads the output
    data from the HDF5 file, filters the data to remove cosmic ray events, and then
    calculates the timing resolution and mean charge per photoelectron. The timing
    resolution is calculated using a weighted mean and variance approach, with an option
    to use a bootstrapping method to estimate the error on the RMS value.

    The method returns the RMS of the timing resolution, the error on the RMS, and the
    mean charge in photoelectrons.
    """

    name = "TimingResolutionTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, bootstrap=False, *args, **kwargs):
        output = super().finish(return_output_component=True, *args, **kwargs)

        # Default runs use a laser source and apply a subarray trigger
        # Newer runs use flat-field events
        try:
            charge_container = output[0].containers[EventType.SUBARRAY]
        except Exception:
            charge_container = output[0].containers[EventType.FLATFIELD]

        charge_all = charge_container["charges_hg"]
        tom_no_fit_all = charge_container["peak_hg"]
        npixels = charge_container["npixels"]
        good_evts = np.where(
            np.max(charge_all, axis=1) < 10 * np.mean(charge_all, axis=1)
        )[0]

        charge = charge_all[good_evts]
        mean_charge_pe = np.mean(np.mean(charge, axis=0)) / GAIN_DEFAULT

        # tom_sigma = np.array(tom_sigma_all[good_evts]).reshape(len(good_evts),
        # output[0].npixels)
        tom_no_fit = np.array(tom_no_fit_all[good_evts]).reshape(
            len(good_evts), npixels
        )
        # print(tom_no_fit)
        # print(tom_no_fit)

        # rms_mu = np.zeros(output[0].npixels)
        rms_no_fit = np.zeros(npixels)

        # rms_mu_err = np.zeros(output[0].npixels)
        rms_no_fit_err = np.zeros(npixels)

        # bootstrapping method

        for pix in range(npixels):
            for tom, rms, err in zip(
                [tom_no_fit[:, pix]], [rms_no_fit], [rms_no_fit_err]
            ):
                tom_pos = tom[tom > 20]
                boot_rms = []

                sample = tom_pos[tom_pos < 32]
                # print(sample)
                bins = np.linspace(20, 32, 50)
                hist_values, bin_edges = np.histogram(sample, bins=bins)

                # Compute bin centers
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                if bootstrap:
                    # print(len(sample))
                    if len(sample) != 0:
                        for _ in range(1000):
                            bootsample = np.random.choice(
                                sample, size=int(3 / 4 * (len(sample))), replace=True
                            )
                            #             print(len(bootsample), bootsample.mean(),
                            # bootsample.std())
                            boot_rms.append(bootsample.std())
                            # simulated mean of rms
                        bootrms_mean = np.mean(boot_rms)
                        # simulated standard deviation of rms
                        bootrms_std = np.std(boot_rms)
                    else:
                        bootrms_std = 0
                        bootrms_mean = 0
                    # print(bootrms_std)
                    err[pix] = bootrms_std
                    rms[pix] = bootrms_mean

                else:
                    try:
                        weighted_mean = np.average(bin_centers, weights=hist_values)
                        # print("Weighted Mean:", weighted_mean)

                        # Compute weighted variance
                        weighted_variance = np.average(
                            (bin_centers - weighted_mean) ** 2, weights=hist_values
                        )
                        # print("Weighted Variance:", weighted_variance)

                        # Compute RMS value (Standard deviation)
                        rms[pix] = np.sqrt(weighted_variance)
                        # print("RMS:", rms[pix])

                        # Compute the total number of data points (sum of histogram
                        # values, i.e. N)
                        N = np.sum(hist_values)
                        # print("Total number of events (N):", N)

                        # Error on the standard deviation
                        err[pix] = rms[pix] / np.sqrt(2 * N)
                        # print("Error on RMS:", err[pix])
                    except Exception:
                        # no data
                        rms[pix] = np.nan
                        err[pix] = np.nan

        return rms_no_fit, rms_no_fit_err, mean_charge_pe


class ToMPairsTool(EventsLoopNectarCAMCalibrationTool):
    """This class, `ToMPairsTool`, is an `EventsLoopNectarCAMCalibrationTool`\
        that is used to process ToM (Time of maximum) data from NectarCAM.

    The `finish` method has the following functionalities:

    - It reads in ToM data from an HDF5 file and applies a transit time correction to\
        the ToM values using a provided lookup table.
    - It calculates the time difference between ToM pairs for both corrected and\
        uncorrected ToM values.
    - It returns the uncorrected ToM values, the corrected ToM values, the pixel IDs,\
        and the time difference calculations for the uncorrected and corrected\
            ToM values.

    The class has several configurable parameters, including the list of NectarCAM\
        components to apply, the maximum number of events to process, and the output\
            file path.
    """

    name = "ToMPairsTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        output = super().finish(return_output_component=True, *args[1:], **kwargs)

        tt_path = args[0]
        pmt_tt = pd.read_csv(tt_path)["tom_pmttt_delta_correction"].values

        # Default runs use a laser source and apply a subarray trigger
        # Newer runs use flat-field events
        try:
            charge_container = output[0].containers[EventType.SUBARRAY]
        except Exception:
            charge_container = output[0].containers[EventType.FLATFIELD]

        n_pixels = charge_container["npixels"]
        pixels_id = charge_container["pixels_id"]
        tom_no_fit_all = charge_container["peak_hg"]

        pixels_id = np.array(pixels_id, dtype=np.uint16)

        tom_no_fit = np.array(tom_no_fit_all, dtype=np.float64)
        tom_corrected = -np.ones(tom_no_fit.shape, dtype=np.float128)

        valid_mask = (tom_no_fit > 0) & (tom_no_fit < 60)
        corrections = pmt_tt[pixels_id]
        tom_corrected = tom_no_fit - corrections[None, :]
        tom_corrected[~valid_mask] = -1

        # Indices of all pixel pairs (i < j)
        pair_indices = np.triu_indices(n_pixels, k=1)
        pixel_pairs = list(zip(pair_indices[0], pair_indices[1]))

        # Compute all pairwise differences at once
        diff_no_corr = tom_no_fit[:, :, None] - tom_no_fit[:, None, :]
        diff_corr = tom_corrected[:, :, None] - tom_corrected[:, None, :]

        # Extract only upper-triangular pairs -> shape (n_pairs, n_events)
        dt_no_correction = diff_no_corr[:, pair_indices[0], pair_indices[1]].T
        dt_corrected = diff_corr[:, pair_indices[0], pair_indices[1]].T

        valid_no_corr = valid_mask[:, :, None] & valid_mask[:, None, :]
        valid_corr = (tom_corrected > 0) & (tom_corrected < 60)
        valid_corr = valid_corr[:, :, None] & valid_corr[:, None, :]

        valid_no_corr_pairs = valid_no_corr[:, pair_indices[0], pair_indices[1]].T
        valid_corr_pairs = valid_corr[:, pair_indices[0], pair_indices[1]].T

        dt_no_correction[~valid_no_corr_pairs] = np.nan
        dt_corrected[~valid_corr_pairs] = np.nan

        return (
            tom_no_fit,
            tom_corrected,
            pixels_id,
            dt_no_correction,
            dt_corrected,
            pixel_pairs,
        )


class UCTSContainer(NectarCAMContainer):
    """Defines the fields for the UCTSContainer class, which is used to store various
    data related to UCTS events.

    The fields include:
    - `run_number`: The run number associated with the waveforms.
    - `npixels`: The number of effective pixels.
    - `pixels_id`: The IDs of the pixels.
    - `ucts_timestamp`: The UCTS timestamp of the events.
    - `event_type`: The trigger event type.
    - `event_id`: The IDs of the events.
    - `ucts_busy_counter`: The UCTS busy counter.
    - `ucts_event_counter`: The UCTS event counter.
    """

    run_number = Field(
        type=np.uint16,
        description="run number associated to the waveforms",
    )
    npixels = Field(
        type=np.uint16,
        description="number of effective pixels",
    )
    pixels_id = Field(type=np.ndarray, dtype=np.uint16, ndim=1, description="pixel ids")
    ucts_timestamp = Field(
        type=np.ndarray, dtype=np.uint64, ndim=1, description="events ucts timestamp"
    )
    mean_event_charge = Field(
        type=np.ndarray,
        dtype=np.uint32,
        ndim=1,
        description="average pixel charge for event",
    )
    event_type = Field(
        type=np.ndarray, dtype=np.uint8, ndim=1, description="trigger event type"
    )
    event_id = Field(type=np.ndarray, dtype=np.uint32, ndim=1, description="event ids")
    ucts_busy_counter = Field(
        type=np.ndarray, dtype=np.uint32, ndim=1, description="busy counter"
    )
    ucts_event_counter = Field(
        type=np.ndarray, dtype=np.uint32, ndim=1, description="event counter"
    )


class UCTSComp(NectarCAMComponent):
    """The `__init__` method initializes the `UCTSComp` class, which is a
    NectarCAMComponent. It sets up several member variables to store UCTS related data,
    such as timestamps, event types, event IDs, busy counters, and event counters.

    The `__call__` method is called for each event, and it appends the UCTS-related\
        data from the event to the corresponding member variables.

    The `finish` method creates and returns a `UCTSContainer` object, which is a\
        container for the UCTS-related data that was collected during the event loop.
    """

    window_shift = Integer(
        default_value=6,
        help="the time in ns before the peak to extract charge",
    ).tag(config=True)

    window_width = Integer(
        default_value=16,
        help="the duration of the extraction window in ns",
    ).tag(config=True)

    def __init__(
        self, subarray, config=None, parent=None, excl_muons=None, *args, **kwargs
    ):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        # If you want you can add here members of MyComp, they will contain interesting
        # quantity during the event loop process

        self.__ucts_timestamp = []
        self.__event_type = []
        self.__event_id = []
        self.__ucts_busy_counter = []
        self.__ucts_event_counter = []
        self.excl_muons = None
        self.__mean_event_charge = []

    # This method need to be defined !
    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        take_event = True

        # exclude muon events for the trigger timing test
        if self.excl_muons:
            wfs = []
            wfs.append(event.r0.tel[0].waveform[constants.HIGH_GAIN][self.pixels_id])
            # print(self.pixels_id)
            wf = np.array(wfs[0])
            # print(wf.shape)
            index_peak = np.argmax(wf, axis=1)  # tom per event/pixel
            # print(wf[100])
            # print(index_peak[100])
            index_peak[index_peak < 20] = 20
            index_peak[index_peak > 40] = 40
            signal_start = index_peak - self.window_shift
            # signal_stop = index_peak + self.window_width - self.window_shift
            # print(index_peak)
            # print(signal_start)
            # print(signal_stop)
            chg = np.zeros(len(self.pixels_id))

            ped = np.array(
                [
                    np.mean(wf[pix, 0 : signal_start[pix]])
                    for pix in range(len(self.pixels_id))
                ]
            )

            for pix in range(len(self.pixels_id)):
                # print("iterating through pixels")
                # print("pix", pix)

                y = (
                    wf[pix] - ped[pix]
                )  # np.maximum(wf[pix] - ped[pix],np.zeros(len(wf[pix])))
                charge_sum = y[
                    signal_start[pix] : signal_start[pix] + self.window_width
                ].sum()
                # print(charge_sum)
                chg[pix] = charge_sum

            # is it a good event?
            if np.max(chg) > 10 * np.mean(chg):
                # print("is not good evt")
                take_event = False
            mean_charge = np.mean(chg) / 58.0

        if take_event:
            self.__event_id.append(np.uint32(event.index.event_id))
            self.__event_type.append(event.trigger.event_type.value)
            self.__ucts_timestamp.append(event.nectarcam.tel[0].evt.ucts_timestamp)
            self.__ucts_busy_counter.append(
                event.nectarcam.tel[0].evt.ucts_busy_counter
            )
            self.__ucts_event_counter.append(
                event.nectarcam.tel[0].evt.ucts_event_counter
            )

            if self.excl_muons:
                self.__mean_event_charge.append(mean_charge)

    # This method need to be defined !
    def finish(self):
        output = UCTSContainer(
            run_number=UCTSContainer.fields["run_number"].type(self._run_number),
            npixels=UCTSContainer.fields["npixels"].type(self._npixels),
            pixels_id=UCTSContainer.fields["pixels_id"].dtype.type(self.pixels_id),
            ucts_timestamp=UCTSContainer.fields["ucts_timestamp"].dtype.type(
                self.__ucts_timestamp
            ),
            mean_event_charge=UCTSContainer.fields["mean_event_charge"].dtype.type(
                self.__mean_event_charge
            ),
            event_type=UCTSContainer.fields["event_type"].dtype.type(self.__event_type),
            event_id=UCTSContainer.fields["event_id"].dtype.type(self.__event_id),
            ucts_busy_counter=UCTSContainer.fields["ucts_busy_counter"].dtype.type(
                self.__ucts_busy_counter
            ),
            ucts_event_counter=UCTSContainer.fields["ucts_event_counter"].dtype.type(
                self.__ucts_event_counter
            ),
        )
        return output


class DeadtimeTestTool(EventsLoopNectarCAMCalibrationTool):
    """The `DeadtimeTestTool` class is an `EventsLoopNectarCAMCalibrationTool` that is
    used to test the deadtime of NectarCAM.

    The `finish` method is responsible for reading the data from the HDF5 file,\
        extracting the relevant information (UCTS timestamps, event counters, and\
            busy counters), and calculating the deadtime-related metrics. The method\
                returns the UCTS timestamps, the time differences between consecutive\
                    UCTS timestamps, the event counters, the busy counters,\
                        the collected\
                        trigger rate, the total time, and the deadtime percentage.
    """

    name = "DeadtimeTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be applied, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        id = kwargs.pop("id")
        output = super().finish(return_output_component=True, *args, **kwargs)

        # Specify event type
        # NOTE: will probably need to be revisited
        if id == 0:  # FFCLS
            event_type = EventType.FLATFIELD
        elif id == 1:  # NSB
            event_type = EventType.SUBARRAY
        elif id == 2:  # Laser
            event_type = EventType.SUBARRAY

        charge_container = output[0].containers[event_type]

        ucts_timestamps = charge_container["ucts_timestamp"]
        event_counter = charge_container["ucts_event_counter"]
        busy_counter = charge_container["ucts_busy_counter"]

        ucts_deltat = [
            ucts_timestamps[i] - ucts_timestamps[i - 1]
            for i in range(1, len(ucts_timestamps))
        ]

        time_tot = ((ucts_timestamps[-1] - ucts_timestamps[0]) * u.ns).to(u.s)
        collected_trigger_rate = (event_counter[-1] + busy_counter[-1]) / time_tot
        deadtime_pc = busy_counter[-1] / (event_counter[-1] + busy_counter[-1]) * 100

        return (
            ucts_timestamps,
            ucts_deltat,
            event_counter,
            busy_counter,
            collected_trigger_rate,
            time_tot,
            deadtime_pc,
        )


class TriggerTimingTestTool(EventsLoopNectarCAMCalibrationTool):
    """The `TriggerTimingTestTool` class is an `EventsLoopNectarCAMCalibrationTool` that
    is used to test the trigger timing of NectarCAM.

    The `finish` method is responsible for reading the data from the HDF5 file,\
        extracting the relevant information (UCTS timestamps), and calculating\
            the RMS value of the difference between consecutive triggers. The method\
                returns the UCTS timestamps, the time differences between consecutive\
                    triggers for events concerning more than 10 pixels (non-muon\
                        related events).
    """

    name = "TriggerTimingTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        output = super().finish(return_output_component=True, *args, **kwargs)

        # Default runs use a laser source and apply a subarray trigger
        # Newer runs use flat-field events
        try:
            charge_container = output[0].containers[EventType.SUBARRAY]
        except Exception:
            charge_container = output[0].containers[EventType.FLATFIELD]

        ucts_timestamps = charge_container["ucts_timestamp"]
        charges_hg = charge_container["charges_hg"]
        good_events = np.where(
            np.max(charges_hg, axis=1) < 10 * np.mean(charges_hg, axis=1),
            True,
            False,
        )

        # Only select "good events"
        ucts_timestamps = ucts_timestamps[good_events]
        charges_hg = charges_hg[good_events]

        # Compute mean charge per event and convert to PE
        charge_per_event = np.mean(charges_hg) / GAIN_DEFAULT

        # dt in nanoseconds
        delta_t = [
            ucts_timestamps[i] - ucts_timestamps[i - 1]
            for i in range(1, len(ucts_timestamps))
        ]

        # make hist to get rms value
        hist_values, bin_edges = np.histogram(delta_t, bins=50)
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        weighted_mean = np.average(bin_centers, weights=hist_values)
        # print("Weighted Mean:", weighted_mean)

        # Compute weighted variance
        weighted_variance = np.average(
            (bin_centers - weighted_mean) ** 2, weights=hist_values
        )
        # print("Weighted Variance:", weighted_variance)

        # Compute RMS value (Standard deviation)
        rms = np.sqrt(weighted_variance)
        # print("RMS:", rms[pix])

        # Compute the total number of data points (sum of histogram values, i.e. N)
        N = np.sum(hist_values)
        # print("Total number of events (N):", N)

        # Error on the standard deviation
        err = rms / np.sqrt(2 * N)
        # print("Error on RMS:", err[pix])

        # charge per run
        charge_per_run = np.mean(charge_per_event)

        return ucts_timestamps, delta_t, rms, err, charge_per_run
