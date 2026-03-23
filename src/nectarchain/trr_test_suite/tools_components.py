import os
import pathlib

import numpy as np
import pandas as pd
from astropy import units as u

# Imports from ctapipe
from ctapipe.containers import EventType
from ctapipe.core.traits import ComponentNameList, Float
from ctapipe_io_nectarcam.constants import N_PIXELS

# Imports from nectarchain
from nectarchain.data.container import ChargesContainers, SPEfitContainer
from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent
from nectarchain.trr_test_suite.utils import get_bad_pixels_list
from nectarchain.utils.constants import GAIN_DEFAULT

get_bad_pixels_list()


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


class ChargeResolutionTestTool(EventsLoopNectarCAMCalibrationTool):
    """This class, `ChargeResolutionTestTool`, is a subclass of
    `EventsLoopNectarCAMCalibrationTool`. It is responsible for performing a linearity
    test on NectarCAM data. The class has a `componentsList` attribute that specifies
    the list of NectarCAM components to be applied.

    The `finish` method is the main functionality of this class. It reads the charge
    data from the output file, calculates the mean charge, standard deviation, and
    standard error for both the high gain and low gain channels, and returns these
    values. This information can be used to assess the linearity of the NectarCAM
    system.
    """

    name = "ChargeResolutionTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        id = kwargs.pop("id")
        super().finish(*args, **kwargs)
        outputs = [c for c in ChargesContainers.from_hdf5(self.output_path)]
        print(f"{len(outputs)=}")
        # output = super().finish(return_output_component=True, *args, **kwargs)
        # print(f'{outputs =}')
        # print(f'{outputs[0].containers=}')
        # print(f'{output[1].containers=}')
        (
            mean_charge_all,
            std_charge_all,
            std_err_all,
            mean_resolution_all,
            ratio_hglg_all,
        ) = ([[], []], [[], []], [[], []], [[], []], [])
        # Default runs use a laser source and apply a subarray trigger
        # Newer runs use flat-field events
        if id == 0:  # FFCLS
            event_type = EventType.FLATFIELD
        elif id == 1:  # NSB
            event_type = EventType.SUBARRAY
        elif id == 2:  # Laser
            event_type = EventType.SUBARRAY
        for output in outputs:
            charge_container = output.containers[event_type]

            # Read gain
            try:
                # print(kwargs["gain_file"])
                gain_data = next(SPEfitContainer.from_hdf5(kwargs["gain_file"]))
                # print(gain_data)
                adc_to_pe = gain_data.high_gain[:, 0]
            except Exception:
                adc_to_pe = GAIN_DEFAULT

            # Read charges
            mean_charge = [0, 0]  # per channel
            std_charge = [0, 0]
            std_err = [0, 0]

            charge_hg = charge_container["charges_hg"]
            charge_lg = charge_container["charges_lg"]
            tom = charge_container["peak_hg"]
            npixels = charge_container["npixels"]
            # print("charge hg ",charge_hg, len(charge_hg), len(charge_hg[0]))
            charge_hg = np.array(charge_hg, dtype=float)
            charge_lg = np.array(charge_lg, dtype=float)

            # ToM cut==============
            tom_mean = np.nanmean(tom, axis=0)
            diff = np.abs(tom - tom_mean)
            # mask events shifted by more than 6 ns
            charge_hg[np.where(diff > 6)] = np.nan
            charge_lg[np.where(diff > 6)] = np.nan

            bad_pix = get_bad_pixels_list()
            # no implementation found even though it's there
            if bad_pix is not None:
                charge_lg[:, bad_pix] = np.nan
                charge_hg[:, bad_pix] = np.nan

            print("adc_to_pe", adc_to_pe)

            # print("bad pix list ", bad_pix)

            charge_lg = np.array(charge_lg)
            charge_hg = np.array(charge_hg)

            mean_charge = [0, 0]  # per channel
            std_charge = [0, 0]
            std_err = [0, 0]

            mean_resolution = [0, 0]

            charge_pe_hg = charge_hg / (adc_to_pe)
            charge_pe_lg = charge_lg / (adc_to_pe)

            n_events = len(charge_pe_hg)
            print("n_events", n_events)

            """
            print(
                charge_pe_lg,
                len(charge_pe_lg),
                len(charge_pe_hg),
                np.nanmean(charge_pe_hg, axis=0),
                np.nanmean(charge_pe_lg, axis=0),
            )
            """
            # print("min ", np.min(np.concatenate(charge_pe_lg)),
            # np.min(np.concatenate(charge_pe_hg)))

            ratio_hglg = np.nanmean(np.nanmean(charge_pe_hg / charge_pe_lg, axis=0))
            print("ratio ", ratio_hglg)
            ratio_hglg_all.append(ratio_hglg)
            for channel, charge in enumerate([charge_pe_hg, charge_pe_lg]):
                # print(channel,charge)
                pix_mean_charge = np.nanmean(charge, axis=0)  # in pe
                # print(pix_mean_charge)

                pix_std_charge = np.nanstd(charge, axis=0)

                pix_resolution = pix_std_charge / pix_mean_charge

                # average of all pixels
                mean_charge[channel] = np.nanmean(pix_mean_charge)
                mean_charge_all[channel].append(mean_charge[channel])
                mean_resolution[channel] = np.nanmean(pix_resolution)
                mean_resolution_all[channel].append(mean_resolution[channel])
                print(f"{mean_resolution_all=}")
                # print("pix ",npixels,channel,pix_resolution,min(pix_resolution),
                # max(pix_resolution),np.where(pix_mean_charge<0),max(pix_std_charge))

                # mean_res_std[channel]    = np.std(pix_resolution[pix_resolution>-500])
                std_charge[channel] = np.nanmean(pix_std_charge)
                std_charge_all[channel].append(std_charge[channel])
                # for the charge resolution
                std_err[channel] = np.nanstd(pix_std_charge)
                std_err_all[channel].append(std_err[channel])

            print(f"{mean_charge=}")

        return (
            mean_charge_all,
            std_charge_all,
            std_err_all,
            npixels,
            mean_resolution_all,
            ratio_hglg_all,
        )


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

    mean_charge_threshold = Float(
        help="Threshold below which to select good events,"
        "in units of mean camera charge.",
        default_value=10,
    ).tag(config=True)

    def finish(self, bootstrap=False, *args, **kwargs):
        id = kwargs.pop("id")
        super().finish(*args, **kwargs)
        outputs = [c for c in ChargesContainers.from_hdf5(self.output_path)]
        # output = super().finish(return_output_component=True, *args, **kwargs)
        # print(f'{outputs =}')
        # print(f'{outputs[0].containers=}')
        # print(f'{output[1].containers=}')
        rms_no_fit_all, rms_no_fit_err_all, mean_charge_pe_all = [], [], []
        # Default runs use a laser source and apply a subarray trigger
        # Newer runs use flat-field events
        if id == 0:  # FFCLS
            event_type = EventType.FLATFIELD
        elif id == 1:  # NSB
            event_type = EventType.SUBARRAY
        elif id == 2:  # Laser
            event_type = EventType.SUBARRAY
        for output in outputs:
            charge_container = output.containers[event_type]
            # try:
            # charge_container = output.containers[EventType.FLATFIELD]
            # except Exception:
            # "charge_container = output.containers[EventType.SUBARRAY]
            # print(f"{charge_container=}")
            charge_all = charge_container["charges_hg"]
            # print(f'{charge_all=}')
            tom_no_fit_all = charge_container["peak_hg"]
            # print(f"{tom_no_fit_all =}")
            npixels = charge_container["npixels"]
            good_evts = np.where(
                np.max(charge_all, axis=1)
                < self.mean_charge_threshold * np.mean(charge_all, axis=1)
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
                    tom_pos = tom[tom < 20]
                    # print(f'{tom_pos=}')
                    boot_rms = []

                    sample = tom_pos[tom_pos < 32]
                    # print(sample)
                    bins = np.linspace(0, 32, 133)
                    hist_values, bin_edges = np.histogram(sample, bins=bins)
                    # print(f'{hist_values=}, {bin_edges=}')
                    # Compute bin centers
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    # print(f'{bin_centers=}')
                    if bootstrap:
                        # print(len(sample))
                        if len(sample) != 0:
                            for _ in range(1000):
                                bootsample = np.random.choice(
                                    sample,
                                    size=int(3 / 4 * (len(sample))),
                                    replace=True,
                                )
                                #             print(len(bootsample), bootsample.mean(),
                                # bootsample.std())
                                boot_rms.append(bootsample.std())
                                # simulated mean of rms
                            bootrms_mean = np.mean(boot_rms)
                            # print(f'{bootrms_mean=}')
                            # simulated standard deviation of rms
                            bootrms_std = np.std(boot_rms)
                            # print(f'{bootrms_std=}')
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
                            # print (f'{weighted_mean=}')
                            # Compute weighted variance
                            weighted_variance = np.average(
                                (bin_centers - weighted_mean) ** 2, weights=hist_values
                            )
                            # print("Weighted Variance:", weighted_variance)

                            # Compute RMS value (Standard deviation)
                            rms[pix] = np.sqrt(weighted_variance)
                            # print(f'{rms=}')
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
            rms_no_fit_all.append(rms_no_fit)
            rms_no_fit_err_all.append(rms_no_fit_err)
            mean_charge_pe_all.append(mean_charge_pe)

        return rms_no_fit_all, rms_no_fit_err_all, mean_charge_pe_all


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
        super().finish(*args, **kwargs)
        outputs = [c for c in ChargesContainers.from_hdf5(self.output_path)]
        # Specify event type
        # NOTE: will probably need to be revisited
        if id == 0:  # FFCLS
            event_type = EventType.FLATFIELD
        elif id == 1:  # NSB
            event_type = EventType.SUBARRAY
        elif id == 2:  # Laser
            event_type = EventType.SUBARRAY
        ucts_timestamps_all = []
        ucts_deltat_all = []
        event_counter_all = []
        busy_counter_all = []
        collected_trigger_rate_all = []
        time_tot_all = []
        deadtime_pc_all = []
        for output in outputs:
            charge_container = output.containers[event_type]

            ucts_timestamps = charge_container["ucts_timestamp"]
            event_counter = charge_container["ucts_event_counter"]
            busy_counter = charge_container["ucts_busy_counter"]

            ucts_deltat = [
                ucts_timestamps[i] - ucts_timestamps[i - 1]
                for i in range(1, len(ucts_timestamps))
            ]

            time_tot = ((ucts_timestamps[-1] - ucts_timestamps[0]) * u.ns).to(u.s)
            collected_trigger_rate = (event_counter[-1] + busy_counter[-1]) / time_tot
            deadtime_pc = (
                busy_counter[-1] / (event_counter[-1] + busy_counter[-1]) * 100
            )
            ucts_timestamps_all.append(ucts_timestamps)
            ucts_deltat_all.append(ucts_deltat)
            event_counter_all.append(event_counter)
            busy_counter_all.append(busy_counter)
            collected_trigger_rate_all.append(collected_trigger_rate.value)
            time_tot_all.append(time_tot)
            deadtime_pc_all.append(deadtime_pc)

        return (
            ucts_timestamps_all,
            ucts_deltat_all,
            event_counter_all,
            busy_counter_all,
            collected_trigger_rate_all,
            time_tot_all,
            deadtime_pc_all,
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

    mean_charge_threshold = Float(
        help="Threshold below which to select good events,"
        "in units of mean camera charge.",
        default_value=10,
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        id = kwargs.pop("id")
        super().finish(*args, **kwargs)
        outputs = [c for c in ChargesContainers.from_hdf5(self.output_path)]
        if id == 0:  # FFCLS
            event_type = EventType.FLATFIELD
        elif id == 1:  # NSB
            event_type = EventType.SUBARRAY
        elif id == 2:  # Laser
            event_type = EventType.SUBARRAY
        timestamps, delta_t_all, rms_all, err_all, charge_per_slice = [], [], [], [], []
        for output in outputs:
            # Default runs use a laser source and apply a subarray trigger
            # Newer runs use flat-field events
            charge_container = output.containers[event_type]
            print(f"{charge_container=}")
            # try:
            # charge_container = output[0].containers[EventType.SUBARRAY]
            # except Exception:
            # charge_container = output[0].containers[EventType.FLATFIELD]

            ucts_timestamps = charge_container["ucts_timestamp"]
            charges_hg = charge_container["charges_hg"]
            print(f"{charges_hg=}")
            good_events = np.where(
                np.max(charges_hg, axis=1)
                < self.mean_charge_threshold * np.mean(charges_hg, axis=1),
                True,
                False,
            )
            print(good_events)

            # Only select "good events"
            ucts_timestamps = ucts_timestamps[good_events]
            charges_hg = charges_hg[good_events]

            # Compute mean charge per event and convert to PE
            charge_per_event = np.mean(charges_hg) / GAIN_DEFAULT

            # dt in nanoseconds
            delta_t = np.diff(ucts_timestamps)

            # make hist to get rms value
            hist_values, bin_edges = np.histogram(delta_t, bins=50)
            print(hist_values)
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

            timestamps.append(ucts_timestamps)
            delta_t_all.append(delta_t)
            rms_all.append(rms)
            err_all.append(err)
            charge_per_slice.append(charge_per_run)

        return timestamps, delta_t_all, rms_all, err_all, charge_per_slice


class TempLongRunTestTool(EventsLoopNectarCAMCalibrationTool):
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

    name = "TempLongRunTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargesComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    mean_charge_threshold = Float(
        help="Threshold below which to select good events,"
        "in units of mean camera charge.",
        default_value=10,
    ).tag(config=True)

    def finish(self, bootstrap=False, *args, **kwargs):
        id = kwargs.pop("id")
        super().finish(*args, **kwargs)
        outputs = [c for c in ChargesContainers.from_hdf5(self.output_path)]
        # print (f'{len(outputs)=}')
        # output = super().finish(return_output_component=True, *args, **kwargs)
        # print(f'{outputs =}')
        # print(f'{outputs[0].containers=}')
        # print(f'{output[1].containers=}')
        (
            mean_charge_all,
            std_charge_all,
            std_err_all,
            mean_resolution_all,
            err_resolution_all,
            ratio_hglg_all,
            tom_all,
            tom_all_err,
            rms_no_fit_all,
            rms_no_fit_err_all,
            ucts_deltat_all,
            deadtime_err,
            event_rate_all,
            # busy_counter_all,
            collected_trigger_rate_all,
            time_tot_all,
            deadtime_pc_all,
            trig_rms_all,
            trig_err_all,
            tmin,
            tmax,
        ) = (
            [[], []],
            [[], []],
            [[], []],
            [[], []],
            [[], []],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        # Default runs use a laser source and apply a subarray trigger
        # Newer runs use flat-field events
        if id == 0:  # FFCLS
            event_type = EventType.FLATFIELD
        elif id == 1:  # NSB
            event_type = EventType.SUBARRAY
        elif id == 2:  # Laser
            event_type = EventType.SUBARRAY
        for output in outputs:
            charge_container = output.containers[event_type]

            # Read gain
            try:
                print(kwargs["gain_file"])
                gain_data = next(
                    SPEfitContainer.from_hdf5(
                        cls=SPEfitContainer, path=kwargs["gain_file"]
                    )
                )
                # gain_data = next(SPEfitContainer.from_hdf5(kwargs["gain_file"]))
                print(gain_data)
                adc_to_pe = gain_data.high_gain[:, 0]
                print("IT WORKS")
            except Exception:
                print("EXCEPTION")
                adc_to_pe = GAIN_DEFAULT

            # Read charges
            mean_charge = [0, 0]  # per channel
            std_charge = [0, 0]
            std_err = [0, 0]

            mean_resolution = [0, 0]
            err_resolution = [0, 0]
            charge_hg = charge_container["charges_hg"]
            charge_lg = charge_container["charges_lg"]

            tom = charge_container["peak_hg"]
            npixels = charge_container["npixels"]
            ucts_timestamps = charge_container["ucts_timestamp"]
            event_counter = charge_container["ucts_event_counter"]
            busy_counter = charge_container["ucts_busy_counter"]
            events_id = charge_container["event_id"]
            print(f"{events_id}")
            tmin.append(np.min(ucts_timestamps))
            tmax.append(np.max(ucts_timestamps))

            print(f"{charge_hg=},\
                {charge_lg=},\
                {tom=},\
                {npixels=},\
                {ucts_timestamps=},\
                {event_counter=},\
                {busy_counter=}")

            # print("charge hg ",charge_hg, len(charge_hg), len(charge_hg[0]))
            charge_hg = np.array(charge_hg, dtype=float)
            charge_lg = np.array(charge_lg, dtype=float)
            good_evts = np.where(
                np.max(charge_hg, axis=1)
                < self.mean_charge_threshold * np.mean(charge_hg, axis=1),
                True,
                False,
            )

            charge_hg = charge_hg[good_evts]

            # ToM cut==============

            bad_pix = get_bad_pixels_list()
            lenpix = N_PIXELS - len(bad_pix)
            if bad_pix is not None:
                charge_lg[:, bad_pix] = np.nan
                charge_hg[:, bad_pix] = np.nan
                tom[:, bad_pix] = np.nan

            tom_mean = np.nanmean(tom, axis=0)
            tom_err = np.nanstd(tom, axis=0) / np.sqrt(len(tom))
            tom_slice = np.nanmean(tom_mean)
            tom_slice_err = (1 / lenpix) * np.sqrt(np.nansum(tom_err**2))
            tom_all.append(tom_slice)
            print(f"{tom_all=}")
            tom_all_err.append(tom_slice_err)
            print(f"{tom_all_err=}")
            # diff = np.abs(tom - tom_mean)
            # mask events shifted by more than 6 ns
            # charge_hg[np.where(diff > 4)] = np.nan
            # charge_lg[np.where(diff > 4)] = np.nan

            print("adc_to_pe", adc_to_pe)

            print("bad pix list ", bad_pix)

            charge_lg = np.array(charge_lg)
            charge_hg = np.array(charge_hg)

            charge_pe_hg = charge_hg / (adc_to_pe)
            charge_pe_lg = charge_lg / (adc_to_pe)
            n_events = len(charge_pe_hg)

            print("n_events", n_events)

            ratio_hglg = np.nanmean(np.nanmean(charge_pe_hg / charge_pe_lg, axis=0))
            print("ratio ", ratio_hglg)

            ratio_hglg_all.append(ratio_hglg)
            for channel, charge in enumerate([charge_pe_hg, charge_pe_lg]):
                # print(channel,charge)
                pix_mean_charge = np.nanmean(charge, axis=0)  # in pe
                # print(pix_mean_charge)

                pix_std_charge = np.nanstd(charge, axis=0)

                pix_resolution = pix_std_charge / pix_mean_charge

                # average of all pixels
                mean_charge[channel] = np.nanmean(pix_mean_charge)
                mean_charge_all[channel].append(mean_charge[channel])
                mean_resolution[channel] = np.nanmean(pix_resolution)
                err_resolution[channel] = np.nanstd(pix_resolution) / np.sqrt(
                    2 * lenpix
                )
                err_resolution_all[channel].append(err_resolution[channel])
                mean_resolution_all[channel].append(mean_resolution[channel])
                print(mean_resolution_all)
                print(f"{mean_resolution_all=}")

                std_charge[channel] = np.nanmean(pix_std_charge)
                std_charge_all[channel].append(std_charge[channel])

                # for the charge resolution
                std_err[channel] = np.nanstd(pix_std_charge)
                std_err_all[channel].append(std_err[channel])

            print(f"{mean_charge=}")

            tom_no_fit = np.array(tom[good_evts]).reshape(len(good_evts), npixels)
            rms_no_fit = np.zeros(npixels)

            rms_no_fit_err = np.zeros(npixels)

            # bootstrapping method

            for pix in range(npixels):
                for tom, rms, err in zip(
                    [tom_no_fit[:, pix]], [rms_no_fit], [rms_no_fit_err]
                ):
                    tom_pos = tom[tom < 20]

                    boot_rms = []

                    sample = tom_pos[tom_pos < 32]

                    bins = np.linspace(0, 32, 133)
                    hist_values, bin_edges = np.histogram(sample, bins=bins)

                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                    if bootstrap:

                        if len(sample) != 0:
                            for _ in range(1000):
                                bootsample = np.random.choice(
                                    sample,
                                    size=int(3 / 4 * (len(sample))),
                                    replace=True,
                                )

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

                            # Compute weighted variance
                            weighted_variance = np.average(
                                (bin_centers - weighted_mean) ** 2, weights=hist_values
                            )

                            # Compute RMS value (Standard deviation)
                            rms[pix] = np.sqrt(weighted_variance)

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
            if bad_pix is not None:
                rms_no_fit[bad_pix] = np.nan
                rms_no_fit_err[bad_pix] = np.nan

            rms_cam_nofit = np.nanmean(rms_no_fit)
            rms_cam_nofit_err = (1 / lenpix) * np.sqrt(np.nansum(rms_no_fit_err**2))
            rms_no_fit_all.append(rms_cam_nofit)
            rms_no_fit_err_all.append(rms_cam_nofit_err)

            ucts_deltat = [
                ucts_timestamps[i] - ucts_timestamps[i - 1]
                for i in range(1, len(ucts_timestamps))
            ]
            ucts_deltat = np.diff(ucts_timestamps)
            time_tot = ((ucts_timestamps[-1] - ucts_timestamps[0]) * u.ns).to(u.s)
            collected_trigger_rate = (event_counter[-1] + busy_counter[-1]) / time_tot

            deadtime_pc = (
                busy_counter[-1] / (event_counter[-1] + busy_counter[-1]) * 100
            )
            # the error is not yet calculated
            counter = len(event_counter) - 1
            print(f"{counter=}")
            num_events = event_counter[-1]
            print(f"{num_events=}")
            event_rate = counter / num_events
            print(f"{event_rate=}")
            mean_deltat = np.nanmean(ucts_deltat)
            deadtime_std = np.nanstd(ucts_deltat) / np.sqrt(counter)

            # CONTINUER ICI
            # ucts_timestamps_all.append(ucts_timestamps)
            ucts_deltat_all.append(mean_deltat)
            deadtime_err.append(deadtime_std)
            event_rate_all.append(event_rate)
            # busy_counter_all.append(busy_counter)
            collected_trigger_rate_all.append(collected_trigger_rate.value)
            time_tot_all.append(time_tot)
            deadtime_pc_all.append(deadtime_pc)
            print(f"{charge_container=}")

            # Only select "good events"
            ucts_timestamps = ucts_timestamps[good_evts]

            # dt in nanoseconds

            # make hist to get rms value
            hist_values, bin_edges = np.histogram(ucts_deltat, bins=50)
            print(hist_values)
            # Compute bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            weighted_mean = np.average(bin_centers, weights=hist_values)

            # Compute weighted variance
            weighted_variance = np.average(
                (bin_centers - weighted_mean) ** 2, weights=hist_values
            )

            # Compute RMS value (Standard deviation)
            trig_rms = np.sqrt(weighted_variance)

            # Compute the total number of data points (sum of histogram values, i.e. N)
            N = np.sum(hist_values)

            # Error on the standard deviation
            trig_err = trig_rms / np.sqrt(2 * N)
            # print("Error on RMS:", err[pix])

            trig_rms_all.append(trig_rms)
            trig_err_all.append(trig_err)
        print(tmin, tmax)
        return (
            mean_charge_all,
            std_charge_all,
            std_err_all,
            mean_resolution_all,
            err_resolution_all,
            ratio_hglg_all,
            tom_all,
            tom_all_err,
            rms_no_fit_all,
            rms_no_fit_err_all,
            trig_rms_all,
            trig_err_all,
            ucts_deltat_all,
            deadtime_err,
            event_rate_all,
            # busy_counter_all,
            collected_trigger_rate_all,
            time_tot_all,
            deadtime_pc_all,
            tmin,
            tmax,
        )
