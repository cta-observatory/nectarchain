import os
import pathlib
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
from astropy import units as u
from ctapipe.containers import EventType, Field
from ctapipe.core.traits import ComponentNameList, Integer
from ctapipe_io_nectarcam import constants
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import find_peaks
from utils import adc_to_pe, argmedian

from nectarchain.data.container import NectarCAMContainer
from nectarchain.makers import EventsLoopNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent


# overriding so we can have maxevents in the path
def _init_output_path(self):
    """
    Initializes the output path for the NectarCAMCalibrationTool.

    If `max_events` is `None`, the output file name will be in the format `{self.name}_run{self.run_number}.h5`. Otherwise, the file name will be in the format `{self.name}_run{self.run_number}_maxevents{self.max_events}.h5`.

    The output path is constructed by joining the `NECTARCAMDATA` environment variable (or `/tmp` if not set) with the `tests` subdirectory and the generated file name.
    """

    if self.max_events is None:
        filename = f"{self.name}_run{self.run_number}.h5"
    else:
        filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}.h5"
    self.output_path = pathlib.Path(
        f"{os.environ.get('NECTARCAMDATA','/tmp')}/tests/{filename}"
    )


EventsLoopNectarCAMCalibrationTool._init_output_path = _init_output_path


class ChargeContainer(NectarCAMContainer):
    """
    This class contains fields that store various properties and data related to NectarCAM events, including:

    - `run_number`: The run number associated with the waveforms.
    - `npixels`: The number of effective pixels.
    - `pixels_id`: An array of pixel IDs.
    - `ucts_timestamp`: An array of UCTS timestamps for the events.
    - `event_type`: An array of trigger event types.
    - `event_id`: An array of event IDs.
    - `charge_hg`: A 2D array of high gain charge values.
    - `charge_lg`: A 2D array of low gain charge values.
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
    event_type = Field(
        type=np.ndarray, dtype=np.uint8, ndim=1, description="trigger event type"
    )
    event_id = Field(type=np.ndarray, dtype=np.uint32, ndim=1, description="event ids")

    charge_hg = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="The high gain charge"
    )
    charge_lg = Field(
        type=np.ndarray, dtype=np.float64, ndim=2, description="The low gain charge"
    )


class ChargeComp(NectarCAMComponent):
    """
    This class `ChargeComp` is a NectarCAMComponent that processes NectarCAM event data. It extracts the charge information from the waveforms of each event, handling cases of saturated or noisy events. The class has the following configurable parameters:

    - `window_shift`: The time in ns before the peak to extract the charge.
    - `window_width`: The duration of the charge extraction window in ns.

    The `__init__` method initializes important members of the component, such as timestamps, event type, event ids, pedestal and charge for both gain channels.
    The `__call__` method is the main processing logic, which is called for each event. It extracts the charge information for both high gain and low gain channels, handling various cases such as saturated events and events with no signal.
    The `finish` method collects all the processed data and returns a `ChargeContainer` object containing the run number, number of pixels, pixel IDs, UCTS timestamps, event types, event IDs, and the high and low gain charge values.
    """

    window_shift = Integer(
        default_value=6,
        help="the time in ns before the peak to extract charge",
    ).tag(config=True)

    window_width = Integer(
        default_value=12,
        help="the duration of the extraction window in ns",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        ## If you want you can add here members of MyComp, they will contain interesting quantity during the event loop process

        self.__ucts_timestamp = []
        self.__event_type = []
        self.__event_id = []

        self.__pedestal_hg = []
        self.__pedestal_lg = []

        self.__charge_hg = []
        self.__charge_lg = []

    ##This method need to be defined !
    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        self.__event_id.append(np.uint32(event.index.event_id))

        # print(event.index.event_id)

        self.__event_type.append(event.trigger.event_type.value)
        self.__ucts_timestamp.append(event.nectarcam.tel[0].evt.ucts_timestamp)

        wfs = []
        wfs.append(event.r0.tel[0].waveform[constants.HIGH_GAIN][self.pixels_id])
        wfs.append(event.r0.tel[0].waveform[constants.LOW_GAIN][self.pixels_id])

        #####THE JOB IS HERE######
        for i, (pedestal, charge) in enumerate(
            zip(
                [self.__pedestal_hg, self.__pedestal_lg],
                [self.__charge_hg, self.__charge_lg],
            )
        ):
            wf = np.array(wfs[i], dtype=np.float16)

            index_peak = np.argmax(wf, axis=1)
            index_peak[index_peak < 20] = 20
            index_peak[index_peak > 40] = 40

            signal_start = index_peak - self.window_shift
            signal_stop = index_peak + self.window_width - self.window_shift

            if event.trigger.event_type == EventType.FLATFIELD:
                integral = np.zeros(len(self.pixels_id))

                for pix in range(len(self.pixels_id)):
                    # search for saturated events or events with no signal
                    ped = np.round(np.mean(wf[pix, 0 : signal_start[pix]]))

                    peaks_sat = find_peaks(
                        wf[pix, 20:45], height=1000, plateau_size=self.window_width
                    )
                    if len(peaks_sat[0]) == 1:
                        # saturated

                        # print("saturated")

                        signal_start[pix] = (
                            argmedian(wf[pix, 20:45]) + 20 - self.window_shift
                        )
                        signal_stop[pix] = signal_start[pix] + self.window_width
                        integral[pix] = np.sum(
                            wf[pix, signal_start[pix] : signal_stop[pix]]
                        ) - ped * (signal_stop[pix] - signal_start[pix])

                    else:
                        peaks_signal = find_peaks(wf[pix], prominence=10)
                        if len(peaks_signal[0]) >= 12:
                            # print("noisy event")
                            integral[pix] = 0

                        elif len(peaks_signal[0]) < 1:
                            # flat
                            integral[pix] = 0

                        else:
                            # x = np.linspace(0,signal_stop[pix]-signal_start[pix],signal_stop[pix]-signal_start[pix])
                            # spl = UnivariateSpline(x,y)
                            # integral[pix] = spl.integral(0,signal_stop[pix]-signal_start[pix])

                            integral[pix] = np.sum(
                                wf[pix, signal_start[pix] : signal_stop[pix]]
                            ) - ped * (signal_stop[pix] - signal_start[pix])

                chg = integral

                charge.append(chg)

    ##This method need to be defined !
    def finish(self):
        output = ChargeContainer(
            run_number=ChargeContainer.fields["run_number"].type(self._run_number),
            npixels=ChargeContainer.fields["npixels"].type(self._npixels),
            pixels_id=ChargeContainer.fields["pixels_id"].dtype.type(self._pixels_id),
            ucts_timestamp=ChargeContainer.fields["ucts_timestamp"].dtype.type(
                self.__ucts_timestamp
            ),
            event_type=ChargeContainer.fields["event_type"].dtype.type(
                self.__event_type
            ),
            event_id=ChargeContainer.fields["event_id"].dtype.type(self.__event_id),
            charge_hg=ChargeContainer.fields["charge_hg"].dtype.type(
                np.array(self.__charge_hg)
            ),
            charge_lg=ChargeContainer.fields["charge_lg"].dtype.type(
                np.array(self.__charge_lg)
            ),
        )

        return output


class LinearityTestTool(EventsLoopNectarCAMCalibrationTool):
    """
    This class, `LinearityTestTool`, is a subclass of `EventsLoopNectarCAMCalibrationTool`. It is responsible for performing a linearity test on NectarCAM data. The class has a `componentsList` attribute that specifies the list of NectarCAM components to be applied.

    The `finish` method is the main functionality of this class. It reads the charge data from the output file, calculates the mean charge, standard deviation, and standard error for both the high gain and low gain channels, and returns these values. This information can be used to assess the linearity of the NectarCAM system.
    """

    name = "LinearityTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ChargeComp"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        super().finish(return_output_component=True, *args, **kwargs)

        mean_charge = [0, 0]  # per channel
        std_charge = [0, 0]
        std_err = [0, 0]

        charge_hg = []
        charge_lg = []

        output_file = h5py.File(self.output_path)

        for thing in output_file:
            group = output_file[thing]
            dataset = group["ChargeContainer"]
            data = dataset[:]
            # print("data",data)
            for tup in data:
                try:
                    npixels = tup[1]
                    charge_hg.extend(tup[6])
                    charge_lg.extend(tup[7])

                except:
                    break

        output_file.close()

        charge_pe_hg = np.array(charge_hg) / adc_to_pe
        charge_pe_lg = np.array(charge_lg) / adc_to_pe

        for channel, charge in enumerate([charge_pe_hg, charge_pe_lg]):
            pix_mean_charge = np.mean(charge, axis=0)  # in pe

            pix_std_charge = np.std(charge, axis=0)

            # average of all pixels
            mean_charge[channel] = np.mean(pix_mean_charge)

            std_charge[channel] = np.mean(pix_std_charge)
            # for the charge resolution
            std_err[channel] = np.std(pix_std_charge)

        return mean_charge, std_charge, std_err, npixels


class ToMContainer(NectarCAMContainer):
    """
    Attributes:
        run_number (np.uint16): The run number associated with the waveforms.
        npixels (np.uint16): The number of effective pixels.
        pixels_id (np.ndarray[np.uint16]): The pixel IDs.
        ucts_timestamp (np.ndarray[np.uint64]): The UCTS timestamps of the events.
        event_type (np.ndarray[np.uint8]): The trigger event types.
        event_id (np.ndarray[np.uint32]): The event IDs.
        charge_hg (np.ndarray[np.float64]): The mean high gain charge per event.
        tom_no_fit (np.ndarray[np.float64]): The time of maximum from the data (no fitting).
        good_evts (np.ndarray[np.uint32]): The IDs of the good (non-cosmic ray) events.
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
    event_type = Field(
        type=np.ndarray, dtype=np.uint8, ndim=1, description="trigger event type"
    )
    event_id = Field(type=np.ndarray, dtype=np.uint32, ndim=1, description="event ids")

    charge_hg = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="The mean high gain charge per event",
    )

    # tom_mu = Field(
    #     type=np.ndarray, dtype=np.float64, ndim=2, description="Time of maximum of signal fitted with gaussian"
    # )

    # tom_sigma = Field(
    #     type=np.ndarray, dtype=np.float64, ndim=2, description="Time of fitted maximum sigma"
    # )
    tom_no_fit = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="Time of maximum from data (no fitting)",
    )
    good_evts = Field(
        type=np.ndarray,
        dtype=np.uint32,
        ndim=1,
        description="good (non cosmic ray) event ids",
    )


class ToMComp(NectarCAMComponent):
    """
    This class, `ToMComp`, is a component of the NectarCAM system that is responsible for processing waveform data. It has several configurable parameters, including the width and shift before the peak of the time window for charge extraction, the peak height threshold.

    The `__init__` method initializes some important component members, such as timestamps, event type, event ids, pedestal and charge values for both gain channels.

    The `__call__` method is the main entry point for processing an event. It extracts the waveform data, calculates the pedestal, charge, and time of maximum (ToM) for each pixel, and filters out events that do not meet the peak height threshold. The results are stored in various member variables, which are then returned in the `finish` method.

    The `finish` method collects the processed data from the member variables and returns a `ToMContainer` object, which contains the run number, number of pixels, pixel IDs, UCTS timestamps, event types, event IDs, high-gain charge, ToM without fitting, and IDs of good (non-cosmic ray) events.
    """

    window_shift = Integer(
        default_value=6,
        help="the time in ns before the peak to extract charge",
    ).tag(config=True)

    window_width = Integer(
        default_value=16,
        help="the duration of the extraction window in ns",
    ).tag(config=True)

    peak_height = Integer(
        default_value=10,
        help="height of peak to consider event not to be just pedestal (ADC counts)",
    ).tag(config=True)

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        ## If you want you can add here members of MyComp, they will contain interesting quantity during the event loop process

        self.__ucts_timestamp = []
        self.__event_type = []
        self.__event_id = []

        self.__charge_hg = []
        self.__pedestal_hg = []

        # self.__tom_mu = []

        # self.__tom_sigma = []

        self.__tom_no_fit = []

        self.__good_evts = []

        self.__ff_event_ind = -1

    ##This method need to be defined !
    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        self.__event_id.append(np.uint32(event.index.event_id))

        self.__event_type.append(event.trigger.event_type.value)
        self.__ucts_timestamp.append(event.nectarcam.tel[0].evt.ucts_timestamp)

        wfs = []
        wfs.append(event.r0.tel[0].waveform[constants.HIGH_GAIN][self.pixels_id])

        self.__ff_event_ind += 1

        #####THE JOB IS HERE######

        for i, (pedestal, charge, tom_no_fit) in enumerate(
            zip([self.__pedestal_hg], [self.__charge_hg], [self.__tom_no_fit])
        ):
            wf = np.array(wfs[i])  # waveform per gain
            index_peak = np.argmax(wf, axis=1)  # tom per event/pixel
            # use it to first find pedestal and then will filter out no signal events
            index_peak[index_peak < 20] = 20
            index_peak[index_peak > 40] = 40
            signal_start = index_peak - self.window_shift
            signal_stop = index_peak + self.window_width - self.window_shift

            ped = np.array(
                [
                    np.mean(wf[pix, 0 : signal_start[pix]])
                    for pix in range(len(self.pixels_id))
                ]
            )

            pedestal.append(ped)

            tom_no_fit_evt = np.zeros(len(self.pixels_id))
            chg = np.zeros(len(self.pixels_id))
            # will calculate tom & charge like federica
            for pix in range(len(self.pixels_id)):
                y = (
                    wf[pix] - ped[pix]
                )  # np.maximum(wf[pix] - ped[pix],np.zeros(len(wf[pix])))

                x = np.linspace(0, len(y), len(y))
                xi = np.linspace(0, len(y), 251)
                ius = InterpolatedUnivariateSpline(x, y)
                yi = ius(xi)
                peaks, _ = find_peaks(
                    yi,
                    height=self.peak_height,
                )
                # print(peaks)
                peaks = peaks[xi[peaks] > 20]
                peaks = peaks[xi[peaks] < 32]
                # print(peaks)

                if len(peaks) > 0:
                    # find the max peak
                    # max_peak = max(yi[peaks])
                    max_peak_index = np.argmax(yi[peaks], axis=0)

                    # Check if there is not a peak but a plateaux
                    yi_rounded = np.around(yi[peaks] / max(yi[peaks]), 1)
                    maxima_peak_index = np.argwhere(yi_rounded == np.amax(yi_rounded))

                    # saturated events
                    if (
                        (len(maxima_peak_index) > 1)
                        and (min(xi[peaks[maxima_peak_index]]) > signal_start[pix])
                        and (max(xi[peaks[maxima_peak_index]]) < signal_stop[pix])
                    ):
                        # saturated event

                        max_peak_index = int(np.median(maxima_peak_index))

                        # simple sum integration
                        x_max_pos = np.argmin(
                            np.abs(x - xi[peaks[max_peak_index]])
                        )  # find the maximum in the not splined array
                        charge_sum = y[
                            (x_max_pos - (self.window_width - 10)) : (
                                x_max_pos + (self.window_width - 6)
                            )
                        ].sum()

                        # gaussian fit
                        # change_grad_pos_left = 3
                        # change_grad_pos_right = 3

                        # mask = np.ma.masked_where(y > (4095 - 400), x)
                        # x_fit = np.ma.compressed(mask)
                        # mask = np.ma.masked_where(y > (4095 - 400), y)
                        # y_fit = np.ma.compressed(mask)
                        # mean = xi[peaks[max_peak_index]]
                        # sigma = change_grad_pos_right + change_grad_pos_left

                        # # fit
                        # model = Model(gaus)
                        # params = model.make_params(a=yi[peaks[max_peak_index]] * 3, mu=mean, sigma=sigma)
                        # result = model.fit(y_fit, params, x=x_fit)

                        # result_sigma = result.params['sigma'].value
                        # result_mu = result.params['mu'].value

                        max_position_x_prefit = xi[peaks[max_peak_index]]

                    elif (
                        (len(maxima_peak_index) == 1)
                        and (xi[peaks[max_peak_index]] > signal_start[pix])
                        and (xi[peaks[max_peak_index]] < signal_stop[pix])
                    ):
                        # simple sum integration
                        x_max_pos = np.argmin(
                            np.abs(x - xi[peaks[max_peak_index]])
                        )  # find the maximum in the not splined array
                        charge_sum = y[
                            (x_max_pos - (self.window_width - 10)) : (
                                x_max_pos + (self.window_width - 6)
                            )
                        ].sum()

                        # gaussian fit
                        # change_grad_pos_left = 3
                        # change_grad_pos_right = 3
                        # mean = xi[peaks[max_peak_index]]
                        # sigma = change_grad_pos_right + change_grad_pos_left # define window for the gaussian fit

                        # x_fit =  xi[peaks[max_peak_index]-change_grad_pos_left:peaks[max_peak_index]+change_grad_pos_right]
                        # y_fit =  yi[peaks[max_peak_index]-change_grad_pos_left:peaks[max_peak_index]+change_grad_pos_right]
                        # model = Model(gaus)
                        # params = model.make_params(a=yi[peaks[max_peak_index]],mu=mean,sigma=sigma)
                        # result = model.fit(y_fit, params, x=x_fit)

                        max_position_x_prefit = xi[peaks[max_peak_index]]
                        # result_sigma  = result.params['sigma'].value
                        # result_mu  = result.params['mu'].value

                    else:
                        # index_x_window_min = list(xi).index(closest_value(xi, signal_start[pix]))
                        charge_sum = y[
                            signal_start[pix] : signal_start[pix] + self.window_width
                        ].sum()

                        max_position_x_prefit = -1
                        # result_sigma = -1
                        # result_mu = -1

                else:
                    # If no maximum is found, the integration is done between 20 and 36 ns.
                    signal_start[pix] = 20

                    # index_x_window_min = list(xi).index(closest_value(xi, signal_start[pix]))
                    charge_sum = y[
                        signal_start[pix] : signal_start[pix] + self.window_width
                    ].sum()

                    max_position_x_prefit = -1

                    # result_sigma = -1
                    # result_mu = -1

                # tom_mu_evt[pix] = result_mu
                # tom_sigma_evt[pix] = result_sigma
                tom_no_fit_evt[pix] = max_position_x_prefit
                chg[pix] = charge_sum

            # tom_mu.append(tom_mu_evt)
            # tom_sigma.append(tom_sigma_evt)
            tom_no_fit.append(tom_no_fit_evt)
            # print("tom for event", tom_no_fit_evt)
            charge.append(chg)

            # is it a good event?
            if np.max(chg) < 10 * np.mean(chg):
                # print("is good evt")
                self.__good_evts.append(self.__ff_event_ind)

    ##This method need to be defined !
    def finish(self):
        output = ToMContainer(
            run_number=ToMContainer.fields["run_number"].type(self._run_number),
            npixels=ToMContainer.fields["npixels"].type(self._npixels),
            pixels_id=ToMContainer.fields["pixels_id"].dtype.type(self.pixels_id),
            ucts_timestamp=ToMContainer.fields["ucts_timestamp"].dtype.type(
                self.__ucts_timestamp
            ),
            event_type=ToMContainer.fields["event_type"].dtype.type(self.__event_type),
            event_id=ToMContainer.fields["event_id"].dtype.type(self.__event_id),
            charge_hg=ToMContainer.fields["charge_hg"].dtype.type(self.__charge_hg),
            # tom_mu=ToMContainer.fields["tom_mu"].dtype.type(
            #     self.__tom_mu
            # ),
            # tom_sigma=ToMContainer.fields["tom_sigma"].dtype.type(
            #     self.__tom_sigma
            # ),
            tom_no_fit=ToMContainer.fields["tom_no_fit"].dtype.type(self.__tom_no_fit),
            good_evts=ToMContainer.fields["good_evts"].dtype.type(self.__good_evts),
        )
        return output


class TimingResolutionTestTool(EventsLoopNectarCAMCalibrationTool):
    """
    This class, `TimingResolutionTestTool`, is a subclass of `EventsLoopNectarCAMCalibrationTool` and is used to perform timing resolution tests on NectarCAM data. It reads the output data from the `ToMContainer` dataset and processes the charge, timing, and event information to calculate the timing resolution and mean charge in photoelectrons.

    The `finish()` method is the main entry point for this tool. It reads the output data from the HDF5 file, filters the data to remove cosmic ray events, and then calculates the timing resolution and mean charge per photoelectron. The timing resolution is calculated using a weighted mean and variance approach, with an option to use a bootstrapping method to estimate the error on the RMS value.

    The method returns the RMS of the timing resolution, the error on the RMS, and the mean charge in photoelectrons.
    """

    name = "TimingResolutionTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ToMComp"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, bootstrap=False, *args, **kwargs):
        super().finish(return_output_component=False, *args, **kwargs)

        # tom_mu_all= output[0].tom_mu
        # tom_sigma_all= output[0].tom_sigma
        output_file = h5py.File(self.output_path)

        charge_all = []
        tom_no_fit_all = []
        good_evts = []

        for thing in output_file:
            group = output_file[thing]
            dataset = group["ToMContainer"]
            data = dataset[:]
            # print("data",data)
            for tup in data:
                try:
                    npixels = tup[1]
                    charge_all.extend(tup[6])
                    tom_no_fit_all.extend(tup[7])
                    good_evts.extend(tup[8])
                except:
                    break

        output_file.close()

        tom_no_fit_all = np.array(tom_no_fit_all)
        charge_all = np.array(charge_all)
        # print(tom_no_fit_all)
        # print(charge_all)

        # clean cr events
        good_evts = np.array(good_evts)
        # print(good_evts)
        charge = charge_all[good_evts]
        mean_charge_pe = np.mean(np.mean(charge, axis=0)) / 58.0
        # tom_mu = np.array(tom_mu_all[good_evts]).reshape(len(good_evts),output[0].npixels)

        # tom_sigma = np.array(tom_sigma_all[good_evts]).reshape(len(good_evts),output[0].npixels)
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
                            #             print(len(bootsample), bootsample.mean(), bootsample.std())
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

                        # Compute the total number of data points (sum of histogram values, i.e. N)
                        N = np.sum(hist_values)
                        # print("Total number of events (N):", N)

                        # Error on the standard deviation
                        err[pix] = rms[pix] / np.sqrt(2 * N)
                        # print("Error on RMS:", err[pix])
                    except:
                        # no data
                        rms[pix] = np.nan
                        err[pix] = np.nan

        return rms_no_fit, rms_no_fit_err, mean_charge_pe


class ToMPairsTool(EventsLoopNectarCAMCalibrationTool):
    """
    This class, `ToMPairsTool`, is an `EventsLoopNectarCAMCalibrationTool` that is used to process ToM (Time of maximum) data from NectarCAM.

    The `finish` method has the following functionalities:

    - It reads in ToM data from an HDF5 file and applies a transit time correction to the ToM values using a provided lookup table.
    - It calculates the time difference between ToM pairs for both corrected and uncorrected ToM values.
    - It returns the uncorrected ToM values, the corrected ToM values, the pixel IDs, and the time difference calculations for the uncorrected and corrected ToM values.

    The class has several configurable parameters, including the list of NectarCAM components to apply, the maximum number of events to process, and the output file path.

    """

    name = "ToMPairsTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["ToMComp"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        super().finish(return_output_component=True, *args[1:], **kwargs)

        tt_path = args[0]
        pmt_tt = pd.read_csv(tt_path)["tom_pmttt_delta_correction"].values

        tom_no_fit_all = []

        pixels_id = []

        output_file = h5py.File(self.output_path)

        for thing in output_file:
            group = output_file[thing]
            dataset = group["ToMContainer"]
            data = dataset[:]
            # print("data",data)
            for tup in data:
                try:
                    pixels_id.extend(tup[2])
                    tom_no_fit_all.extend(tup[7])
                except:
                    break

        output_file.close()
        pixels_id = np.array(pixels_id)
        tom_no_fit_all = np.array(tom_no_fit_all)

        # clean cr events
        # good_evts = output[0].good_evts
        # charge=charge_all[good_evts]
        # mean_charge_pe = np.mean(np.mean(charge,axis=0))/58.
        tom_no_fit = np.array(tom_no_fit_all, dtype=np.float64)  # tom(event,pixel)
        # tom_no_fit = tom_no_fit[np.all(tom_no_fit>0,axis=0)]
        tom_corrected = -np.ones(
            tom_no_fit.shape, dtype=np.float128
        )  # -1 for the ones that have tom beyond 0-60

        iter = enumerate(pixels_id)

        for i, pix in iter:
            # print(pix, pmt_tt[pix])
            normal_values = [
                a and b for a, b in zip(tom_no_fit[:, i] > 0, tom_no_fit[:, i] < 60)
            ]

            tom_corrected[normal_values, i] = (
                tom_no_fit[:, i][normal_values] - pmt_tt[pix]
            )

            # print(tom_corrected)

        pixel_ind = [
            i for i in range(len(pixels_id))
        ]  # dealing with indices of pixels in array
        pixel_pairs = list(combinations(pixel_ind, 2))
        dt_no_correction = np.zeros((len(pixel_pairs), tom_no_fit_all.shape[0]))
        dt_corrected = np.zeros((len(pixel_pairs), tom_no_fit_all.shape[0]))

        for i, pair in enumerate(pixel_pairs):
            pix1_ind = pixel_ind[pair[0]]
            pix2_ind = pixel_ind[pair[1]]

            for event in range(tom_no_fit_all.shape[0]):
                cond_no_correction = (
                    tom_no_fit[event, pix1_ind] > 0
                    and tom_no_fit[event, pix1_ind] < 60
                    and tom_no_fit[event, pix2_ind] > 0
                    and tom_no_fit[event, pix2_ind] < 60
                )
                cond_correction = (
                    tom_corrected[event, pix1_ind] > 0
                    and tom_corrected[event, pix1_ind] < 60
                    and tom_corrected[event, pix2_ind] > 0
                    and tom_corrected[event, pix2_ind] < 60
                )

                if cond_no_correction:  # otherwise will be nan
                    dt_no_correction[i, event] = (
                        tom_no_fit[event, pix1_ind] - tom_no_fit[event, pix2_ind]
                    )

                else:
                    dt_no_correction[i, event] = np.nan

                if cond_correction:
                    dt_corrected[i, event] = (
                        tom_corrected[event, pix1_ind] - tom_corrected[event, pix2_ind]
                    )

                else:
                    dt_corrected[i, event] = np.nan

        # rms_no_fit = np.zeros(output[0].npixels)
        # rms_no_fit_err = np.zeros(output[0].npixels)

        return (
            tom_no_fit,
            tom_corrected,
            pixels_id,
            dt_no_correction,
            dt_corrected,
            pixel_pairs,
        )


class PedestalContainer(NectarCAMContainer):
    """
    Attributes of the PedestalContainer class that store various data related to the pedestal of a NectarCAM event.

    Attributes:
        run_number (np.uint16): The run number associated with the waveforms.
        npixels (np.uint16): The number of effective pixels.
        pixels_id (np.ndarray[np.uint16]): The IDs of the pixels.
        ucts_timestamp (np.ndarray[np.uint64]): The UCTS timestamp of the events.
        event_type (np.ndarray[np.uint8]): The trigger event type.
        event_id (np.ndarray[np.uint32]): The event IDs.
        pedestal_hg (np.ndarray[np.float64]): The high gain pedestal per event.
        pedestal_lg (np.ndarray[np.float64]): The low gain pedestal per event.
        rms_ped_hg (np.ndarray[np.float64]): The high gain pedestal RMS per event.
        rms_ped_lg (np.ndarray[np.float64]): The low gain pedestal RMS per event.
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
    event_type = Field(
        type=np.ndarray, dtype=np.uint8, ndim=1, description="trigger event type"
    )
    event_id = Field(type=np.ndarray, dtype=np.uint32, ndim=1, description="event ids")

    pedestal_hg = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="High gain pedestal per event",
    )

    pedestal_lg = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="Low gain pedestal per event",
    )

    rms_ped_hg = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="High gain pedestal rms per event",
    )

    rms_ped_lg = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="Low gain pedestal rms per event",
    )


class PedestalComp(NectarCAMComponent):
    """
    The `PedestalComp` class is a NectarCAMComponent that is responsible for processing the pedestal and RMS of the high and low gain waveforms for each event.

    The `__init__` method initializes the `PedestalComp` class. It sets up several member variables to store pedestal related data such as timestamps, event types, event IDs, pedestal and pedestal rms values for both gains.

    The `__call__` method is called for each event, and it processes the waveforms to calculate the pedestal and RMS for the high and low gain channels. The results are stored in the class attributes `__pedestal_hg`, `__pedestal_lg`, `__rms_ped_hg`, and `__rms_ped_lg`.

    The `finish` method is called at the end of processing, and it returns a `PedestalContainer` object containing the calculated pedestal and RMS values, as well as other event information.
    """

    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        ## If you want you can add here members of MyComp, they will contain interesting quantity during the event loop process

        self.__ucts_timestamp = []
        self.__event_type = []
        self.__event_id = []

        self.__pedestal_hg = []

        self.__pedestal_lg = []

        self.__rms_ped_hg = []
        self.__rms_ped_lg = []

    ##This method need to be defined !
    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        self.__event_id.append(np.uint32(event.index.event_id))

        # print(event.trigger.event_type)

        self.__event_type.append(event.trigger.event_type.value)
        self.__ucts_timestamp.append(event.nectarcam.tel[0].evt.ucts_timestamp)

        wfs = []
        wfs.append(event.r0.tel[0].waveform[constants.HIGH_GAIN][self.pixels_id])
        wfs.append(event.r0.tel[0].waveform[constants.LOW_GAIN][self.pixels_id])

        #####THE JOB IS HERE######

        for i, (pedestal, rms_pedestal) in enumerate(
            zip(
                [self.__pedestal_hg, self.__pedestal_lg],
                [self.__rms_ped_hg, self.__rms_ped_lg],
            )
        ):
            wf = np.array(wfs[i])  # waveform per gain

            ped = np.array([np.mean(wf[pix]) for pix in range(len(self.pixels_id))])
            rms_ped = np.array([np.std(wf[pix]) for pix in range(len(self.pixels_id))])

            ped[ped > 1000] = np.nan
            rms_ped[ped > 1000] = np.nan
            pedestal.append(ped)
            rms_pedestal.append(rms_ped)

    ##This method need to be defined !
    def finish(self):
        output = PedestalContainer(
            run_number=PedestalContainer.fields["run_number"].type(self._run_number),
            npixels=PedestalContainer.fields["npixels"].type(self._npixels),
            pixels_id=PedestalContainer.fields["pixels_id"].dtype.type(self.pixels_id),
            ucts_timestamp=PedestalContainer.fields["ucts_timestamp"].dtype.type(
                self.__ucts_timestamp
            ),
            event_type=PedestalContainer.fields["event_type"].dtype.type(
                self.__event_type
            ),
            event_id=PedestalContainer.fields["event_id"].dtype.type(self.__event_id),
            pedestal_hg=PedestalContainer.fields["pedestal_hg"].dtype.type(
                self.__pedestal_hg
            ),
            pedestal_lg=PedestalContainer.fields["pedestal_lg"].dtype.type(
                self.__pedestal_lg
            ),
            rms_ped_hg=PedestalContainer.fields["rms_ped_hg"].dtype.type(
                self.__rms_ped_hg
            ),
            rms_ped_lg=PedestalContainer.fields["rms_ped_lg"].dtype.type(
                self.__rms_ped_lg
            ),
        )
        return output


class PedestalTool(EventsLoopNectarCAMCalibrationTool):
    """
    This class is a part of the PedestalTool, which is an EventsLoopNectarCAMCalibrationTool.

    The finish() method opens the output file, which is an HDF5 file, and extracts the `rms_ped_hg` (root mean square of the high gain pedestal) values from the `PedestalContainer` dataset. Finally, it closes the output file and returns the list of `rms_ped_hg` values.

    This method is used to post-process the output of the PedestalTool and extract specific information from the generated HDF5 file.
    """

    name = "PedestalTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["PedestalComp"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def _init_output_path(self):
        if self.max_events is None:
            filename = f"{self.name}_run{self.run_number}.h5"
        else:
            filename = f"{self.name}_run{self.run_number}_maxevents{self.max_events}.h5"
        self.output_path = pathlib.Path(
            f"{os.environ.get('NECTARCAMDATA','/tmp')}/tests/{filename}"
        )

    def finish(self, *args, **kwargs):
        super().finish(return_output_component=False, *args, **kwargs)
        # print(self.output_path)
        output_file = h5py.File(self.output_path)

        rms_ped_hg = []

        for thing in output_file:
            group = output_file[thing]
            dataset = group["PedestalContainer"]
            data = dataset[:]
            # print("data",data)
            for tup in data:
                try:
                    rms_ped_hg.extend(tup[8])
                except:
                    break

        output_file.close()

        return rms_ped_hg


class UCTSContainer(NectarCAMContainer):
    """
    Defines the fields for the UCTSContainer class, which is used to store various data related to UCTS events.

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
    """
    The `__init__` method initializes the `UCTSComp` class, which is a NectarCAMComponent. It sets up several member variables to store UCTS related data, such as timestamps, event types, event IDs, busy counters, and event counters.

    The `__call__` method is called for each event, and it appends the UCTS-related data from the event to the corresponding member variables.

    The `finish` method creates and returns a `UCTSContainer` object, which is a container for the UCTS-related data that was collected during the event loop.
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
        ## If you want you can add here members of MyComp, they will contain interesting quantity during the event loop process

        self.__ucts_timestamp = []
        self.__event_type = []
        self.__event_id = []
        self.__ucts_busy_counter = []
        self.__ucts_event_counter = []
        self.excl_muons = None
        self.__mean_event_charge = []

    ##This method need to be defined !
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

    ##This method need to be defined !
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
    """
    The `DeadtimeTestTool` class is an `EventsLoopNectarCAMCalibrationTool` that is used to test the deadtime of NectarCAM.

    The `finish` method is responsible for reading the data from the HDF5 file, extracting the relevant information (UCTS timestamps, event counters, and busy counters), and calculating the deadtime-related metrics. The method returns the UCTS timestamps, the time differences between consecutive UCTS timestamps, the event counters, the busy counters, the collected trigger rate, the total time, and the deadtime percentage.
    """

    name = "DeadtimeTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["UCTSComp"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        super().finish(return_output_component=False, *args, **kwargs)
        # print(self.output_path)
        output_file = h5py.File(self.output_path)

        ucts_timestamps = []
        event_counter = []
        busy_counter = []

        for thing in output_file:
            group = output_file[thing]
            dataset = group["UCTSContainer"]
            data = dataset[:]
            for tup in data:
                try:
                    ucts_timestamps.extend(tup[3])
                    event_counter.extend(tup[7])
                    busy_counter.extend(tup[6])
                except:
                    break
        # print(output_file.keys())
        # tom_mu_all= output[0].tom_mu
        # tom_sigma_all= output[0].tom_sigma
        # ucts_timestamps= np.array(output_file["ucts_timestamp"])
        ucts_timestamps = np.array(ucts_timestamps).flatten()
        # print(ucts_timestamps)
        event_counter = np.array(event_counter).flatten()
        busy_counter = np.array(busy_counter).flatten()
        # print(ucts_timestamps)
        delta_t = [
            ucts_timestamps[i] - ucts_timestamps[i - 1]
            for i in range(1, len(ucts_timestamps))
        ]
        # event_counter = np.array(output_file['ucts_event_counter'])
        # busy_counter=np.array(output_file['ucts_busy_counter'])
        output_file.close()

        time_tot = ((ucts_timestamps[-1] - ucts_timestamps[0]) * u.ns).to(u.s)
        collected_trigger_rate = (event_counter[-1] + busy_counter[-1]) / time_tot
        deadtime_pc = busy_counter[-1] / (event_counter[-1] + busy_counter[-1]) * 100

        return (
            ucts_timestamps,
            delta_t,
            event_counter,
            busy_counter,
            collected_trigger_rate,
            time_tot,
            deadtime_pc,
        )


class TriggerTimingTestTool(EventsLoopNectarCAMCalibrationTool):
    """
    The `TriggerTimingTestTool` class is an `EventsLoopNectarCAMCalibrationTool` that is used to test the trigger timing of NectarCAM.

    The `finish` method is responsible for reading the data from the HDF5 file, extracting the relevant information (UCTS timestamps), and calculating the RMS value of the difference between consecutive triggers. The method returns the UCTS timestamps, the time differences between consecutive triggers for events concerning more than 10 pixels (non-muon related events).
    """

    name = "TriggerTimingTestTool"

    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["UCTSComp"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def setup(self):
        super().setup()
        for component in self.components:
            if isinstance(component, UCTSComp):
                component.excl_muons = True

    def finish(self, *args, **kwargs):
        super().finish(return_output_component=False, *args, **kwargs)
        # print(self.output_path)
        output_file = h5py.File(self.output_path)

        ucts_timestamps = []
        charge_per_event = []

        for thing in output_file:
            group = output_file[thing]
            dataset = group["UCTSContainer"]
            data = dataset[:]
            # print("data",data)
            for tup in data:
                try:
                    ucts_timestamps.extend(tup[3])
                    charge_per_event.extend(tup[4])

                except:
                    break
        # print(output_file.keys())
        # tom_mu_all= output[0].tom_mu
        # tom_sigma_all= output[0].tom_sigma
        # ucts_timestamps= np.array(output_file["ucts_timestamp"])
        ucts_timestamps = np.array(ucts_timestamps).flatten()

        # dt in nanoseconds
        delta_t = [
            ucts_timestamps[i] - ucts_timestamps[i - 1]
            for i in range(1, len(ucts_timestamps))
        ]
        # event_counter = np.array(output_file['ucts_event_counter'])
        # busy_counter=np.array(output_file['ucts_busy_counter'])
        output_file.close()

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
