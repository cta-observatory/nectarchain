# don't forget to set environment variable NECTARCAMDATA

import argparse
import os
import pathlib
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.containers import EventType, Field
from ctapipe.core.traits import ComponentNameList
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from scipy.optimize import curve_fit

from nectarchain.data.container import NectarCAMContainer
from nectarchain.makers import DelimiterLoopNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent
from nectarchain.trr_test_suite.utils import get_bad_pixels_list, linear_fit_function


def get_args():
    print("getting arguments")
    """Parses command-line arguments for the linearity test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Source calibration for NSB source \n"
        + "According to the nectarchain component interface, \
            you have to set a NECTARCAMDATA environment variable\
                in the folder where you have the data from your runs\
                    or where you want them to be downloaded.\n"
        + "If the data is not in NECTARCAMDATA, the files will be downloaded through\
            DIRAC.\n For the purposes of testing this script, default data is from the\
                runs used for this test in the TRR document.\n"
    )
    parser.add_argument(
        "-r",
        "--run_no",
        type=int,
        help="run number",
        required=False,
        default=6189,
    )

    parser.add_argument(
        "-s",
        "--step_current",
        type=int,
        nargs="+",
        help="steps in which current increases (in mA)",
        required=False,
        default=5,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory. If none, plot will be saved in current directory",
        required=False,
        default="./",
    )
    parser.add_argument(
        "--temp_output", help="Temporary output directory for GUI", default=None
    )

    return parser


class NSBRateContainer(NectarCAMContainer):
    """Attributes of the PedestalContainer class that store various data related to the
    pedestal of a NectarCAM event.

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
    slice_no = Field(
        type=np.uint16,
        description="slice of the dataset",
    )
    pedestal_std = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="High gain pedestal per event",
    )

    pedestal_mean = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="Low gain pedestal per event",
    )


class NSBRateComponent(NectarCAMComponent):
    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        # If you want you can add here members of MyComp, they will contain
        # interesting quantity during the event loop process
        print("initialize")
        self.__run_number__ = []
        self.__slice_no__ = []
        self.__pedestal_std__ = []
        self.__pedestal_mean__ = []
        self.__wf_sum__ = []

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        # print(event.trigger.event_type)
        if event.trigger.event_type == EventType.SKY_PEDESTAL:
            self.__wf_sum__.append(event.r0.tel[self.tel_id].waveform[0].T.sum(axis=0))

        elif event.trigger.event_type == EventType.UNKNOWN:
            # print("here break")
            self.__wf_sum__ = np.array(self.__wf_sum__)

            self.__pedestal_mean__.append(np.mean(self.__wf_sum__, axis=0))
            self.__pedestal_std__.append(np.std(self.__wf_sum__, axis=0))
            self.__wf_sum__ = []

    def finish(self):
        slice_no = 1
        output = NSBRateContainer(
            run_number=NSBRateContainer.fields["run_number"].type(self._run_number),
            slice_no=NSBRateContainer.fields["slice_no"].type(slice_no),
            pedestal_std=NSBRateContainer.fields["pedestal_std"].dtype.type(
                self.__pedestal_std__
            ),
            pedestal_mean=NSBRateContainer.fields["pedestal_mean"].dtype.type(
                self.__pedestal_mean__
            ),
        )
        return output


class NSBRateTestTool(DelimiterLoopNectarCAMCalibrationTool):
    componentsList = ComponentNameList(
        NectarCAMComponent,
        default_value=["NSBRateComponent"],
        help="List of Component names to be apply, the order will be respected",
    ).tag(config=True)

    def finish(self, *args, **kwargs):
        # super().finish(return_output_component=False, *args, **kwargs)
        # print(self.output_path)
        output_file = h5py.File(self.output_path)
        # print(output_file.keys())

        pedestal_std = []

        # for i in range(2,10):

        for i in range(1, len(output_file.keys()) + 1):
            group_name = f"data_{i}"
            if i == 1:
                group_name = "data"
            group = output_file[group_name]
            dataset = group["NSBRateContainer_0"]
            data = dataset[:]

            # print(group_name,data)

            pedestal_std.append(data[0][2][0])

        # for tup in data:
        # pedestal_std.append(tup[2])

        output_file.close()
        return pedestal_std


def main():
    print("In main")
    """
    The `main()` function is the entry point of the NSB calibration code. It parses \
            the command-line arguments, processes the calibration run,\
                  taken using aiv/nectarpy/take_nsb_scan_run.py\
            The run takes PEDESTAL runs seperated by UNKNOWN event type when\
              the NSB configuration changes\
    1. Parses the command-line arguments using the `get_args()` function,\
          which sets up\
            the argument parser and handles the input parameters.\
    2. Processes the run and produces the\
          NSB rate as a function of intensity of NSB source.\
    """
    parser = get_args()
    args = parser.parse_args()

    run = args.run_no

    step_current = args.step_current

    output_dir = os.path.abspath(args.output)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    print(f"Output directory: {output_dir}")  # Debug print
    print(f"Temporary output file: {temp_output}")  # Debug print

    sys.argv = sys.argv[:1]

    # runlist = [3441]

    print("PROCESSING RUN {}".format(run))
    output_file_name = pathlib.Path(
        f"{output_dir}" f"/NSBRateTestTool_run{str(run)}.h5"
    )

    tool = NSBRateTestTool(
        progress_bar=True,
        run_number=run,
        log_level=20,
        overwrite=True,
        output_path=output_file_name,
    )
    tool.initialize()
    tool.setup()
    tool.start()
    pedestal_std = tool.finish()

    pedestal_std = np.array(pedestal_std)

    # print("pedestal std", np.shape(pedestal_std))
    bad_pix = get_bad_pixels_list()
    pedestal_std[:, bad_pix] = np.nan

    # print(np.nanmean(pedestal_std[0]),np.nanmean(pedestal_std[1]))

    Dark_std = pow(pedestal_std[0], 2)

    NSB_rate = (pow(pedestal_std, 2) - Dark_std) / (pow(58, 2) * 58.0 * pow(10, -9))
    # print("NSB_rate",NSB_rate[0], np.nanmean(NSB_rate[0]))

    NSB_rate_mean = (np.nanmean(NSB_rate, axis=1)) * pow(10, -9)
    # print(len(pedestal_std),len(NSB_rate_mean),NSB_rate_mean)

    I_ma = step_current * np.arange(0, len(NSB_rate_mean))

    plt.plot(I_ma, np.abs(NSB_rate_mean), marker="o")

    params, covariance = curve_fit(
        linear_fit_function, I_ma, NSB_rate_mean, p0=[pow(10, 7), pow(10, 5)]
    )

    m, c = params
    m_err = np.sqrt(covariance[0, 0])
    c_err = np.sqrt(covariance[1, 1])

    fit_pts = m * I_ma + c

    plt.plot(I_ma, fit_pts, color="red")

    # Text for plot
    exp_m = int(np.floor(np.log10(abs(m)))) if m != 0 else 0
    scale_m = 10**exp_m
    m_scaled = m / scale_m
    m_err_scaled = m_err / scale_m

    m_err_rounded = float(f"{m_err_scaled:.1g}")
    dec_m = -int(np.floor(np.log10(abs(m_err_rounded)))) if m_err_rounded != 0 else 0
    m_rounded = round(m_scaled, dec_m)

    exp_c = int(np.floor(np.log10(abs(c)))) if c != 0 else 0
    scale_c = 10**exp_c
    c_scaled = c / scale_c
    c_err_scaled = c_err / scale_c

    c_err_rounded = float(f"{c_err_scaled:.1g}")
    dec_c = -int(np.floor(np.log10(abs(c_err_rounded)))) if c_err_rounded != 0 else 0
    c_rounded = round(c_scaled, dec_c)

    s = (
        rf"$m = ({m_rounded} \pm {m_err_rounded})\times 10^{{{exp_m}}}$"
        "\n"
        rf"$c = ({c_rounded} \pm {c_err_rounded})\times 10^{{{exp_c}}}$"
    )
    # print(s)
    plt.text(
        0.05,
        0.98,
        s,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
    )
    plt.xlabel("I (mA)")
    plt.ylabel("NSB rate (GHz)")

    plt.savefig(f"NSB_rate_calibration_{run}.png")
    # print(m,c)


if __name__ == "__main__":
    main()
