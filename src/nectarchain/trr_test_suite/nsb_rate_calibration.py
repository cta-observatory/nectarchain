# don't forget to set environment variable NECTARCAMDATA

import argparse
import logging
import os
import pathlib
import pickle
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from ctapipe.containers import EventType, Field
from ctapipe.core.traits import ComponentNameList
from ctapipe_io_nectarcam import N_SAMPLES
from ctapipe_io_nectarcam.containers import NectarCAMDataContainer
from scipy.optimize import curve_fit

from nectarchain.data.container import NectarCAMContainer
from nectarchain.makers import DelimiterLoopNectarCAMCalibrationTool
from nectarchain.makers.component import NectarCAMComponent
from nectarchain.trr_test_suite.utils import get_bad_pixels_list, linear_fit_function
from nectarchain.utils.constants import ALLOWED_CAMERAS, GAIN_DEFAULT

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)

plt.style.use(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../utils/plot_style.mpltstyle"
    )
)


def get_args():
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
        "--run",
        type=int,
        help="run number",
        required=False,
        default=6189,
    )
    parser.add_argument(
        "-c",
        "--camera",
        choices=ALLOWED_CAMERAS,
        default=[camera for camera in ALLOWED_CAMERAS if "QM" in camera][0],
        help="Process data for a specific NectarCAM camera.",
        type=str,
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
        help="Output base directory",
        required=False,
        default=f"{os.environ.get('NECTARCHAIN_FIGURES', f'/tmp/{os.getpid()}')}",
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
        pedestal_std (np.ndarray[np.float64]): Standard deviation of pedestal per event.
        pedestal_mean (np.ndarray[np.float64]): Mean pedestal per event.
    """

    run_number = Field(
        type=np.uint16,
        description="run number associated to the waveforms",
    )

    pedestal_std = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="Standard deviation of pedestal per event",
    )

    pedestal_mean = Field(
        type=np.ndarray,
        dtype=np.float64,
        ndim=2,
        description="Mean pedestal per event",
    )


class NSBRateComponent(NectarCAMComponent):
    def __init__(self, subarray, config=None, parent=None, *args, **kwargs):
        super().__init__(
            subarray=subarray, config=config, parent=parent, *args, **kwargs
        )
        self.__run_number__ = []
        self.__pedestal_std__ = []
        self.__pedestal_mean__ = []
        self.__wf_sum__ = []

    def __call__(self, event: NectarCAMDataContainer, *args, **kwargs):
        if event.trigger.event_type == EventType.SKY_PEDESTAL:
            self.__wf_sum__.append(event.r0.tel[self.tel_id].waveform[0].T.sum(axis=0))

        elif event.trigger.event_type == EventType.UNKNOWN:
            self.__wf_sum__ = np.array(self.__wf_sum__)

            self.__pedestal_mean__.append(np.mean(self.__wf_sum__, axis=0))
            self.__pedestal_std__.append(np.std(self.__wf_sum__, axis=0))
            self.__wf_sum__ = []

    def finish(self):
        output = NSBRateContainer(
            run_number=NSBRateContainer.fields["run_number"].type(self._run_number),
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
        output_file = h5py.File(self.output_path)

        pedestal_std = []

        for i in range(1, len(output_file.keys()) + 1):
            group_name = f"data_{i}"
            if i == 1:
                group_name = "data"
            group = output_file[group_name]
            dataset = group["NSBRateContainer_0"]
            data = dataset[:]

            pedestal_std.append(data[0][2][0])

        output_file.close()
        return pedestal_std


def main():
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

    run = args.run
    camera = args.camera

    step_current = args.step_current

    output_dir = os.path.join(
        os.path.abspath(args.output),
        f"trr_camera_{camera}/{Path(__file__).stem}",
    )
    os.makedirs(output_dir, exist_ok=True)
    temp_output = os.path.abspath(args.temp_output) if args.temp_output else None

    log.debug(f"Output directory: {output_dir}")
    log.debug(f"Temporary output file: {temp_output}")

    sys.argv = sys.argv[:1]

    log.info(f"PROCESSING RUN {run}")
    output_file_name = pathlib.Path(f"{output_dir}/NSBRateTestTool_run{str(run)}.h5")

    tool = NSBRateTestTool(
        progress_bar=True,
        run_number=run,
        camera=camera,
        log_level=20,
        overwrite=True,
        output_path=output_file_name,
    )
    tool.initialize()
    tool.setup()
    tool.start()
    pedestal_std = tool.finish()

    pedestal_std = np.array(pedestal_std)

    bad_pix = get_bad_pixels_list()
    pedestal_std[:, bad_pix] = np.nan

    Dark_std = pow(pedestal_std[0], 2)

    T_0 = 2.0  # Offset to be subtracted from N_SAMPLES
    NSB_rate = (pow(pedestal_std, 2) - Dark_std) / (
        pow(GAIN_DEFAULT, 2) * (N_SAMPLES - T_0) * pow(10, -9)
    )

    NSB_rate_mean = (np.nanmean(NSB_rate, axis=1)) * pow(10, -9)

    I_ma = step_current * np.arange(0, len(NSB_rate_mean))

    fig, ax = plt.subplots()

    ax.plot(I_ma, np.abs(NSB_rate_mean), marker="o")

    params, covariance = curve_fit(
        linear_fit_function, I_ma, NSB_rate_mean, p0=[pow(10, 7), pow(10, 5)]
    )

    m, c = params
    m_err = np.sqrt(covariance[0, 0])
    c_err = np.sqrt(covariance[1, 1])

    fit_pts = m * I_ma + c

    ax.plot(I_ma, fit_pts, color="red")

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
    ax.text(
        0.05,
        0.98,
        s,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
    )
    ax.set_xlabel("I (mA)")
    ax.set_ylabel("NSB rate (GHz)")

    fig_name = f"NSB_rate_calibration_{run}"
    plot_path = os.path.join(output_dir, f"{fig_name}.png")
    plt.savefig(plot_path)

    if temp_output:
        with open(os.path.join(temp_output, f"plot_{fig_name}.pkl"), "wb") as f:
            pickle.dump(fig, f)

    plt.close("all")


if __name__ == "__main__":
    main()
