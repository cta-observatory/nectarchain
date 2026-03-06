import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from charge_resolution import run_charge_resolution
from deadtime import run_deadtime
from linearity import run_linearity
from lmfit.models import Model

from nectarchain.trr_test_suite.utils import linear_fit_function

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)

try:
    plt.style.use(
        os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "../utils/plot_style_new.mpltstyle",
        )
    )
except FileNotFoundError as e:
    raise e

# TODO: this may or may not be used if we want to save plots at the path
# NECTARCHAIN_FIGURES/trr_camera_X/observation_temperature/
figures_output_path = os.environ.get("NECTARCHAIN_FIGURES", "./")


def get_args():
    """Parses command-line arguments for the linearity test script.

    Returns:
        argparse.ArgumentParser: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="""Intensity resolution B-TEL-1010 using FF+NSB runs.

According to the nectarchain component interface, you have to set a NECTARCAMDATA
environment variable in the folder where you have the data from your runs or where
you want them to be downloaded.

You have to give a list of runs in <run_file>.json, e.g.
`charge_resolution_run_list.json` and pass it to the args corresponding value of voltage
and the NSB value of the sets and an output directory to save the final plot.

If the data are not in `$NECTARCAMDATA`, the files will be downloaded through DIRAC.

For the purposes of testing this script, default data are from the runs used for this
test in the TRR document.

You can optionally specify the number of events to be processed (default 500) and the
number of pixels used (default 1000).
"""
    )
    parser.add_argument(
        "-r",
        "--run_file",
        type=str,
        help="Run file path and name",
        required=False,
        default="resources/observation_temperature_run_list.json",
    )

    parser.add_argument(
        "-m",
        "--run_module",
        type=str,
        help="Module to run",
        choices=["all", "linearity", "charge_resolution", "deadtime"],
        required=False,
        default="all",
    )

    parser.add_argument(
        "-e",
        "--evts",
        type=int,
        help="Number of events to process from each run. Default is 500",
        required=False,
        default=500,
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


class ObservationTemperaturePipeline:
    """Generates the temperature dependent plots"""

    def __init__(self, run_module, run_file, max_events, output_dir, temp_output):
        # Initialise parameters by reading run_file

        self.run_module = run_module
        self.run_file = run_file
        self.max_events = max_events
        self.output_dir = output_dir
        self.temp_output = temp_output

        self.df = pd.read_json(run_file)
        self.df_module = self.df.copy()

    def run(self):
        """
        Dispatches processing to the correct module functions
        based on self.run_module or all modules in the metadata.
        """
        # Determine which modules to run
        if self.run_module.lower() == "all":
            modules_to_run = self.df_module["module"].unique()
        else:
            modules_to_run = [self.run_module]

        # Loop over each module and call the corresponding method
        for module in modules_to_run:
            logging.info(f"Running module: {module}")
            df_module_filtered = self.df_module[self.df_module["module"] == module]

            if module == "charge_resolution":
                self.plot_charge_resolution_temperature(df_module_filtered)
            elif module == "linearity":
                self.plot_linearity_temperature(df_module_filtered)
            elif module == "deadtime":
                self.plot_deadtime_temperature(df_module_filtered)
            else:
                logging.warning(f"No function defined for module: {module}")

    def plot_charge_resolution_temperature(self, df_module_filtered):
        """Plots
        1. Charge as function of FF_v for different temperatures
        2. Charge resolution as function of temperature for different NSB levels
        Computed percentage change in charge resolution with temperature.
        """

        temperatures = sorted(df_module_filtered["temperature"].unique())

        # print(temperatures)
        outputs_comb = []

        for t in temperatures:
            df_temp = df_module_filtered[df_module_filtered["temperature"] == t]

            NSB_list = df_temp["NSB"].unique()

            runs_list = df_temp["runs"].tolist()
            ff_v_list = df_temp["ff_v"].tolist()
            log.info("parameters =============== ", t, NSB_list, runs_list, ff_v_list)

            output = run_charge_resolution(
                NSB=NSB_list,
                runs_list=runs_list,
                ff_v_list=ff_v_list,
                temperature=t,
                nevents=self.max_events,
                output_dir=self.output_dir,
            )
            (
                mean_charge,
                mean_charge_err,
                mean_resolution_nsb,
                mean_resolution_nsb_err,
                ratio_hglg,
            ) = output
            for nsb_index, nsb in enumerate(NSB_list):
                outputs_comb.append(
                    {
                        "temperature": t,
                        "NSB": nsb,
                        "ff_v": np.array(ff_v_list[nsb_index]),
                        "mean_charge": mean_charge[nsb_index],
                        "mean_charge_err": mean_charge_err[nsb_index],
                        "mean_charge_resolution": mean_resolution_nsb[nsb_index],
                        "mean_charge_resolution_err": mean_resolution_nsb_err[
                            nsb_index
                        ],
                    }
                )

        df_output = pd.DataFrame(outputs_comb)
        df_output = df_output.explode(
            [
                "ff_v",
                "mean_charge",
                "mean_charge_err",
                "mean_charge_resolution",
                "mean_charge_resolution_err",
            ]
        )

        colors = plt.cm.viridis(np.linspace(0, 1, len(df_output["ff_v"].unique())))
        ff_list = sorted(df_output["ff_v"].unique())
        print("ff_list ", ff_list)
        marker = ["o", "s", "d", "^", "*"]

        # Calibration curves dependent on temperature
        df_nsb0 = df_output[df_output["NSB"] == 0]
        print(df_nsb0)
        plt.figure()
        for temp_fixed in df_nsb0["temperature"].unique():
            print("temperature ", temp_fixed)
            df_nsb_temp = df_nsb0[df_nsb0["temperature"] == temp_fixed]
            ff_v = df_nsb_temp["ff_v"]
            mean_charge = df_nsb_temp["mean_charge"]

            plt.errorbar(
                ff_v,
                mean_charge,
                yerr=df_nsb_temp["mean_charge_err"],
                marker="o",
                label=f"T={temp_fixed}C",
            )

        plt.legend()
        plt.xlabel("FF voltage (V)")
        plt.ylabel("Charge (p.e)")
        plt.savefig("Charge_ff_v_temp.png")

        # Charge resolution as function of
        # temperature for different FF_voltages and NSBs.

        plt.figure(figsize=(8, 6))
        print(
            "===before loop =====",
            sorted(df_output["ff_v"].unique()),
            enumerate(sorted(df_output["ff_v"].unique())),
        )
        for i, v_fixed in enumerate(ff_list):
            print("====In loop==== ", v_fixed)
            df_v = df_output[df_output["ff_v"] == v_fixed].sort_values("temperature")
            for j, nsb in enumerate(sorted(df_v["NSB"].unique())):
                df_v_nsb = df_v[df_v["NSB"] == nsb]
                plt.plot(
                    df_v_nsb["temperature"],
                    df_v_nsb["mean_charge_resolution"],
                    marker=marker[j],
                    color=colors[i],
                    label=f"FF={v_fixed}V, NSB={nsb}",
                )

        plt.xlabel("Temperature (°C)")
        plt.ylabel("Charge Resolution")
        plt.legend(fontsize=8, ncol=2)
        plt.savefig("ChargeResolution_T_nsb.png")

        for i, v_fixed in enumerate(ff_list):
            df_v = df_output[df_output["ff_v"] == v_fixed]

            # Group by temperature and aggregate over NSB
            grouped = df_v.groupby("temperature")["mean_charge_resolution"]

            mean_res = grouped.mean()
            min_res = grouped.min()
            max_res = grouped.max()

            temps = mean_res.index

            # Plot mean curve
            plt.plot(
                temps,
                mean_res.values,
                color=colors[i],
                marker="o",
                label=f"FF={v_fixed}V",
            )

            # Shade min–max band
            plt.fill_between(
                temps, min_res.values, max_res.values, color=colors[i], alpha=0.25
            )

        plt.xlabel("Temperature (°C)")
        plt.ylabel("Charge Resolution")
        plt.legend(fontsize=8, ncol=2)
        plt.savefig("ChargeResolution_T_nsb_paper.png")

    def plot_linearity_temperature(self, df_module_filtered):
        """Plots
        1. a, b coefficients with temperature
        2. HG/LG with temperature
        """

        results = []

        for _, row in df_module_filtered.iterrows():
            t = row["temperature"]
            runlist = row["runs"]
            transmission = row["transmissions"]

            print(row, "====", t, runlist, transmission)

            fit_parameters, ratio, ratio_std = run_linearity(
                runlist,
                transmission,
                temperature=t,
                nevents=self.max_events,
                output_dir=self.output_dir,
            )

            # unpack HG and LG
            (a_HG, b_HG, a_err_HG, b_err_HG) = fit_parameters[0]
            (a_LG, b_LG, a_err_LG, b_err_LG) = fit_parameters[1]

            results.append(
                {
                    "temperature": t,
                    "a_HG": a_HG,
                    "b_HG": b_HG,
                    "a_err_HG": a_err_HG,
                    "b_err_HG": b_err_HG,
                    "a_LG": a_LG,
                    "b_LG": b_LG,
                    "a_err_LG": a_err_LG,
                    "b_err_LG": b_err_LG,
                    "ratio": ratio,
                    "ratio_std": ratio_std,
                    "transmission": transmission,
                }
            )

        df_lin = pd.DataFrame(results)
        df_lin = df_lin.sort_values("temperature")  # sort values with temperature

        # PLOT 1 : Parameters vs temperature
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(6, 6))

        # Parameter a with temperature
        ax1.errorbar(
            df_lin["temperature"],
            df_lin["a_HG"],
            yerr=df_lin["a_err_HG"],
            marker="o",
            ls=" ",
            label="$a_{HG}$",
        )
        model = Model(linear_fit_function)
        params = model.make_params(a=10, b=0)
        a_HG_fit = model.fit(
            df_lin["a_HG"],
            params,
            weights=1 / df_lin["a_err_HG"],
            x=df_lin["temperature"],
        )
        ax1.plot(
            df_lin["temperature"],
            linear_fit_function(
                df_lin["temperature"],
                a_HG_fit.params["a"].value,
                a_HG_fit.params["b"].value,
            ),
            ls="--",
            label=r" Fit line, slope: $(%2.4f\pm%2.4f)$ $^{\circ} C^{-1}$ "
            % (
                a_HG_fit.params["a"].value,
                a_HG_fit.params["a"].stderr,
            ),
        )
        ax1.errorbar(
            df_lin["temperature"],
            df_lin["a_LG"],
            yerr=df_lin["a_err_LG"],
            marker="s",
            ls=" ",
            label="$a_{LG}$",
        )
        model = Model(linear_fit_function)
        params = model.make_params(a=10, b=0)
        a_LG_fit = model.fit(
            df_lin["a_LG"],
            params,
            weights=1 / df_lin["a_err_LG"],
            x=df_lin["temperature"],
        )
        ax1.plot(
            df_lin["temperature"],
            linear_fit_function(
                df_lin["temperature"],
                a_LG_fit.params["a"].value,
                a_LG_fit.params["b"].value,
            ),
            ls="--",
            label=r" Fit line, slope: $(%2.4f\pm%2.4f)$ $^{\circ} C^{-1}$ "
            % (
                a_LG_fit.params["a"].value,
                a_LG_fit.params["a"].stderr,
            ),
        )

        ax1.legend(fontsize=8)
        ax1.set_xlim(-7, 28)
        # Parameter b vs temperature
        ax2.errorbar(
            df_lin["temperature"],
            df_lin["b_HG"],
            yerr=df_lin["b_err_HG"],
            marker="o",
            ls=" ",
            label="$b_{HG}$",
        )
        model = Model(linear_fit_function)
        params = model.make_params(a=10, b=0)
        b_HG_fit = model.fit(
            df_lin["b_HG"],
            params,
            weights=1 / df_lin["b_err_HG"],
            x=df_lin["temperature"],
        )
        ax2.plot(
            df_lin["temperature"],
            linear_fit_function(
                df_lin["temperature"],
                b_HG_fit.params["a"].value,
                b_HG_fit.params["b"].value,
            ),
            ls="--",
            label=r" Fit line, slope: $(%2.3f\pm%2.3f)$ $^{\circ} p.e. C^{-1}$ "
            % (
                b_HG_fit.params["a"].value,
                b_HG_fit.params["a"].stderr,
            ),
        )
        ax2.errorbar(
            df_lin["temperature"],
            df_lin["b_LG"],
            yerr=df_lin["b_err_LG"],
            marker="s",
            ls=" ",
            label="$b_{LG}$",
        )

        model = Model(linear_fit_function)
        params = model.make_params(a=10, b=0)
        b_LG_fit = model.fit(
            df_lin["b_LG"],
            params,
            weights=1 / df_lin["b_err_LG"],
            x=df_lin["temperature"],
        )
        ax2.plot(
            df_lin["temperature"],
            linear_fit_function(
                df_lin["temperature"],
                b_LG_fit.params["a"].value,
                b_LG_fit.params["b"].value,
            ),
            ls="--",
            label=r" Fit line, slope: $(%2.3f\pm%2.3f)$ $^{\circ} p.e. C^{-1}$ "
            % (
                b_LG_fit.params["a"].value,
                b_LG_fit.params["a"].stderr,
            ),
        )

        ax2.set_xlabel("Temperature ($deg C$)")
        ax2.legend(fontsize=8)
        ax2.set_xlim(-7, 28)
        plt.savefig("Linearity_params_temp.png")

        # PLOT2 : HG/LG as function of different transmissions
        slope_hglg = []
        print("loop", df_lin)
        df_lin = df_lin.explode(["transmission", "ratio", "ratio_std"])
        plt.clf()
        print("before loop ", sorted(df_lin["transmission"].unique()))
        for transmission in sorted(df_lin["transmission"].unique()):
            print(transmission)

            df_lin_transmission = df_lin[
                df_lin["transmission"] == transmission
            ].sort_values("temperature")
            print(df_lin_transmission)
            plt.errorbar(
                df_lin_transmission["temperature"],
                df_lin_transmission["ratio"],
                marker="s",
                alpha=0.8,
                label=f"Tr = {transmission}",
            )

            x = df_lin_transmission["temperature"].to_numpy(dtype=float)
            y = df_lin_transmission["ratio"].to_numpy(dtype=float)
            yerr = df_lin_transmission["ratio_std"].to_numpy(dtype=float)

            y1 = y - yerr
            y2 = y + yerr

            plt.fill_between(x, y1, y2, alpha=0.3)

            model = Model(linear_fit_function)
            params = model.make_params(a=10, b=0)
            ratio_fit = model.fit(
                df_lin_transmission["ratio"],
                params,
                weights=1 / df_lin_transmission["ratio_std"],
                x=df_lin_transmission["temperature"],
            )
            slope_hglg.append(ratio_fit.params["a"].value)
        slope_hglg = np.array(slope_hglg)
        plt.text(
            0.05,
            0.98,
            r"Avg slope : $(%2.3f\pm%2.3f)$"
            % (np.mean(slope_hglg), np.std(slope_hglg) / np.sqrt(len(slope_hglg))),
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            fontsize=6,
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9
            ),
        )
        plt.xlabel("Temperature $deg C")
        plt.ylabel("HG/LG")
        plt.legend(fontsize=5, ncol=3)
        plt.xlim(-10, 30)
        plt.ylim(-1, 20)
        plt.savefig("HGLGRatio_Linearity_temp.png")

    def plot_deadtime_temperature(self, df_module_filtered):
        """Plots
        1. Deadtime as function of run temperatures
        2. Relative difference between collected and fitted trigger rates
        """

        temperatures = np.sort(df_module_filtered["temperatures"].tolist()[0])
        argsorted_temps = np.argsort(df_module_filtered["temperatures"].tolist()[0])
        run_list = np.array(df_module_filtered["runs"].tolist()[0])[
            argsorted_temps
        ].tolist()
        source_ids = np.array(df_module_filtered["source_ids"].tolist()[0])[
            argsorted_temps
        ]
        intensities = np.array(df_module_filtered["intensity"].tolist()[0])[
            argsorted_temps
        ].tolist()

        (
            collected_trigger_rates,
            fitted_trigger_rates,
            fitted_trigger_rates_err,
            deadtime,
            deadtime_err,
            _,
            _,
            _,
            _,
        ) = run_deadtime(
            nevents=self.max_events,
            runlist=run_list,
            ids=source_ids,
            output_dir=self.output_dir,
            temp_output=self.temp_output,
        )

        # Deadtime [mus] vs Temperature [°C]
        plt.figure(figsize=(8, 6))
        plt.errorbar(
            x=temperatures,
            y=deadtime,
            yerr=deadtime_err,
            ls="",
            marker="o",
            color="green",
        )
        for ii, intensity in enumerate(intensities):
            plt.text(
                x=temperatures[ii] + 0.1,
                y=deadtime[ii] + 0.02,
                s=f"{intensity} mA",
                ha="left",
                va="bottom",
                fontsize=9,
                color="green",
            )

        plt.xlabel("Temperature (°C)")
        plt.ylabel(r"Deadtime ($\mu s$)")
        plt.ylim(0.65, 1.15)
        plt.xlim(temperatures.min() - 2, temperatures.max() + 2)
        plt.grid()
        plot_path = os.path.join(self.output_dir, "deadtime_vs_temperature.png")
        plt.savefig(plot_path)

        # Relative difference between fitted
        # and collected trigger rates vs Temperature [°C]
        plt.figure(figsize=(8, 6))
        collected_trigger_rates = np.array(collected_trigger_rates) / 1000
        fitted_trigger_rates = np.array(fitted_trigger_rates)
        relative = (
            np.abs(
                (fitted_trigger_rates - collected_trigger_rates)
                / collected_trigger_rates
            )
            * 100
        )
        rate_err = np.array(fitted_trigger_rates_err)
        rate_err = relative * (
            (rate_err) / np.abs(fitted_trigger_rates - collected_trigger_rates)
        )
        plt.errorbar(
            x=temperatures,
            y=relative,
            yerr=rate_err,
            ls="",
            marker="o",
            color="red",
        )
        for ii, intensity in enumerate(intensities):
            plt.text(
                x=temperatures[ii] + 0.1,
                y=relative[ii] + 0.02,
                s=f"{intensity} mA",
                ha="left",
                va="bottom",
                fontsize=9,
                color="red",
            )

        plt.axhline(10, ls="--", lw=2, color="grey")
        plt.text(
            x=temperatures.min() + 0.1,
            y=10 + 0.3,
            s="CTA requirement",
            ha="left",
            va="center",
            fontsize=10,
            color="grey",
        )

        plt.xlabel("Temperature (°C)")
        plt.ylabel(r"Relative Difference (%)")
        plt.ylim(0, 15)
        plt.xlim(temperatures.min() - 2, temperatures.max() + 2)
        plt.grid()
        plot_path = os.path.join(
            self.output_dir, "relative_trigger_difference_vs_temperature.png"
        )
        plt.savefig(plot_path)

        plt.show()


def main():
    parser = get_args()
    args = parser.parse_args()
    print(args)
    run_file = args.run_file

    run_module = args.run_module
    max_events = args.evts
    output_dir = args.output
    temp_output = args.temp_output

    sys.argv = sys.argv[:1]

    pipeline = ObservationTemperaturePipeline(
        run_module, run_file, max_events, output_dir, temp_output
    )

    pipeline.run()


if __name__ == "__main__":
    main()
