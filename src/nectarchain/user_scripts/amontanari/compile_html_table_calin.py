import argparse
import os
from glob import glob


def get_plot_name(path_to_plot, run_number):
    return path_to_plot.split(f"run{run_number}/run{run_number}_")[1].split(".png")[0]


def get_html_table_limits():
    table_header = "<table border='1' cellspacing='0' cellpadding='5' style='border-collapse:collapse; width:100%; text-align:left;'>"
    table_row_column_names = """
    <tr style='background-color:#5B3F85; color:white; font-weight:bold;'>
        <th>Plot name</th>
        <th>Plot description / units</th>
        <th style='width:50%'>Example plot</th>
        <th>Notes</th>
        <th>Is already/Goes on the Web App</th>
    </tr>
    """
    table_footer = "</table>"

    return table_header, table_row_column_names, table_footer


def get_separator_row(section_name):
    return f"""


    <tr style='background-color:violet;'>
        <td> <b>{section_name}</b> </td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
    </tr>
    
    
    """


def main(path_to_data):
    path_to_output = path_to_data + "output_calin/"
    if not os.path.exists(path_to_output):
        raise OSError("The output path does not exists. Have you run the DQM?")

    list_available_paths = glob(path_to_output + "*")
    list_available_runs = [
        int(path_.split("run")[1].split("-")[0]) for path_ in list_available_paths
    ]

    print("These are the runs for which the output is already available:")
    for ii, run in enumerate(list_available_runs):
        print(ii, run)
    chosen_run_index = int(
        input(
            "Please type the index of the run you wish to explore, "
            "exactly as it appeared in the list: "
        )
    )
    chosen_path = list_available_paths[chosen_run_index]

    run_number = int(chosen_path.split("run")[1].split("-")[0])

    list_available_plots = glob(
        chosen_path + "/" + chosen_path.split("calin/")[1].split("-")[0] + "/*.png"
    )

    table_header, table_row_column_names, table_footer = get_html_table_limits()

    html_table_syntax = table_header + table_row_column_names

    sections = {
        "charge": [],
        "trigger": [],
        "temperature": [],
        "psd": [],
        "clock": [],
        "pedestal": [],
        "hvpa": [],
        "waveform": [],
        "event": [],
        "high_low": [],
        "missing_components": [],
        "data": [],
    }

    for path_to_plot in list_available_plots:
        plot_name = get_plot_name(path_to_plot, run_number)
        if "charge" in plot_name:
            sections["charge"].append(plot_name)
        elif "trigger" in plot_name:
            sections["trigger"].append(plot_name)
        elif "temperature" in plot_name:
            sections["temperature"].append(plot_name)
        elif "psd" in plot_name:
            sections["psd"].append(plot_name)
        elif "clock" in plot_name:
            sections["clock"].append(plot_name)
        elif "pedestal" in plot_name:
            sections["pedestal"].append(plot_name)
        elif "hvpa" in plot_name:
            sections["hvpa"].append(plot_name)
        elif "waveform" in plot_name:
            sections["waveform"].append(plot_name)
        elif "event" in plot_name:
            sections["event"].append(plot_name)
        elif "high_low" in plot_name:
            sections["high_low"].append(plot_name)
        elif "missing_components" in plot_name:
            sections["missing_components"].append(plot_name)
        elif "data" in plot_name:
            sections["data"].append(plot_name)

    for section, plots in sections.items():
        html_table_syntax += get_separator_row(section.upper())
        for plot in plots:
            html_table_syntax += f"""
    <tr>
        <td>{plot.upper()}</td>
        <td></td>
        <td>Produced for run {run_number}<br/><img src="run{run_number}_{plot}.png" width="600" alt="" /></td>
        <td></td>
        <td><span style="color:red;">NO/</span><span style="color:orange;">?</span></td>
    </tr>
"""

    html_table_syntax += table_footer

    with open(f"./html_table.html", "w") as f:
        f.write(html_table_syntax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple tool to check the content "
        "of the FITS files after running the DQM"
    )

    parser.add_argument(
        "--path",
        type=str,
        # this should be set to the $NECTARCAMDATA,
        # but I am lazy and the absolute pattern does not always work,
        # so just setting a default for now
        default="../../../../../nectar_cam_data/",
        help="Path to the directory containing the output of the start_dqm.py script",
    )

    args = parser.parse_args()

    main(path_to_data=args.path)
