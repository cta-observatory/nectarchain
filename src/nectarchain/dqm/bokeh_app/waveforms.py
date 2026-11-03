import collections
import json
import os
from urllib.parse import urlparse

import numpy as np

# bokeh imports
from bokeh.layouts import column
from bokeh.models import Div, TabPanel
from bokeh.plotting import figure

from nectarchain.dqm.bokeh_app.logging_config import setup_logger

base_dir = os.path.abspath(os.path.dirname(__file__))
labels_path = os.path.join(base_dir, "data", "labels.json")


logger = setup_logger()


def make_waveforms(waveforms_data, runid=None):
    """Make waveform plots with both average and individual pixel waveforms

    Parameters
    ----------
    waveforms_data : dict
        Dictionary containing categorized waveform data with two keys:
        - 'average': dict of average waveforms
        - 'all': dict of all (non-average) waveforms
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAQM_Run6310'.
        By default None

    Returns
    -------
    dict
        Nested dictionary containing line plots for the waveforms and their headers
    """

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)
        y_axis_labels = labels["y_axis_labels_waveforms"]
        descriptions = labels["descriptions"]
        urls = labels["processors-urls"]

    waveforms = collections.defaultdict(dict)
    average_waveforms = waveforms_data["average"]
    all_waveforms = waveforms_data["all"]

    LOWER_PERCENTILE = 2.5  # For 95% confidence band
    UPPER_PERCENTILE = 97.5

    colors_phy = ["blue", "turquoise"]
    colors_ped = ["red", "orange"]

    for parentkey in average_waveforms.keys():
        colors = colors_phy if "PHY" in parentkey else colors_ped

        # Find corresponding all_waveforms key by removing -AVERAGE-
        all_parentkey = parentkey.replace("-AVERAGE-PIX", "")

        for childkey in average_waveforms[parentkey].keys():
            logger.info(f"Run id {runid}, preparing plot for {parentkey}, {childkey}")

            all_childkey = childkey.replace("-AVERAGE-PIX", "")

            samples = np.arange(len(average_waveforms[parentkey][childkey]))
            avg_waveform = average_waveforms[parentkey][childkey]

            # Determine y-range from average waveform first
            y_min = np.min(avg_waveform) - np.abs(np.min(avg_waveform)) * 0.1
            y_max = np.max(avg_waveform) + np.abs(np.max(avg_waveform)) * 0.1

            # Adjust y-range if individual waveforms are available
            if (
                all_parentkey in all_waveforms.keys()
                and all_childkey in all_waveforms[all_parentkey]
            ):
                pixel_waveforms = all_waveforms[all_parentkey][all_childkey]
                lower_bound = np.percentile(pixel_waveforms, LOWER_PERCENTILE, axis=0)
                upper_bound = np.percentile(pixel_waveforms, UPPER_PERCENTILE, axis=0)
                y_min = min(
                    y_min, np.min(lower_bound) - np.abs(np.min(lower_bound)) * 0.1
                )
                y_max = max(
                    y_max, np.max(upper_bound) + np.abs(np.max(upper_bound)) * 0.1
                )

            # Create figure once with proper ranges
            fig = figure(
                title=all_childkey,
                x_range=(np.min(samples) - 5, np.max(samples) + 5),
                y_range=(y_min, y_max),
            )

            # Plot individual pixel waveforms with low alpha
            if (
                all_parentkey in all_waveforms.keys()
                and all_childkey in all_waveforms[all_parentkey]
            ):
                pixel_waveforms = all_waveforms[all_parentkey][all_childkey]

                # Compute percentiles across all pixels for each sample
                lower_bound = np.percentile(pixel_waveforms, LOWER_PERCENTILE, axis=0)
                upper_bound = np.percentile(pixel_waveforms, UPPER_PERCENTILE, axis=0)

                # Plot shaded 95% percentile band
                x_band = np.concatenate([samples, samples[::-1]])
                y_band = np.concatenate([lower_bound, upper_bound[::-1]])
                fig.patch(
                    x=x_band,
                    y=y_band,
                    fill_color=colors[0],
                    fill_alpha=0.2,
                    line_color=None,
                    legend_label=f"{int(UPPER_PERCENTILE-LOWER_PERCENTILE)}% range",
                )

                # TODO: need to optimise the outliers plotting, still too slow...
                # # Identify outlier waveforms (any point outside the band)
                # is_outlier = np.any(
                #     (pixel_waveforms < lower_bound) | (pixel_waveforms > upper_bound),
                #     axis=1
                # )
                # outlier_indices = np.where(is_outlier)[0]
                # n_outliers = len(outlier_indices)

                # logger.info(f"Found {n_outliers} outlier
                # waveforms out of {pixel_waveforms.shape[0]}")

                # # Plot outliers in turquoise
                # for idx in outlier_indices:
                #     fig.line(
                #         x=samples,
                #         y=pixel_waveforms[idx],
                #         line_width=0.5,
                #         color=colors[1],
                #         alpha=0.3
                #     )

                # # Add legend entry for outliers
                # if n_outliers > 0:
                #     fig.line(
                #         x=[], y=[],
                #         line_width=0.5,
                #         color=colors[1],
                #         alpha=0.3,
                #         legend_label=f'Outliers ({n_outliers})'
                #     )

            # Plot average waveform (thick, opaque line)
            fig.line(
                x=samples,
                y=avg_waveform,
                line_width=3,
                color=colors[0],
                legend_label="Average",
            )

            # Set axis labels and styling
            fig.xaxis.axis_label = "Waveform sample number"
            try:
                fig.yaxis.axis_label = y_axis_labels[parentkey]
                description = descriptions[parentkey]
                url = urls[parentkey]
            except ValueError:
                fig.yaxis.axis_label = ""
                description = ""
                url = ""
            except KeyError:
                fig.yaxis.axis_label = ""
                description = ""
                url = ""

            filename = urlparse(url).path.strip("/").split("/")[-1]

            header = Div(
                text=(
                    f'<div style="font-size: 13pt; font-weight: bold; background-color: lavenderblush; padding: 6px; border-radius: 4px;">{childkey}</div>'  # noqa
                    f'<div style="font-size: 11pt; margin-top: 4px; background-color: lavenderblush; padding: 6px; border-radius: 4px;">'  # noqa
                    f'{description},\nextracted by the nectarchain DQM processor: <a href="{url}" target="_blank">{filename}</a></div>'  # noqa
                    f"</div>"
                )
            )

            fig.title = ""
            fig.xaxis.axis_label_text_font_size = "12pt"
            fig.yaxis.axis_label_text_font_size = "12pt"
            fig.xaxis.major_label_text_font_size = "10pt"
            fig.yaxis.major_label_text_font_size = "10pt"
            fig.xaxis.axis_label_text_font_style = "normal"
            fig.yaxis.axis_label_text_font_style = "normal"

            # waveforms[parentkey][childkey] = fig
            waveforms[parentkey][childkey] = [fig, header]

    logger.info(f"Successfully created waveform plots for run {runid}")

    return dict(waveforms)


def update_waveforms(waveforms_data, runid=None):
    """Reset each waveform previously created by `make_waveforms`

    Parameters
    ----------
    waveforms_data : dict
        Dictionary containing categorized waveform data with two keys:
        - 'average': dict of average waveforms
        - 'all': dict of all (non-average) waveforms
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAMQM_Run6310'.
        By default None

    Returns
    -------
    bokeh.models.TabPanel
        Updated TabPanel containing the bokeh layout for the waveform plots
    """

    # Make new waveform plots
    waveforms = make_waveforms(waveforms_data, runid)

    list_waveforms = [
        column(waveforms[parentkey][childkey][1], waveforms[parentkey][childkey][0])
        for parentkey in waveforms.keys()
        for childkey in waveforms[parentkey].keys()
    ]

    layout_waveforms = column(
        list_waveforms,
        sizing_mode="scale_width",
    )

    tab_waveforms = TabPanel(child=layout_waveforms, title="Waveforms")

    return tab_waveforms
