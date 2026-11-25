import collections
import json
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np

# bokeh imports
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter, Div, TabPanel
from bokeh.plotting import figure

from nectarchain.dqm.bokeh_app.logging_config import setup_logger

base_dir = os.path.abspath(os.path.dirname(__file__))
labels_path = os.path.join(base_dir, "data", "labels.json")


logger = setup_logger()


def make_timelines(timelines_data, runid=None):
    """Make timeline plots for pixel quantities evolving with time

    Parameters
    ----------
    timelines_data : dict
        Dictionary containing timeline data extracted from source.
        Each value is a dict with child keys containing timeline arrays.
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAMQM_Run6310'.
        By default None

    Returns
    -------
    dict
        Nested dictionary containing line plots for the timelines and their headers
    """

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)
        y_axis_labels = labels["y_axis_labels_timelines"]
        descriptions = labels["descriptions"]
        urls = labels["processors-urls"]

    timelines = collections.defaultdict(dict)
    for parentkey, parent_data in timelines_data.items():
        # parent_data is the dict for this parentkey
        for childkey, child_data in parent_data.items():
            if "CAMERA-TEMPERATURE-TREND" not in childkey:
                logger.info(
                    f"Run id {runid}, " f"preparing plot for {parentkey}, {childkey}"
                )
                evts = np.arange(len(child_data))
                fig = figure(
                    x_range=(0, np.max(evts) + 50),
                    y_range=(0, 1),
                    # A fraction is plotted:
                    # y-range values are between 0 and 1 because
                )
                fig.line(
                    x=evts,
                    y=child_data,
                    line_width=3,
                )
                fig.xaxis.axis_label = "Event number"

        if "CAMERA-TEMPERATURE-TREND" in parentkey:
            drawer_ids, temp_trends, drawer_ts = [], [], []
            for childkey, child_data in parent_data.items():
                logger.info(
                    f"Run id {runid}, preparing plot for {parentkey}, {childkey}"
                )
                if childkey == "trend":
                    temp_trends = child_data
                elif childkey == "drawer":
                    drawer_ids = child_data
                elif childkey == "timestamp":
                    drawer_ts = child_data

            try:
                unique_drawer_ids = np.unique(drawer_ids)
                number_of_samples = len(unique_drawer_ids)
                temp_trends_per_drawer = np.reshape(
                    temp_trends,
                    shape=(number_of_samples, int(len(drawer_ids) / number_of_samples)),
                )
                drawer_ts_per_drawer = (
                    np.reshape(
                        drawer_ts,
                        shape=(
                            number_of_samples,
                            int(len(drawer_ids) / number_of_samples),
                        ),
                    )
                    * 1000
                )
                # Convert to ms

                first = datetime.fromtimestamp(
                    drawer_ts_per_drawer.min() / 1000, tz=timezone.utc
                ).strftime("%H:%M:%S")
                last = datetime.fromtimestamp(
                    drawer_ts_per_drawer.max() / 1000, tz=timezone.utc
                ).strftime("%H:%M:%S")

                fig = figure(
                    title=f"From {first} to {last} [UTC]",
                    x_range=(drawer_ts_per_drawer.min(), drawer_ts_per_drawer.max()),
                    y_range=(
                        np.min(temp_trends_per_drawer) - 1,
                        np.max(temp_trends_per_drawer) + 1,
                    ),
                )
                fig.xaxis.formatter = DatetimeTickFormatter(
                    seconds="%H:%M:%S",
                    minutes="%H:%M:%S",
                    hourmin="%H:%M:%S",
                    hours="%H:%M:%S",
                    days="%H:%M:%S",
                    months="%H:%M:%S",
                    years="%H:%M:%S",
                )
                for nd in range(drawer_ts_per_drawer.shape[0]):  # loop over drawers
                    # TODO: this is part of the code that
                    # could be used if we manage to compile the HoverTool
                    # via `compile_hover_tool_val_vs_id_or_sample`
                    # for trend data
                    # data_source = ColumnDataSource(
                    #     data=dict(
                    #         drawer_id=nd,
                    #         sample=samples,
                    #         value=child_data[nd]
                    # ))
                    fig.line(
                        x=drawer_ts_per_drawer[:][nd],
                        y=temp_trends_per_drawer[:][nd],
                        line_width=1,
                        alpha=0.7,
                    )
                fig.xaxis.axis_label = "Timestamps [UTC]"

            except Exception as e:
                logger.warning(
                    f"Caught {type(e).__name__}, creating empty figure. Details: {e}"
                )
                fig = figure(
                    title="No data in the database",
                    x_range=(0, 1),
                    y_range=(0, 1),
                )
                pass

        try:
            fig.yaxis.axis_label = y_axis_labels[parentkey]
            description = descriptions[parentkey]
            url = urls[parentkey]
            title = parentkey
        except ValueError:
            fig.yaxis.axis_label = ""
            description = ""
            url = ""
            title = ""
        except KeyError:
            fig.yaxis.axis_label = ""
            description = ""
            url = ""
            title = ""

        filename = urlparse(url).path.strip("/").split("/")[-1]

        header = Div(
            text=(
                f'<div style="font-size: 13pt; font-weight: bold; background-color: lavenderblush; padding: 6px; border-radius: 4px;">{title}</div>'  # noqa
                f'<div style="font-size: 11pt; margin-top: 4px; background-color: lavenderblush; padding: 6px; border-radius: 4px;">'  # noqa
                f'{description},\nextracted by the nectarchain DQM processor: <a href="{url}" target="_blank">{filename}</a></div>'  # noqa
                f"</div>"
            )
        )

        fig.xaxis.axis_label_text_font_size = "12pt"
        fig.yaxis.axis_label_text_font_size = "12pt"
        fig.xaxis.major_label_text_font_size = "10pt"
        fig.yaxis.major_label_text_font_size = "10pt"
        fig.xaxis.axis_label_text_font_style = "normal"
        fig.yaxis.axis_label_text_font_style = "normal"

        timelines[parentkey][childkey] = [fig, header]

    logger.info(f"Successfully created timeline plots for run {runid}")

    return dict(timelines)


def update_timelines(timelines_data, runid=None):
    """Reset each timeline previously created by `make_timelines`

    Parameters
    ----------
    timelines_data : dict
        Dictionary containing timeline data extracted from source.
        Each value is a dict with child keys containing timeline arrays.
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAMQM_Run6310'.
        By default None

    Returns
    -------
    bokeh.models.TabPanel
        Updated TabPanel containing the bokeh layout for the timeline plots
    """

    # Make new timeline plots
    timelines = make_timelines(timelines_data, runid)

    list_timelines = [
        column(timelines[parentkey][childkey][1], timelines[parentkey][childkey][0])
        for parentkey in timelines.keys()
        for childkey in timelines[parentkey].keys()
    ]

    layout_timelines = column(
        list_timelines,
        sizing_mode="scale_width",
    )

    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    return tab_timelines
