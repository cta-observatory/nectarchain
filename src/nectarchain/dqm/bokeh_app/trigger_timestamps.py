import collections
import json
import os
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


def make_trigger_timestamps_vs_ids(trigger_events_data, runid=None):
    """Make trigger event timestamps plots

    Parameters
    ----------
    trigger_events_data : dict
        Dictionary containing trigger event data extracted from source.
        Each value is a dict with 'IDs' and 'Timestamps' arrays.
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAMQM_Run6310'.
        By default None

    Returns
    -------
    dict
        Nested dictionary containing line plots
        for the trigger event timestamps
    """

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = json.load(file)
        y_axis_labels = labels["y_axis_labels_trig_timestamps"]
        descriptions = labels["descriptions"]
        urls = labels["processors-urls"]

    trig_timestamps = collections.defaultdict(dict)
    for parentkey, data in trigger_events_data.items():
        # TODO: need to understand what we want to do with TRIGGER-EVENTS-WRONG
        logger.info(f"Run id {runid}, preparing plot for {parentkey}")

        evts_ids = data["IDs"]
        evts_timestamps = data["Timestamps"][np.argsort(evts_ids)]
        evts_ids = np.sort(evts_ids)
        evts_diff_timestamps = np.array([0] + np.diff(evts_timestamps).tolist())
        negative_diff = evts_diff_timestamps < 0

        fig = figure(
            x_range=(np.min(evts_ids), np.max(evts_ids)),
            y_range=(
                np.min(evts_timestamps * 1000),
                np.max(evts_timestamps * 1000),
            ),
            # Convert to ms
        )
        fig.yaxis.formatter = DatetimeTickFormatter(
            seconds="%H:%M:%S",
            minutes="%H:%M:%S",
            hourmin="%H:%M:%S",
            hours="%H:%M:%S",
            days="%H:%M:%S",
            months="%H:%M:%S",
            years="%H:%M:%S",
        )
        fig.scatter(
            x=evts_ids, y=evts_timestamps * 1000, size=2, alpha=0.6, color="blue"
        )
        fig.scatter(
            x=evts_ids[negative_diff],
            y=evts_timestamps[negative_diff] * 1000,
            size=4,
            alpha=0.6,
            color="red",
            legend_label="# of negative Timestamp[i+1] - Timestamp[i]: "
            + f"{len(evts_timestamps[negative_diff])}",
        )
        fig.legend.location = "top_left"
        fig.xaxis.axis_label = "Event ID"

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

        trig_timestamps[parentkey] = [fig, header]

    logger.info(f"Successfully created trigger timestamps plots for run {runid}")

    return dict(trig_timestamps)


def update_trigger_timestamps_vs_ids(trigger_events_data, runid=None):
    """Reset each trigger timestamp plot previously
       created by `make_trigger_timestamps_vs_ids`

    Parameters
    ----------
    trigger_events_data : dict
        Dictionary containing trigger event data extracted from source.
        Each value is a dict with 'IDs' and 'Timestamps' arrays.
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAMQM_Run6310'.
        By default None

    Returns
    -------
    bokeh.models.TabPanel
        Updated TabPanel containing the bokeh layout for the trigger timestamp plots
    """

    # Make new trigger timestamp plots
    trig_timestamps = make_trigger_timestamps_vs_ids(trigger_events_data, runid)

    list_trig_timestamps = [
        column(
            trig_timestamps[parentkey][1],
            trig_timestamps[parentkey][0],
        )
        for parentkey in trig_timestamps.keys()
    ]

    layout_trig_timestamps = column(
        list_trig_timestamps,
        sizing_mode="scale_width",
    )

    tab_trig_timestamps = TabPanel(
        child=layout_trig_timestamps, title="Trigger Timestamps"
    )

    return tab_trig_timestamps
