# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage summary card maker for the RTA of NectarCAM.
"""

# imports
import logging
logger = logging.getLogger(__name__)

import numpy as np

# Bokeh imports
from bokeh.models import Div
from bokeh.layouts import column


__all__ = ["make_summary_card", "update_summary_card"]


def make_summary_card(file, display_registry, parentkeys, childkeys, run_index=-1):
    """Make the summary card of the currently observed event.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    parentkeys : dict
        Parent keys of the file to retrieve data.
    childkeys : dict
        Child keys of the file to retrieve data.
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.

    Returns
    -------
    out : column
        Bokeh layout column of the summary card of the seen event.

    """

    tel_id = file[parentkeys["parameter_parentkey"]][childkeys["tel_id"]][run_index]
    event_id = file[parentkeys["parameter_parentkey"]][childkeys["event_id"]][run_index]
    layout_id = np.where(file[parentkeys["layout_parentkey"]][childkeys["tel_id"]] == tel_id)[0][0]
    trigger_id = np.where(file[parentkeys["trigger_parentkey"]][childkeys["event_id"]] == event_id)[0][0]
    name = file[parentkeys["layout_parentkey"]][childkeys["name"]][layout_id].decode("utf-8")
    camera_type = file[parentkeys["layout_parentkey"]][childkeys["camera_type"]][layout_id].decode("utf-8")
    time = file[parentkeys["trigger_parentkey"]][childkeys["time"]][trigger_id]
    timestamp_qns = file[parentkeys["trigger_parentkey"]][childkeys["timestamp_qns"]][trigger_id]
    alt_tel = file[parentkeys["parameter_parentkey"]][childkeys["alt_tel"]][run_index]
    az_tel = file[parentkeys["parameter_parentkey"]][childkeys["az_tel"]][run_index]
    obs_id = file[parentkeys["parameter_parentkey"]][childkeys["obs_id"]][run_index]
    event_type = file[parentkeys["parameter_parentkey"]][childkeys["event_type"]][run_index]
    event_quality = file[parentkeys["parameter_parentkey"]][childkeys["event_quality"]][run_index]
    event_goodness = file[parentkeys["parameter_parentkey"]][childkeys["is_good_event"]][run_index]
    
    title = Div(
        text="<strong>Summary card:</strong>",
    )
    card = Div(
        text=f"""
            <div>
              Telescope: {name} - id: {tel_id}<br>
              Camera: {camera_type}<br>
              Position:<br>
              <dl>
              <dd>altitude: {alt_tel}</dd>
              <dd>azimut: {az_tel}</dd>
              </dl>
              Observation: {obs_id}<br>
              Event:<br>
              <dl>
              <dd>event type: {event_type}</dd>
              <dd>id: {event_id}</dd>
              <dd>event quality: {event_quality}</dd>
              <dd>event goodness: {event_goodness}</dd>
              </dl>
              Observation time: {time} [units]<br>
              Timestamp: {timestamp_qns / 1e9} [s]<br>
            </div>
        """, 
        styles={
            "border": "2px solid #e2e8f0",
            "padding": "12px",
            "border-radius": "8px",
            "width": "300px",
        },
        width=300, height=300
    )

    display = column(title, card)
    display._meta = {
        "type": "summary_card",
        "parentkey": parentkeys,
        "childkey": childkeys,
        "factory": "make_summary_card"
    }
    display_registry.append(display)

    return display


def update_summary_card(disp, file, parentkeys, childkeys, run_index=-1):
    """Make the summary card of the currently observed event.

    Parameters
    ----------
    disp : column
        Summary card to display in the shape of a Bokeh layout column.
    file : dict_like
        Data of the considered run.
    parentkeys : dict
        Parent keys of the file to retrieve data.
    childkeys : dict
        Child keys of the file to retrieve data.
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.

    """

    try:
        if hasattr(disp, "children") and len(disp.children) > 1:
            if hasattr(disp.children[1], "text"):
                
                tel_id = file[parentkeys["parameter_parentkey"]][childkeys["tel_id"]][run_index]
                event_id = file[parentkeys["parameter_parentkey"]][childkeys["event_id"]][run_index]
                layout_id = np.where(file[parentkeys["layout_parentkey"]][childkeys["tel_id"]] == tel_id)[0][0]
                trigger_id = np.where(file[parentkeys["trigger_parentkey"]][childkeys["event_id"]] == event_id)[0][0]
                name = file[parentkeys["layout_parentkey"]][childkeys["name"]][layout_id].decode("utf-8")
                camera_type = file[parentkeys["layout_parentkey"]][childkeys["camera_type"]][layout_id].decode("utf-8")
                time = file[parentkeys["trigger_parentkey"]][childkeys["time"]][trigger_id]
                timestamp_qns = file[parentkeys["trigger_parentkey"]][childkeys["timestamp_qns"]][trigger_id]
                alt_tel = file[parentkeys["parameter_parentkey"]][childkeys["alt_tel"]][run_index]
                az_tel = file[parentkeys["parameter_parentkey"]][childkeys["az_tel"]][run_index]
                obs_id = file[parentkeys["parameter_parentkey"]][childkeys["obs_id"]][run_index]
                event_type = file[parentkeys["parameter_parentkey"]][childkeys["event_type"]][run_index]
                event_quality = file[parentkeys["parameter_parentkey"]][childkeys["event_quality"]][run_index]
                event_goodness = file[parentkeys["parameter_parentkey"]][childkeys["is_good_event"]][run_index]   

                disp.children[1].text = f"""
                    <div>
                    Telescope: {name} - id: {tel_id}<br>
                    Camera: {camera_type}<br>
                    Position:<br>
                    <dl>
                    <dd>altitude: {alt_tel}</dd>
                    <dd>azimut: {az_tel}</dd>
                    </dl>
                    Observation: {obs_id}<br>
                    Event:<br>
                    <dl>
                    <dd>event type: {event_type}</dd>
                    <dd>id: {event_id}</dd>
                    <dd>event quality: {event_quality}</dd>
                    <dd>event goodness: {event_goodness}</dd>
                    </dl>
                    Observation time: {time} [units]<br>
                    Timestamp: {timestamp_qns / 1e9} [s]<br>
                    </div>
                """
            else:
                logger.warning("No .text attribute found.")
        else:
            logger.warning("No .children attribute found.")
    except Exception as e:
        logger.warning(f"Failed to update summary card: {e}")