# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores Bokeh webpage update helpers for the RTA of NectarCAM.
"""

# imports
import time
from functools import partial

# Bokeh imports
from bokeh.plotting import curdoc

# Bokeh RTA imports
from utils.display.histogram import (
    update_display_hist,
    update_display_hist_for_1d,
    update_annulus
)
from utils.display.timeline import update_timelines
from utils.display.camera_mapping import update_camera_display
from utils.display.summary import update_summary_card


__all__ = ["periodic_update_display", "start_periodic_updates", "stop_periodic_updates"]


def update_all_figures(
        file,
        display_registry,
        widgets
):
    """Update all displays in display_registry according to their ._meta
    using data from file and widgets.
    Only figures are updated (widgets are left untouched).

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.

    Returns
    -------
    out : None
    
    """

    for disp in display_registry:
        meta = getattr(disp, "_meta", None)
        if not meta:
            continue

        try:
            dtype = meta.get("type", "").lower()

            if dtype == "summary_card":
                run_widget = widgets.get("camera_run")
                run_index = int(run_widget.value) if getattr(run_widget, "value", None) is not None else -1
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                update_summary_card(disp, file, parent, child, run_index = run_index)

            
            if dtype == "hist_1d":
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                label = meta.get("label", meta.get("childkey", "value"))
                # read widget values (fall back to meta defaults)
                n_bins_widget = widgets.get("hist_bins")
                n_bins = int(n_bins_widget.value) if getattr(n_bins_widget, "value", None) is not None else meta.get("n_bins", 50)
                update_display_hist_for_1d(disp, parent, child, file, label, n_bins)

            elif dtype == "hist_avg":
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                label = meta.get("label", meta.get("childkey", "value"))
                # read widget values (fall back to meta defaults)
                n_runs_widget = widgets.get("hist_runs")
                n_bins_widget = widgets.get("hist_bins")
                n_runs = int(n_runs_widget.value) if getattr(n_runs_widget, "value", None) is not None else meta.get("n_runs", 1)
                n_bins = int(n_bins_widget.value) if getattr(n_bins_widget, "value", None) is not None else meta.get("n_bins", 50)
                update_display_hist(disp, parent, child, file, label, n_runs, n_bins)

            elif dtype == "annulus":
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                update_annulus(disp, parent, child, file)

            elif dtype == "camera":
                # choose run index from widget if present
                run_widget = widgets.get("camera_run")
                run_index = int(run_widget.value) if getattr(run_widget, "value", None) is not None else -1
                image_parent = meta.get("image_parentkey")
                param_parent = meta.get("parameter_parentkeys")
                child = meta.get("childkey")
                param = meta.get("parameterkeys")
                update_camera_display(
                    disp, child, image_parent, param_parent, param, file, run_index = run_index
                )

            elif dtype.startswith("timeline"):
                parent = meta.get("parentkey")
                child = meta.get("childkey")
                time_parent = meta.get("time_parentkey")
                time_child = meta.get("time_childkey")
                update_timelines(disp, parent, child, time_parent, time_child, file)

        except Exception as e:
            print("update_all_figures: update failed for display meta=", meta, "error=", e)

def update_timestamp(status_col):
    """Update the time of last update of the page.

    Parameters
    ----------
    status_col : column
        Bokeh column layout for the status of the webpage.

    Returns
    -------
    out : None

    """

    ts = time.strftime('%H:%M:%S')
    print(f"Real time mode: updating figure - {ts}")
    status_col.children[1].text = f"Last update: {ts}"

def periodic_update_display(file, display_registry, widgets, status_col):
    """Update all the figures of the webpage and the status divider.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    status_col : column
        Bokeh column layout for the status of the webpage.

    Returns
    -------
    out : None

    """

    update_all_figures(file, display_registry, widgets)
    update_timestamp(status_col)

def start_periodic_updates(
        file, display_registry, widgets, status_col, interval_ms=1000
    ):
    """Start the periodic update of the webpage every ``interval_ms`` milliseconds.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    status_col : column
        Bokeh column layout for the status of the webpage.
    interval_ms : int, optional
        Interval between each refresh, in milliseconds.
        Default is 1000, i.e. every second.

    Returns
    -------
    periodic_cb_id : callback
        Callback function used for the update

    """

    periodic_cb_id = curdoc().add_periodic_callback(
        partial(
            periodic_update_display,
            file=file,
            display_registry=display_registry,
            widgets=widgets,
            status_col=status_col
        ),
        interval_ms
    )
    widgets["PERIODIC_CB_ID"] = periodic_cb_id
    # print(f"Periodic updates started (id={periodic_cb_id}, interval_ms={interval_ms})")
    return periodic_cb_id


def stop_periodic_updates(widgets):
    """Stop the periodic update of the webpage.

    Parameters
    ----------
    widgets : list
        Storage of all interactive widgets for manual update.

    Returns
    -------
    out : None

    """

    try:
        curdoc().remove_periodic_callback(widgets["PERIODIC_CB_ID"])
    except Exception:
        pass
    widgets["PERIODIC_CB_ID"] = None
    print(f"Periodic updates stopped (id={None})")
    return None
