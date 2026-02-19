# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module builds the high level builders for Bokeh webpage of the RTA of NectarCAM.
"""

# imports
import logging
logger = logging.getLogger(__name__)

from functools import partial

# Bokeh imports
from bokeh.models import (
    Div,
    TabPanel,
    Tabs,
    Select,
)
from bokeh.layouts import (
    row,
    column
)
from bokeh.plotting import curdoc

# Bokeh RTA imports
from .display.summary import make_summary_card
from .display.camera_mapping import make_tab_camera_displays
from .display.timeline import make_tab_timelines
from .display.histogram import make_full_histogram_sections
from .display.skymaps import make_tab_skymaps
from .display.header import (
    make_header_menu,
    _on_header_select_change,
    _set_status_text
)

from .data_fetch_helpers import open_file_from_selection


__all__ = ["make_body", "buld_ui"]


def make_body(
    file,
    display_registry,
    widgets,
    summary_parentkeys="dl1/event/subarray/trigger",
    summary_childkeys="time",
    time_parentkey="dl1/event/telescope/images/tel_001",
    time_childkey="trigger_time",
    parentkeys_camera="dl1/event/telescope/images/tel_001",
    childkeys_camera={
        "image_key": "image",
        "peak_time_key": "peak_time"
    },
    parentkeys_parameter="dl1/event/telescope/parameters/tel_001",
    childkeys_parameter={
        "hillas_x_key": "hillas_x",
        "hillas_y_key": "hillas_y",
        "hillas_length_key": "hillas_length",
        "hillas_width_key": "hillas_width",
        "hillas_phi_key": "hillas_phi"
    },
    parentkeys_monitoring="dl1/event/telescope/parameters/tel_001",
    childkeys_monitoring={
        "event_type_key": "event_type",
        "is_good_event_key": "is_good_event",
        "event_quality_key": "event_quality"
    },
    label_2d_timeline={
        "min_func_key": "Min",
        "mean_func_key": "Mean",
        "median_func_key": "Median",
        "max_func_key": "Max"
    },
    func_timeline={
        "min_func_key": "min",
        "mean_func_key": "mean",
        "median_func_key": "median",
        "max_func_key": "max"
    },
    ylabels_2d={
        "image_key": "Image [p.e.]",
        "peak_time_key": "Peak time [ns]"
    },
    ylabels_1d={
        "hillas_x_key": "Hillas: x [unit]",
        "hillas_y_key": "Hillas: y [unit]",
        "hillas_phi_key": "Hillas: phi [unit]",
        "hillas_width_key": "Hillas: width [unit]",
        "hillas_length_key": "Hillas: length [unit]",
        "hillas_max_int_key": "Hillas: maximum intensity [unit]",
        "hillas_total_int_key": "Hillas: total intensity [unit]",
        "hillas_r_key": "Hillas: radius [unit]",
        "hillas_psi_key": "Hillas: psi [unit]",
        "hillas_skew_key": "Hillas: skewness [unit]",
        "hillas_kurt_key": "Hillas: kurtosis [unit]"
    },
    ylabels_step=[
        "Event type",
        "Is good event",
        "Event quality"
    ],
    suptitle_2d="Raw camera data",
    suptitle_1d="Hillas parameters",
    suptitle_step="Event monitoring",
    labels_2d={
        "image_key": "image",
        "peak_time_key": "peak_time"
    },
    labels_1d={
        "hillas_x_key": "hillas_x",
        "hillas_y_key": "hillas_y",
        "hillas_phi_key": "hillas_phi",
        "hillas_width_key": "hillas_width",
        "hillas_length_key": "hillas_length",
        "hillas_max_int_key": "hillas_intensity",
        "hillas_total_int_key": "total_intensity",
        "hillas_r_key": "hillas_r",
        "hillas_psi_key": "hillas_psi",
        "hillas_skew_key": "hillas_skewness",
        "hillas_kurt_key": "hillas_kurtosis"
    },
    labels_colorbar={
        "image_key": "p.e.",
        "peak_time_key": "ns"
    },
    n_runs=1,
    n_bins=20,
    run_index=-1,
):
    """Create the static body of the Bokeh webpage.

    Parameters
    ----------
    file : hdf5 file
        File of data to display.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    summary_parentkeys : string, optional
        Parent key for the summary card data.
    summary_childkeys : string, optional
        Child key for the summary card data.
    time_parentkey : string, optional
        Parent key for the time of events.
    time_childkey : string, optional
        Child key for the time of events.
    parentkeys_camera : list of string, optional
        Parent keys for the camera data.
    childkeys_camera : list of string, optional
        Child keys for the camera data.
    parentkeys_parameter : list of string, optional
        Parent keys for the parameters
        of the Hillas reconstruction.
    childkeys_parameter : list of string, optional
        Child keys for the parameters
        of the Hillas reconstruction.
    parentkeys_monitoring : list of string, optional
        Parent keys for the monitoring of the quality of the event.
    childkeys_monitoring : list of string, optional
        Child keys for the monitoring of the quality of the event.
    label_2d_timeline : list of string, optional
        Names of the statistic functions to use
        for the timelines of 2d data.
    func_timeline : list of numpy functions, optional
        Statistic functions to use
        for the timelines of 2d data.
    ylabels_2d : list of string, optional
        Labels of the y-axes of the timelines from the 2d data.
    ylabels_1d : list of string, optional
        Labels of the y-axes of the timelines from the 1d data.
    ylabels_step : list of string, optional
        Labels of the y-axes of the timelines from the discrete data.
    suptitle_2d : string, optional
        Super title of the 2d timelines section.
    suptitle_1d : string, optional
        Super title of the 1d timelines section.
    suptitle_step : string, optional
        Super title of the discrete timelines section.
    labels_2d : list of string, optional
        List of the labels of the displayed 2d timelines.
    labels_1d : list of string, optional
        List of the labels of the displayed timelines.
    labels_colorbar : list of string, optional
        List of labels for the colorbars of the camera mapping.
    n_runs : int, optional
        Number of latest events to average the histogram on.
        Default is 1.
    n_bins : int, optional
        Number of bins of the histogram.
        Default is 20.
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.

    Returns
    -------
    out : row
        Static initialization of the Bokeh webpage body.

    """
    
    # Summary card
    try:
        summary_card = make_summary_card(
            file,
            parentkeys=summary_parentkeys,
            childkeys=summary_childkeys,
            display_registry=display_registry
        )
    except Exception as e:
        logger.warning("make_summary_card failed:", e)
        summary_card = column(Div(text="Error building summary card"))
    
    # Camera displays
    try:
        tab_camera_displays = make_tab_camera_displays(
            file,
            childkeys=childkeys_camera,
            image_parentkeys=parentkeys_camera,
            parameter_parentkeys=parentkeys_parameter,
            parameterkeys=childkeys_parameter,
            display_registry=display_registry,
            widgets=widgets,
            run_index=run_index,
            labels_colorbar=labels_colorbar
        )
    except Exception as e:
        logger.warning("make_tab_camera_displays failed:", e)
        tab_camera_displays = TabPanel(
            child=column(Div(text="Error building camera display")),
            title="Camera displays"
        )
    
    # Timelines
    try:
        tab_timelines = make_tab_timelines(
            file,
            display_registry=display_registry,
            time_parentkey=time_parentkey,
            time_childkey=time_childkey,
            childkeys_2d=childkeys_camera,
            parentkeys_2d=parentkeys_camera,
            childkeys_1d=childkeys_parameter,
            parentkeys_1d=parentkeys_parameter,
            childkeys_step=childkeys_monitoring,
            parentkeys_step=parentkeys_monitoring,
            labels_2d=label_2d_timeline,
            funcs=func_timeline,
            suptitle_2d=suptitle_2d,
            suptitle_1d=suptitle_1d,
            suptitle_step=suptitle_step,
            ylabels_2d=ylabels_2d,
            ylabels_1d=ylabels_1d,
            ylabels_step=ylabels_step,
        )
    except Exception as e:
        logger.warning("make_tab_timelines failed:", e)
        tab_timelines = TabPanel(
            child=column(Div(text="Error building timeline")),
            title="Timelines"
        )
    
    # Histograms
    tab_histograms = make_full_histogram_sections(
            file,
            display_registry=display_registry,
            widgets=widgets,
            childkeys_avg=childkeys_camera,
            parentkeys_avg=parentkeys_camera,
            childkeys_1d=childkeys_parameter,
            parentkeys_1d=parentkeys_parameter,
            childkeys_pie=childkeys_monitoring,
            parentkeys_pie=parentkeys_monitoring,
            n_runs=n_runs,
            n_bins=n_bins,
            suptitle_avg=suptitle_2d,
            suptitle_1d=suptitle_1d,
            suptitle_pie=suptitle_step,
            titles_avg=ylabels_2d,
            titles_1d=ylabels_1d,
            titles_pie=ylabels_step,
            labels_avg=labels_2d,
            labels_1d=labels_1d,
            xaxes_avg=ylabels_2d,
            xaxes_1d=ylabels_1d
        )
    try:
        tab_histograms = make_full_histogram_sections(
            file,
            display_registry=display_registry,
            widgets=widgets,
            childkeys_avg=childkeys_camera,
            parentkeys_avg=parentkeys_camera,
            childkeys_1d=childkeys_parameter,
            parentkeys_1d=parentkeys_parameter,
            childkeys_pie=childkeys_monitoring,
            parentkeys_pie=parentkeys_monitoring,
            n_runs=n_runs,
            n_bins=n_bins,
            suptitle_avg=suptitle_2d,
            suptitle_1d=suptitle_1d,
            suptitle_pie=suptitle_step,
            titles_avg=ylabels_2d,
            titles_1d=ylabels_1d,
            titles_pie=ylabels_step,
            labels_avg=labels_2d,
            labels_1d=labels_1d,
            xaxes_avg=ylabels_2d,
            xaxes_1d=ylabels_1d
        )
    except Exception as e:
        logger.warning("make_full_histogram_sections failed:", e)
        tab_histograms = TabPanel(
            child=column(Div(text="Error building histogram")),
            title="Histograms"
        )
    
    # Skymaps
    try:
        tab_skymaps = make_tab_skymaps()
    except Exception as e:
        logger.warning("make_tab_skymaps failed:", e)
        tab_skymaps = TabPanel(
            child=column(Div(text="Error building skymap")),
            title="Skymaps"
        )
    
    # Tabs
    tabs = Tabs(
        tabs=[tab_camera_displays, tab_timelines, tab_histograms, tab_skymaps],
    )
    
    # Build layout
    return row(summary_card, tabs)


def build_ui(
        ressource_path,
        file,
        filepath,
        display_registry,
        widgets,
        real_time_tag,
        default_update_ms,
        extension=".h5",
        time_parentkeys=None,
        time_childkeys=None,
        **body_kwargs,      
):
    """Build the user interface of the Bokeh webpage.

    Parameters
    ----------
    ressource_path : string
        Path of the directory where to find the files to list.
        Can be relative or absolute (careful if it is relative,
        might be an issue for portability).
    file : hdf5 file
        File of data to display.
    filepath : string
        Path of the selected file.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    real_time_tag : string
        Tag representing the real-time mode.
        Stored in static.constants.json.
    default_update_ms : int
        Interval between each refresh, in milliseconds.
        Stored in static.constants.json.
    extension : string, optional
        Extension of the format for files.
        Default is .h5.
    time_parentkeys : list of strings, optional
        Parentkeys of data that can be time ordered.
    time_childkeys : list of strings, optional
        Childkeys of data that can be time ordered.
    body_kwargs : dict
        Arguments to pass to ``make_body``.

    Returns
    -------
    header_ret : tuple of (Select, column of Div)
        Full header menu

    """

    # call make_header_menu; be defensive about its return signature
    header_select, status_col = None, None
    header_ret = make_header_menu(ressource_path, real_time_tag, file, extension=extension)
    header_select, status_col = header_ret[0], header_ret[1]

    # If no select found, create a simple select as fallback
    if header_select is None:
        header_select = Select(title="Run", value=real_time_tag, options=[real_time_tag])
        status_col = column(Div(text="(no header provided)", width=600))

    # create initial file: either latest or nothing
    try:
        file, filepath = open_file_from_selection(
            header_select.value if hasattr(header_select, "value") else real_time_tag,
            ressource_path=ressource_path,
            real_time_tag=real_time_tag,
            time_parentkeys=time_parentkeys,
            time_childkeys=time_childkeys
        )
    except Exception:
        file, filepath = None, None
    
    # Build body (try passing the file if make_body accepts it)
    body_ret = None
    try:
        # attempt call with file
        body_ret = make_body(
            file=file, 
            display_registry=display_registry,
            widgets=widgets,
            **body_kwargs
        )
    except Exception as e:
        logger.warning("make_body failed:", e)
        body_ret = Div(text="Error building body")

    # assemble root layout
    root_layout = column(
        row(header_ret[0], header_ret[1]),
        body_ret,
        sizing_mode="scale_width"
        )

    # curdoc().add_root(root_layout)

    # wire select callback
    try:
        header_select.on_change(
            "value",
            partial(
                _on_header_select_change,
                fobj=file,
                fpath=filepath,
                ressource_path=ressource_path,
                status_col=status_col,
                real_time_tag=real_time_tag,
                default_update_ms=default_update_ms,
                display_registry=display_registry,
                widgets=widgets,
                time_parentkeys=time_parentkeys,
                time_childkeys=time_childkeys,
            )
        )
    except Exception:
        pass

    # set status if available
    if status_col is not None:
        _set_status_text(f"Loaded: {filepath}")

    return root_layout, header_ret