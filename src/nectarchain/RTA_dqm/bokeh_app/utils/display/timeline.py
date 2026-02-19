# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage timeline maker for the RTA of NectarCAM.
"""

# imports
import numpy as np
from inspect import isfunction
from collections.abc import Iterable

# Bokeh imports
from bokeh.models import (
    HoverTool,
    TabPanel,
    ColumnDataSource,
    Div,
    Range1d
)
from bokeh.layouts import (
    column,
    gridplot
)
from bokeh.palettes import Inferno

# ctapipe imports
from ctapipe.visualization.bokeh import BokehPlot


__all__ = [
    "make_1d_timelines", "make_2d_timelines", "make_tab_timelines", "update_timelines"
]


def make_2d_timeline(
    file,
    childkey,
    parentkey,
    time_childkey,
    time_parentkey,
    display_registry,
    ylabel=None,
    labels=None,
    funcs=None
):
    """Create the statistics timelines
    based on the 2d-data from the input file.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkey : string
        Child key of the file to retrieve 2d-data.
    parentkey : string
        Parent keys of the file to retrieve 2d-data.
    time_childkey : string
        Child key of the file to retrieve the time axis.
    time_parentkey : string
        Parent key of the file to retrieve the time axis.
    display_registry : list
        Storage of all the displays for later update.
    ylabel : string, optional
        Label of the y axis.
        Default is ``None``, meaning it will be ``childkey``.
    labels : list of string, optional
        List of the labels of the displayed timelines.
        Default is ``None``, meaning only the mean is displayed.
    funcs : list of func, optional
        List of the functions used for the displayed timelines.
        Default is ``None``, meaning only the mean is displayed.

    Returns
    -------
    display : BokehPlot
        Display of the 2d timelines.

    """
    
    if funcs is None:
        funcs = {"func_key": np.mean}
    if labels is None:
        labels = {"func_key": "Mean"}
    if len(labels) != len(funcs):
        funcs = {"func_key": np.mean}
        labels = {"func_key": "Mean"}
    data = np.asarray(file[parentkey][childkey])
    x_axis = np.asarray(file[time_parentkey][time_childkey])
    data = data[x_axis.argsort()]
    x_axis = x_axis[x_axis.argsort()]
    x_axis -= x_axis[-1]
    if data.ndim != 2:
        data = data.reshape((data.shape[0],1))
    if ylabel is None:
        ylabel = childkey
    
    display = BokehPlot(
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        x_range=(x_axis.min(), x_axis.max()),
        toolbar_location="above"
    )
    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.xaxis.axis_label = "Time [unit time]"
    fig.yaxis.axis_label = ylabel.capitalize()

    computed_data = ColumnDataSource(
        data={
            "time": x_axis,
        }
        | {
            labels[key]: funcs[key](data, axis=-1)
            for key in labels.keys()
        }
    )

    colors = Inferno[len(funcs)+2][1:-1]
    for index, key in enumerate(funcs):
        r = fig.line(
            source=computed_data,
            x="time",
            y=labels[key],
            line_width=2,
            name=labels[key],
            color=colors[index],
            alpha=1,
            muted_alpha=.2,
            legend_label=labels[key].capitalize()
        )
        hover = HoverTool(
            tooltips=[(labels[key], "@{}".format(labels[key]))],
            renderers=[r]
        )
        fig.add_tools(hover)

    fig.legend.location = "bottom_left"
    fig.legend.click_policy = "mute"
    fig.hover.mode = "vline"
    display.update()

    display._meta = {
        "type": "timeline_2d", 
        "parentkey": parentkey,
        "childkey": childkey,
        "time_childkey": time_childkey,
        "time_parentkey": time_parentkey,
        "funcs": funcs,       
        "labels": labels,
        "factory": "make_2d_timeline"
    }
    display_registry.append(display)

    return display

def make_2d_timelines(
    file,
    childkeys,
    parentkeys,
    time_childkey,
    time_parentkey,
    display_registry,
    ylabels=[None],
    labels=[None],
    funcs=[[np.mean]],
    suptitle=None
):
    """Create the group of statistics timelines
    based on the 2d-data from the input file.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkeys : list of string
        Child keys of the file to retrieve 2d-data.
    parentkeys : list of string
        Parent keys of the file to retrieve 2d-data.
    time_childkey : string
        Child key of the file to retrieve the time axis.
    time_parentkey : string
        Parent key of the file to retrieve the time axis.
    display_registry : list
        Storage of all the displays for later update.
    ylabel : list of string, optional
        Labels of the y axis.
        Default is ``None``, meaning it will be ``childkeys``.
    labels : list of list of string, optional
        Lists of the labels of the displayed timelines.
        Default is ``None``, meaning only the means are displayed.
    funcs : list of list of func, optional
        Lists of the functions used for the displayed timelines.
        Default is ``None``, meaning only the means are displayed.
    suptitle : string
        Title of the seciotn.
        Default is ``None``, meaning it is set as **Subsection**.

    Returns
    -------
    display : column
        Bokeh layout column of displays of the 2d timelines.

    """

    if isinstance(parentkeys, str):
        parentkeys = {key: parentkeys for key in childkeys.keys()}
    if len(childkeys) != len(parentkeys):
        parentkeys = {key: parentkeys[0] for key in childkeys.keys()}
    if not isinstance(ylabels, dict):
        ylabels = childkeys
    if (
        isinstance(funcs, dict) 
        and set(funcs.keys()) != set(childkeys.keys())
    ):
        funcs = {key:funcs for key in childkeys.keys()}
    elif isfunction(funcs):
        {key:{"func_key": funcs} for key in childkeys.keys()}
    elif isinstance(funcs, Iterable):
        funcs = {
            key:{
                f"func{i}_key": funcs[i]
                for i in range(len(funcs))
            } for key in childkeys.keys()
        }
    if (
        isinstance(labels, dict) 
        and set(labels.keys()) != set(childkeys.keys())
    ):
        labels = {key:labels for key in childkeys.keys()}
    elif isinstance(labels, str):
        {key:{"func_key": labels} for key in childkeys.keys()}
    elif isinstance(labels, Iterable):
        labels = {
            key:{
                f"func{i}_key": labels[i]
                for i in range(len(labels))
            } for key in childkeys.keys()
        }
            
    displays = []
    for key in childkeys.keys():
        displays.append(
            make_2d_timeline(
                file,
                childkeys[key],
                parentkeys[key],
                time_childkey,
                time_parentkey,
                display_registry,
                ylabel=ylabels[key],
                labels=labels[key],
                funcs=funcs[key]
            ).figure
        )

    if suptitle is None:
        suptitle = "<strong>Subsection</strong>"
    else:
        suptitle = "<strong>" + suptitle + "</strong>"
    name = Div(text=suptitle)
    
    return column(name, gridplot(displays, ncols=2))

def make_1d_timeline(
    file,
    childkey,
    parentkey,
    time_childkey,
    time_parentkey,
    display_registry,
    ylabel=None,
    label=None,
    step=False
):
    """Create the timelines
    based on the data from the input file.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkey : string
        Child key of the file to retrieve data.
    parentkey : string
        Parent keys of the file to retrieve data.
    time_childkey : string
        Child key of the file to retrieve the time axis.
    time_parentkey : string
        Parent key of the file to retrieve the time axis.
    display_registry : list
        Storage of all the displays for later update.
    ylabel : string, optional
        Label of the y axis.
        Default is ``None``, meaning it will be ``childkey``.
    label : string, optional
        Labels of the displayed timelines.
        Default is ``None``, meaning it will be ``childkey``.
    step : bool, optional
        If ``True``, display step functions for timelines
        instead of linear functions.
        Default is ``False``.

    Returns
    -------
    display : BokehPlot
        Display of the timelines.

    """

    if label is None:
        label = childkey
    data = np.asarray(file[parentkey][childkey])
    x_axis = np.asarray(file[time_parentkey][time_childkey])
    data = data[x_axis.argsort()]
    x_axis = x_axis[x_axis.argsort()]
    x_axis -= x_axis[-1]

    if data.ndim != 1:
        data = data.reshape((data.shape[0]))
    if ylabel is None:
        ylabel = childkey
    display = BokehPlot(
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        x_range=(x_axis.min(), x_axis.max()),
        toolbar_location="above"
    )
    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.xaxis.axis_label = "Time [unit time]"
    fig.yaxis.axis_label = ylabel.capitalize()

    column_data = ColumnDataSource(
        data={
            "time": x_axis,
            "y": data
        }
    )

    color = Inferno[3][1]
    if step:
        r = fig.step(
            source=column_data,
            x="time",
            y="y",
            line_width=2,
            name=label,
            color=color,
            alpha=1,
            muted_alpha=.2,
            legend_label=label.capitalize()
        )
    else:
        r = fig.line(
            source=column_data,
            x="time",
            y="y",
            line_width=2,
            name=label,
            color=color,
            alpha=1,
            muted_alpha=.2,
            legend_label=label.capitalize()
        )
    hover = HoverTool(
        tooltips=[(label, "@y")],
        renderers=[r]
    )
    fig.add_tools(hover)

    fig.legend.location = "bottom_left"
    fig.legend.click_policy = "mute"
    fig.hover.mode = "vline"
    display.update()

    display._meta = {
        "type": "timeline_1d",  
        "parentkey": parentkey,
        "childkey": childkey,
        "time_childkey": time_childkey,
        "time_parentkey": time_parentkey,
        "labels": label,
        "factory": "make_1d_timeline"
    }
    display_registry.append(display)

    return display

def make_1d_timelines(
    file,
    childkeys,
    parentkeys,
    time_childkey,
    time_parentkey,
    display_registry,
    ylabels=[None],
    labels=[None],
    suptitle=None,
    step=False
):
    """Create the group of timelines
    based on the data from the input file.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkeys : list of string
        Child keys of the file to retrieve data.
    parentkeys : list of string
        Parent keys of the file to retrieve data.
    time_childkey : string
        Child key of the file to retrieve the time axis.
    time_parentkey : string
        Parent key of the file to retrieve the time axis.
    display_registry : list
        Storage of all the displays for later update.
    ylabels : list of string, optional
        Labels of the y axis.
        Default is ``None``, meaning it will be ``childkeys``.
    labels : list of list of string, optional
        Lists of the labels of the displayed timelines.
        Default is ``None``, meaning it will be ``childkeys``.
    suptitle : string, optional
        Name of the subsection, "Subsection" by default.
    step : bool, optional
        If ``True``, display step functions for timelines
        instead of linear functions.
        Default is ``False``.

    Returns
    -------
    display : column
        Bokeh layout column of displays of the timelines.

    """

    if isinstance(parentkeys, str):
        parentkeys = {key: parentkeys for key in childkeys.keys()}
    if len(childkeys) != len(parentkeys):
        parentkeys = {key: parentkeys[0] for key in childkeys.keys()}
    if not isinstance(ylabels, dict):
        ylabels = childkeys
    if not isinstance(labels, dict):
        labels = childkeys
        
    displays = []
    for key in childkeys.keys():
        try:
            displays.append(
                make_1d_timeline(
                    file,
                    childkeys[key],
                    parentkeys[key],
                    time_childkey,
                    time_parentkey,
                    display_registry,
                    ylabel=ylabels[key],
                    label=labels[key],
                    step=step
                ).figure
            )
        except:
            continue
    if suptitle is None:
        suptitle = "Subsection"
    suptitle = "<strong>" + suptitle + "</strong>"
    name = Div(text=suptitle)
            
    return column(name, gridplot(displays, ncols=2))

def make_tab_timelines(
    file,
    display_registry,
    time_childkey,
    time_parentkey,
    childkeys_2d,
    parentkeys_2d,
    childkeys_1d,
    parentkeys_1d,
    childkeys_step,
    parentkeys_step,
    ylabels_2d=[None],
    labels_2d=[None],
    funcs=[[np.mean]],
    suptitle_2d=None,
    ylabels_1d=[None],
    labels_1d=[None],
    suptitle_1d=None,
    ylabels_step=[None],
    labels_step=[None],
    suptitle_step=None,
):
    """Create the tab of the timelines
    based on the data from the input file.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    time_childkey : string
        Child key of the file to retrieve the time axis.
    time_parentkey : string
        Parent key of the file to retrieve the time axis.
    childkeys_2d : list
        Child keys of the file to retrieve 2d data.
    parentkeys_2d : list or string
        Parent keys of the file to retrieve 2d data.
    childkeys_1d : list
        Child keys of the file to retrieve 1d data.
    parentkeys_1d : list or string
        Parent keys of the file to retrieve 1d data.
    childkeys_step : list
        Child keys of the file to retrieve discrete data.
    parentkeys_step : list or string
        Parent keys of the file to retrieve discrete data.
    ylabels_2d : list of string, optional
        Labels of the y axes.
    labels_2d : list of string, optional
        List of the labels of the displayed 2d timelines.
    funcs : list of funcs, optional
        List of the used functions for the statistics of the 2d data.
    suptitle_2d : string, optional
        Super title of the 2d timelines section.
    ylabels_1d : list of string, optional
        Labels of the y axes.
    labels_1d : list of string, optional
        List of the labels of the displayed timelines.
    suptitle_1d : string, optional
        Super title of the 1d timelines section.
    ylabels_step : list of string, optional
        Labels of the y axes.
    labels_step : list of string, optional
        List of the labels of the displayed timelines.
    suptitle_step : string, optional
        Super title of the discrete timelines section.

    Returns
    -------
    display : TabPanel
        Tab of the timelines.

    """

    timeline_2d_layout = make_2d_timelines(
        file,
        childkeys_2d,
        parentkeys_2d,
        time_childkey,
        time_parentkey,
        display_registry,
        ylabels=ylabels_2d,
        labels=labels_2d,
        funcs=funcs,
        suptitle=suptitle_2d
    )
    timeline_1d_layout = make_1d_timelines(
        file,
        childkeys_1d,
        parentkeys_1d,
        time_childkey,
        time_parentkey,
        display_registry,
        ylabels=ylabels_1d,
        labels=labels_1d,
        suptitle=suptitle_1d,
        step=False
    )
    timeline_step_layout = make_1d_timelines(
        file,
        childkeys_step,
        parentkeys_step,
        time_childkey,
        time_parentkey,
        display_registry,
        ylabels=ylabels_step,
        labels=labels_step,
        suptitle=suptitle_step,
        step=True
    )
    timeline_layout = column(timeline_2d_layout, timeline_1d_layout, timeline_step_layout)
    return TabPanel(child=timeline_layout, title="Timelines")

def update_timelines(
        disp, parentkey, childkey, time_parentkey, time_childkey, current_file
    ):
    """Recompute timeline series and
    update any ColumnDataSource found in the figure.

    Parameters
    ----------
    disp: CameraDisplay
        Display of the camera mapping.
    parentkey : string
        Parent key of the file to retrieve timeline data.
    childkey : string
        Child key of the file to retrieve timeline data.
    time_childkey : string
        Child key of the file to retrieve the time axis.
    time_parentkey : string
        Parent key of the file to retrieve the time axis.
    current_file : dict_like
        Data of the considered run.


    Returns
    -------
    out: None
        
    """

    try:
        arr = np.asarray(current_file[parentkey][childkey])
        x_axis = np.asarray(current_file[time_parentkey][time_childkey])
        arr = arr[x_axis.argsort()]
        x_axis = x_axis[x_axis.argsort()]
        x_axis -= x_axis[-1]
    except Exception as e:
        print("_recompute_timeline_display: read failed:", e)
        return

    # compute standard series if they are typical
    try:
        mean = np.nanmean(arr, axis=-1)
        median = np.nanmedian(arr, axis=-1)
        mx = np.nanmax(arr, axis=-1)
        mn = np.nanmin(arr, axis=-1)
    except Exception:
        print("Could not compute any statistics: empty timelines")
        mean = np.array([])
        median = np.array([])
        mx = np.array([])
        mn = np.array([])

    # update any CDS that contains matching keys
    disp.figure.x_range = Range1d(x_axis.min(), x_axis.max())
    for r in getattr(disp.figure, "renderers", []):
        src = getattr(r, "data_source", None)
        if not isinstance(src, ColumnDataSource):
            continue
        # try to update common column names
        data_keys = src.data.keys()
        update_dict = {"time": x_axis}
        if np.all(np.isin(list(data_keys), ["time", "Min", "Median", "Mean", "Max"])):
            update_dict |= {
                "Min": mn,
                "Mean": mean,
                "Median": median,
                "Max": mx
            }
        elif np.all(np.isin(list(data_keys), ["time", "y"])):
            update_dict |= {"y": arr}
        else:
            print("No right data format found")
            return
        src.data = update_dict