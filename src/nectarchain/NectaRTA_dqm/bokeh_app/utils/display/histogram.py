# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage histogram maker for the RTA of NectarCAM.
"""

# imports
import logging
import time

import numpy as np
from bokeh.layouts import column, gridplot

# Bokeh imports
from bokeh.models import (
    AnnularWedge,
    ColumnDataSource,
    Div,
    HoverTool,
    Legend,
    LegendItem,
    Plot,
    Range1d,
    Slider,
    TabPanel,
)
from bokeh.palettes import Inferno
from bokeh.transform import linear_cmap

# ctapipe imports
from ctapipe.visualization.bokeh import BokehPlot

__all__ = [
    "make_full_histogram_sections",
    "update_display_hist",
    "update_display_hist_for_1d",
    "update_annulus",
]

logger = logging.getLogger(__name__)


def make_averaged_histogram(
    file,
    childkey,
    parentkey,
    display_registry,
    n_runs=1,
    n_bins=20,
    title=None,
    label=None,
    xaxis=None,
):
    """Create the Bokeh histogram plot for 2d data.
    This is the data for a specific event, n_runs accounts for
    the number of events to average on for the histogram.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkey : string
        Child key of the file to retrieve 2d data.
    parentkey : string
        Parent key of the file to retrieve 2d data.
    display_registry : list
        Storage of all the displays for later update.
    n_runs : int, optional
        Number of latest events to average the histogram on.
        Default is 1.
    n_bins : int, optional
        Number of bins of the histogram.
        Default is 20.
    title : string, optional
        Title of the histogram.
        Default is ``None``, meaning it is set as ``childkey``.
    label : string, optional
        Label for the histogram.
        Default is ``None``, meaning it is set as ``childkey``.
    xaxis : string, optional
        Label of the x-axis of the histogram.
        Default is ``None``, meaning it is set as ``label``.

    Returns
    -------
    display : BokehPlot
        Bokeh plot of the histogram.

    """

    if title is None:
        title = childkey
    if label is None:
        label = childkey
    if xaxis is None:
        xaxis = label
    data = np.asarray(file[parentkey][childkey])
    if data.ndim != 2:
        data = np.zeros((data.shape[0], 1))

    data_to_average = data[-n_runs:]
    hist, edges = np.histogram(data_to_average, n_bins)
    hist //= n_runs
    current_data = ColumnDataSource(
        data={
            label: hist,
            "edges": (edges[:-1] + edges[1:]) / 2,
        }
    )

    display = BokehPlot(
        title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        x_range=(current_data.data["edges"].min(), current_data.data["edges"].max()),
        toolbar_location="above",
    )

    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.y_range.start = 0
    fig.xaxis.axis_label = xaxis.capitalize()

    r = fig.vbar(
        source=current_data,
        x="edges",
        width=0.9 * (edges[1] - edges[0]),
        top=label,
        color=linear_cmap(label, "Inferno256", -0.1 * np.max(hist), 1.1 * np.max(hist)),
    )

    hover = HoverTool(tooltips=[("Count", "@{}".format(label))], renderers=[r])
    fig.add_tools(hover)

    display._meta = {
        "type": "hist_avg",
        "parentkey": parentkey,
        "childkey": childkey,
        "label": label,
        "factory": "make_averaged_histogram",
    }
    display_registry.append(display)

    return display


def make_section_averaged_histogram_runs_only(
    file,
    childkeys,
    parentkeys,
    display_registry,
    n_runs=1,
    n_bins=50,
    titles=None,
    labels=None,
    xaxes=None,
):
    """Create multiple Bokeh averaged-histogram plot for 2d data.
    A single slider controls the number of events to average.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkeys : list of string
        Child keys of the file to retrieve 2d data.
    parentkeys : list of string
        Parent keys of the file to retrieve 2d data.
    display_registry : list
        Storage of all the displays for later update.
    n_runs : int, optional
        Number of latest events to average the histogram on.
        Default is 1.
    n_bins : int, optional
        Number of bins of the histogram.
        Default is 20.
    titles : list of string, optional
        Titles of the histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    labels : list of string, optional
        Labels for the histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    xaxes : list of string, optional
        Labels of the x-axes of the histograms.
        Default is ``None``, meaning it is set as ``labels``.

    Returns
    -------
    display : gridplot
        Bokeh plots of the histograms.

    """

    # normalize lists
    if isinstance(parentkeys, str):
        parentkeys = {key: parentkeys for key in childkeys.keys()}
    if titles is None:
        titles = childkeys
    if labels is None:
        labels = childkeys
    if xaxes is None:
        xaxes = labels
    if isinstance(titles, str):
        titles = {key: titles for key in childkeys.keys()}
    if isinstance(labels, str):
        labels = {key: labels for key in childkeys.keys()}
    if isinstance(xaxes, str):
        xaxes = {key: xaxes for key in childkeys.keys()}

    # build displays
    displays = []
    for key in childkeys.keys():
        disp = make_averaged_histogram(
            file,
            childkey=childkeys[key],
            parentkey=parentkeys[key],
            display_registry=display_registry,
            n_runs=n_runs,
            n_bins=n_bins,
            title=titles[key],
            label=labels[key],
            xaxis=xaxes[key],
        )
        displays.append(disp)

    # layout: sliders above grid of figures
    display_gridplot = gridplot([d.figure for d in displays], ncols=2)
    return display_gridplot


def make_histogram(
    file,
    childkey,
    parentkey,
    display_registry,
    n_bins=20,
    title=None,
    label=None,
    xaxis=None,
):
    """Create the Bokeh histogram plot for 1d data.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkey : string
        Child key of the file to retrieve data.
    parentkey : string
        Parent key of the file to retrieve data.
    display_registry : list
        Storage of all the displays for later update.
    n_bins : int, optional
        Number of bins of the histogram.
        Default is 20.
    title : string, optional
        Title of the histogram.
        Default is ``None``, meaning it is set as ``childkey``.
    label : string, optional
        Label for the histogram.
        Default is ``None``, meaning it is set as ``childkey``.
    xaxis : string, optional
        Label of the x-axis of the histogram.
        Default is ``None``, meaning it is set as ``label``.

    Returns
    -------
    display : BokehPlot
        Bokeh plot of the histogram.

    """

    if title is None:
        title = childkey
    if label is None:
        label = childkey
    if xaxis is None:
        xaxis = label
    data = np.asarray(file[parentkey][childkey])
    if data.ndim != 1:
        data = np.zeros((data.shape[0]))

    hist, edges = np.histogram(data, n_bins)
    current_data = ColumnDataSource(
        data={
            label: hist,
            "edges": (edges[:-1] + edges[1:]) / 2,
        }
    )

    display = BokehPlot(
        title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        x_range=(current_data.data["edges"].min(), current_data.data["edges"].max()),
        toolbar_location="above",
    )

    fig = display.figure
    fig.tools = [t for t in fig.tools if not isinstance(t, HoverTool)]
    fig.y_range.start = 0
    fig.xaxis.axis_label = xaxis.capitalize()

    r = fig.vbar(
        source=current_data,
        x="edges",
        width=0.9 * (edges[1] - edges[0]),
        top=label,
        color=linear_cmap(label, "Inferno256", -0.1 * np.max(hist), 1.1 * np.max(hist)),
    )

    hover = HoverTool(tooltips=[("Count", "@{}".format(label))], renderers=[r])
    fig.add_tools(hover)

    display._meta = {
        "type": "hist_1d",
        "parentkey": parentkey,
        "childkey": childkey,
        "label": label,
        "factory": "make_histogram",
    }
    display_registry.append(display)

    return display


def make_histograms(
    file,
    childkeys,
    parentkeys,
    display_registry,
    n_bins=50,
    titles=None,
    labels=None,
    xaxes=None,
    suptitle=None,
):
    """Create multiple Bokeh histogram plots for 1d data.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkeys : list of string
        Child keys of the file to retrieve 2d data.
    parentkeys : list of string
        Parent keys of the file to retrieve 2d data.
    display_registry : list
        Storage of all the displays for later update.
    n_bins : int, optional
        Number of bins of the histograms.
        Default is 20.
    titles : list of string, optional
        Titles of the histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    labels : list of string, optional
        Labels for the histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    xaxes : list of string, optional
        Labels of the x-axes of the histograms.
        Default is ``None``, meaning it is set as ``labels``.
    suptitle : string, optional
        Name of the subsection of histograms.
        Default is ``None``, meaning it is set as **Subsection**.

    Returns
    -------
    display : column
        Bokeh plots of the histograms.

    """

    if not isinstance(parentkeys, list):
        parentkeys = {key: parentkeys for key in childkeys.keys()}
    if titles is None:
        titles = childkeys
    if labels is None:
        labels = childkeys
    if xaxes is None:
        xaxes = labels
    if not isinstance(titles, dict):
        titles = {key: titles for key in childkeys.keys()}
    if not isinstance(labels, dict):
        labels = {key: labels for key in childkeys.keys()}
    if not isinstance(xaxes, dict):
        xaxes = {key: xaxes for key in childkeys.keys()}

    displays = []
    for key in childkeys.keys():
        disp = make_histogram(
            file,
            childkey=childkeys[key],
            parentkey=parentkeys[key],
            display_registry=display_registry,
            n_bins=n_bins,
            title=titles[key],
            label=labels[key],
            xaxis=xaxes[key],
        )
        displays.append(disp)

    if suptitle is None:
        suptitle = "<strong>Subsection</strong>"
    else:
        suptitle = "<strong>" + suptitle + "</strong>"
    name = Div(text=suptitle)

    # layout: sliders above grid of figures
    display_gridplot = gridplot([d.figure for d in displays], ncols=2)
    display_layout = column(name, display_gridplot)
    return display_layout


def update_display_hist(disp, parentkey, childkey, file, label, n_runs, n_bins):
    """Update histogram for one display by updating its 2d data source.

    Parameters
    ----------
    disp : BokehPlot or figure
        Histogram figure to update.
    parentkey : string
        Parent key of the file to retrieve data.
    childkey : string
        Child key of the file to retrieve data.
    file : dict_like
        Data of the considered run.
    label : string
        key label for the stored values ofthe histogram
        in the data source of the figure.
    n_runs : int
        Number of latest events to average the histogram on.
    n_bins : int
        Number of bins of the histogram.

    Returns
    -------
    out : None

    """

    if hasattr(disp, "figure"):
        figure = disp.figure
    else:
        figure = disp
    arr = np.asarray(file[parentkey][childkey])

    # normalize shapes
    if arr.ndim != 2:
        arr = np.zeros((arr.shape[0], 1))

    n_runs = max(1, int(n_runs))
    n_bins = max(1, int(n_bins))

    sample = arr[-n_runs:].ravel()
    hist, edges = np.histogram(sample, bins=n_bins)
    hist //= n_runs
    centers = (edges[:-1] + edges[1:]) / 2.0

    # update the source in-place
    try:
        # adjust vbar width (if present) and x_range
        try:
            width = 0.9 * (centers[1] - centers[0]) if len(centers) > 1 else 1.0
            for r in figure.renderers:
                # vbar glyphs usually expose a 'width' attribute
                if hasattr(r.glyph, "width"):
                    r.data_source.data = {label: hist, "edges": centers}
                    r.glyph.width = width
            figure.x_range.start = centers.min() - width / 2 if len(centers) else 0
            figure.x_range.end = centers.max() + width / 2 if len(centers) else 1
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"_recompute_display_hist: failed to update source: {e}")
    figure.update()


def update_display_hist_for_1d(disp, parentkey, childkey, file, label, n_bins):
    """Update histogram for one display by updating its data source.

    Parameters
    ----------
    disp : BokehPlot or figure
        Histogram figure to update.
    parentkey : string
        Parent key of the file to retrieve data.
    childkey : string
        Child key of the file to retrieve data.
    file : dict_like
        Data of the considered run.
    label : string
        key label for the stored values ofthe histogram
        in the data source of the figure.
    n_bins : int
        Number of bins of the histogram.

    Returns
    -------
    out : None

    """

    if hasattr(disp, "figure"):
        figure = disp.figure
    else:
        figure = disp
    data = np.asarray(file[parentkey][childkey])

    # normalize shapes
    if data.ndim != 1:
        data = np.zeros(data.shape[0])

    n_bins = max(1, int(n_bins))

    hist, edges = np.histogram(data, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # update the source in-place
    try:
        # adjust vbar width (if present) and x_range
        try:
            width = 0.9 * (centers[1] - centers[0]) if len(centers) > 1 else 1.0
            for r in figure.renderers:
                # vbar glyphs usually expose a 'width' attribute
                if hasattr(r.glyph, "width"):
                    r.data_source.data = {label: hist, "edges": centers}
                    r.glyph.width = width
            figure.x_range.start = centers.min() - width / 2 if len(centers) else 0
            figure.x_range.end = centers.max() + width / 2 if len(centers) else 1
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"_recompute_display_hist: failed to update source: {e}")
    figure.update()


def make_histogram_sections(
    file,
    display_registry,
    childkeys_avg,
    parentkeys_avg,
    childkeys_1d,
    parentkeys_1d,
    widgets,
    n_runs=1,
    n_bins=50,
    titles_avg=None,
    labels_avg=None,
    xaxes_avg=None,
    suptitle_avg=None,
    titles_1d=None,
    labels_1d=None,
    xaxes_1d=None,
    suptitle_1d=None,
):
    """Create the histogram section with both averaged- and single-histograms.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    childkeys_avg : list of string
        Child keys of the file to retrieve 2d data.
    parentkeys_avg : list of string
        Parent keys of the file to retrieve 2d data.
    parentkey_1d : string
        Parent key of the file to retrieve data.
    childkey_1d : string
        Child key of the file to retrieve data.
    widgets : list
        Storage of all interactive widgets for manual update.
    n_runs : int, optional
        Number of latest events to average the histogram on.
        Default is 1.
    n_bins : int, optional
        Number of bins of the histogram.
        Default is 20.
    titles_avg : list of string, optional
        Titles of the 2d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    labels_avg : list of string, optional
        Labels for the 2d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    xaxes_avg : list of string, optional
        Labels of the x-axes of the 2d histograms.
        Default is ``None``, meaning it is set as ``labels``.
    suptitle_avg : string, optional
        Name of the subsection of 2d histograms.
        Default is ``None``, meaning it is set as **Subsection**.
    titles_1d : list of string, optional
        Titles of the 1d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    labels_1d : list of string, optional
        Labels for the 1d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    xaxes_1d : list of string, optional
        Labels of the x-axes of the 1d histograms.
        Default is ``None``, meaning it is set as ``labels``.
    suptitle_1d : string, optional
        Name of the subsection of 1d histograms.
        Default is ``None``, meaning it is set as **Subsection**.

    Returns
    -------
    display : column
        Bokeh plots of the histograms.

    """

    histogram_avg_layout = make_section_averaged_histogram_runs_only(
        file,
        childkeys_avg,
        parentkeys_avg,
        display_registry,
        n_runs=n_runs,
        n_bins=n_bins,
        titles=titles_avg,
        labels=labels_avg,
        xaxes=xaxes_avg,
    )
    histogram_1d_layout = make_histograms(
        file,
        childkeys_1d,
        parentkeys_1d,
        display_registry,
        n_bins=n_bins,
        titles=titles_1d,
        labels=labels_1d,
        xaxes=xaxes_1d,
        suptitle=suptitle_1d,
    )

    if not isinstance(parentkeys_avg, dict):
        parentkeys_avg = {key: parentkeys_avg for key in childkeys_avg.keys()}
    if not isinstance(parentkeys_1d, dict):
        parentkeys_1d = {key: parentkeys_1d for key in childkeys_1d.keys()}
    if labels_avg is None:
        labels_avg = childkeys_avg
    if not isinstance(labels_avg, list):
        labels_avg = {key: labels_avg for key in childkeys_avg.keys()}
    if labels_1d is None:
        labels_1d = childkeys_1d
    if not isinstance(labels_1d, list):
        labels_1d = {key: labels_1d for key in childkeys_1d.keys()}

    # Slider run part
    # one slider to control *number of runs* only
    parentkey_for_run_slider = parentkeys_avg[list(childkeys_avg.keys())[0]]
    childkey_for_run_slider = childkeys_avg[list(childkeys_avg.keys())[0]]
    slider_runs = Slider(
        start=1,
        end=file[parentkey_for_run_slider][childkey_for_run_slider].shape[0],
        value=max(1, int(n_runs)),
        step=1,
        title="Number of runs to average",
    )
    widgets["hist_runs"] = slider_runs
    displays_avg = {
        key: child[0]
        for key, child in zip(childkeys_avg.keys(), histogram_avg_layout.children)
    }
    displays_1d = {
        key: child[0]
        for key, child in zip(
            childkeys_1d.keys(), histogram_1d_layout.children[1].children
        )
    }

    # Slider bin part
    slider_bins = Slider(
        start=2, end=100, value=max(2, int(n_bins)), step=1, title="Number of bins"
    )
    widgets["hist_bins"] = slider_bins

    # callback: recompute all displays' histograms using current slider value
    def _on_runs_change(attr, old, new):
        current_runs = slider_runs.value
        n_bins = slider_bins.value
        for key in childkeys_avg.keys():
            update_display_hist(
                displays_avg[key],
                parentkeys_avg[key],
                childkeys_avg[key],
                file,
                labels_avg[key],
                current_runs,
                n_bins,
            )
        current_time = time.strftime("%H:%M:%S")
        logger.info(
            f"Number of averaged runs changed to {current_runs}: {current_time}"
        )

    slider_runs.on_change("value", _on_runs_change)

    # initial recompute (ensures sources are consistent)
    _on_runs_change(None, None, None)

    # Div suptitle
    if suptitle_avg is None:
        suptitle_avg = "<strong>Subsection</strong>"
    else:
        suptitle_avg = "<strong>" + suptitle_avg + "</strong>"
    name = Div(text=suptitle_avg)

    display_layout = column(name, slider_runs, histogram_avg_layout)

    # callback: recompute all displays' histograms using current slider value
    def _on_bins_change(attr, old, new):
        current_bins = slider_bins.value
        current_runs = slider_runs.value
        for key in childkeys_avg.keys():
            update_display_hist(
                displays_avg[key],
                parentkeys_avg[key],
                childkeys_avg[key],
                file,
                labels_avg[key],
                n_runs=current_runs,
                n_bins=current_bins,
            )
        for key in childkeys_1d.keys():
            update_display_hist_for_1d(
                displays_1d[key],
                parentkeys_1d[key],
                childkeys_1d[key],
                file,
                labels_1d[key],
                n_bins=current_bins,
            )
        logger.info(
            f"Number of bins changed to {current_bins}: {time.strftime('%H:%M:%S')}"
        )

    slider_bins.on_change("value", _on_bins_change)

    # initial recompute (ensures sources are consistent)
    _on_bins_change(None, None, None)

    histogram_layout = column(slider_bins, display_layout, histogram_1d_layout)
    return histogram_layout


def make_annulus(file, childkey, parentkey, display_registry, title=None):
    """Create the Bokeh pie chart plot for discrete data.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkey : string
        Child key of the file to retrieve data.
    parentkey : string
        Parent key of the file to retrieve data.
    display_registry : list
        Storage of all the displays for later update.
    n_bins : int, optional
        Number of bins of the pie chart.
        Default is 20.
    title : string, optional
        Title of the pie chart.
        Default is ``None``, meaning it is set as ``childkey``.

    Returns
    -------
    display : BokehPlot
        Bokeh plot of the pie chart.

    """

    if title is None:
        title = childkey

    display = BokehPlot(
        title=title,
        tools=("xpan", "box_zoom", "wheel_zoom", "save", "reset"),
        active_drag="xpan",
        toolbar_location=None,
    )
    xdr = Range1d(start=-2, end=2)
    ydr = Range1d(start=-2, end=2)
    display.figure = Plot(x_range=xdr, y_range=ydr)
    fig = display.figure
    fig.title.text = title

    data = np.asarray(file[parentkey][childkey])
    group, counts = np.unique(data, return_counts=True)
    angles = np.concatenate(([0], 2 * np.pi * np.cumsum(counts) / np.sum(counts)))
    source = ColumnDataSource(
        {
            "start": angles[:-1],
            "end": angles[1:],
            "colors": Inferno[len(group) + 2][1:-1],
            "counts": counts,
        }
    )
    glyph = AnnularWedge(
        x=0,
        y=0,
        inner_radius=0.9,
        outer_radius=1.8,
        start_angle="start",
        end_angle="end",
        line_color="white",
        line_width=3,
        fill_color="colors",
    )
    r = fig.add_glyph(source, glyph)

    hover = HoverTool(tooltips=[("Count", "@{}".format("counts"))], renderers=[r])
    fig.add_tools(hover)

    legend = Legend(location="center")
    for i, name in enumerate(group):
        legend.items.append(LegendItem(label=str(name), renderers=[r], index=i))
    fig.add_layout(legend, "center")

    display._meta = {
        "type": "annulus",
        "parentkey": parentkey,
        "childkey": childkey,
        "factory": "make_annulus",
    }
    display_registry.append(display)

    return display


def make_annulii(
    file, childkeys, parentkeys, display_registry, titles=None, suptitle=None
):
    """Create multiple Bokeh pie charts plot for discrete data.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkeys : list of string
        Child keys of the file to retrieve discrete data.
    parentkeys : list of string
        Parent keys of the file to retrieve discrete data.
    display_registry : list
        Storage of all the displays for later update.
    n_bins : int, optional
        Number of bins of the pie charts.
        Default is 20.
    titles : list of string, optional
        Titles of the pie charts.
        Default is ``None``, meaning it is set as ``childkeys``.
    suptitle : string, optional
        Name of the subsection of pie charts.
        Default is ``None``, meaning it is set as **Subsection**.

    Returns
    -------
    display : column
        Bokeh plots of the pie charts.

    """

    if not isinstance(parentkeys, dict):
        parentkeys = {key: parentkeys for key in childkeys.keys()}
    if titles is None:
        titles = childkeys
    if not isinstance(titles, dict):
        titles = {key: titles for key in childkeys.keys()}

    displays = []
    for key in childkeys.keys():
        displays.append(
            make_annulus(
                file, childkeys[key], parentkeys[key], display_registry, titles[key]
            )
        )

    if suptitle is None:
        suptitle = "<strong>Subsection</strong>"
    name = Div(text=suptitle)

    display_gridplot = gridplot([d.figure for d in displays], ncols=2)
    display_layout = column(name, display_gridplot)
    return display_layout


def update_annulus(disp, parentkey, childkey, current_file):
    """Update pie chart for one display by updating its discrete data source.

    Parameters
    ----------
    disp : BokehPlot or figure
        Histogram figure to update.
    parentkey : string
        Parent key of the file to retrieve data.
    childkey : string
        Child key of the file to retrieve data.
    current_file : dict_like
        Data of the considered run.

    Returns
    -------
    out : None

    """

    # read dataset robustly
    try:
        arr = np.asarray(current_file[parentkey][childkey])
    except Exception as e:
        # dataset missing or unreadable => nothing to update
        logger.warning(f"_recompute_annulus: failed read {parentkey}/{childkey}: {e}")
        return

    group, counts = np.unique(arr, return_counts=True)
    angles = np.concatenate(([0], 2 * np.pi * np.cumsum(counts) / np.sum(counts)))
    source = disp.figure.renderers[0].data_source
    source.data.start = angles[:-1]
    source.data.end = angles[1:]
    source.data.colors = Inferno[len(group) + 2][1:-1]
    source.data.counts = counts


def make_full_histogram_sections(
    file,
    display_registry,
    widgets,
    childkeys_avg,
    parentkeys_avg,
    childkeys_1d,
    parentkeys_1d,
    childkeys_pie,
    parentkeys_pie,
    n_runs=1,
    n_bins=50,
    titles_avg=None,
    labels_avg=None,
    xaxes_avg=None,
    suptitle_avg=None,
    titles_1d=None,
    labels_1d=None,
    xaxes_1d=None,
    suptitle_1d=None,
    titles_pie=None,
    suptitle_pie=None,
):
    """Create the histogram TabPanel with averaged-, single-histograms and annulii.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    childkeys_avg : list of string
        Child keys of the file to retrieve 2d data.
    parentkeys_avg : list of string
        Parent keys of the file to retrieve 2d data.
    parentkey_1d : string
        Parent key of the file to retrieve data.
    childkey_1d : string
        Child key of the file to retrieve data.
    childkeys_pie : list of string
        Child keys of the file to retrieve discrete data.
    parentkeys_pie : list of string
        Parent keys of the file to retrieve discrete data.
    n_runs : int, optional
        Number of latest events to average the histogram on.
        Default is 1.
    n_bins : int, optional
        Number of bins of the histogram.
        Default is 20.
    titles_avg : list of string, optional
        Titles of the 2d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    labels_avg : list of string, optional
        Labels for the 2d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    xaxes_avg : list of string, optional
        Labels of the x-axes of the 2d histograms.
        Default is ``None``, meaning it is set as ``labels``.
    suptitle_avg : string, optional
        Name of the subsection of 2d histograms.
        Default is ``None``, meaning it is set as **Subsection**.
    titles_1d : list of string, optional
        Titles of the 1d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    labels_1d : list of string, optional
        Labels for the 1d histograms.
        Default is ``None``, meaning it is set as ``childkeys``.
    xaxes_1d : list of string, optional
        Labels of the x-axes of the 1d histograms.
        Default is ``None``, meaning it is set as ``labels``.
    suptitle_1d : string, optional
        Name of the subsection of 1d histograms.
        Default is ``None``, meaning it is set as **Subsection**.
    titles_pie : list of string, optional
        Titles of the pie charts.
        Default is ``None``, meaning it is set as ``childkeys``.
    suptitle_pie : string, optional
        Name of the subsection of pie charts.
        Default is ``None``, meaning it is set as **Subsection**.

    Returns
    -------
    display : TabPanel
        Bokeh plots of the histogram panel.
    """

    histogram_layout = make_histogram_sections(
        file,
        display_registry,
        childkeys_avg,
        parentkeys_avg,
        childkeys_1d,
        parentkeys_1d,
        widgets=widgets,
        n_runs=n_runs,
        n_bins=n_bins,
        titles_avg=titles_avg,
        labels_avg=labels_avg,
        xaxes_avg=xaxes_avg,
        suptitle_avg=suptitle_avg,
        titles_1d=titles_1d,
        labels_1d=labels_1d,
        xaxes_1d=xaxes_1d,
        suptitle_1d=suptitle_1d,
    )
    annulii_layout = make_annulii(
        file,
        childkeys_pie,
        parentkeys_pie,
        display_registry,
        titles=titles_pie,
        suptitle=suptitle_pie,
    )
    full_histogram_layout = column(histogram_layout, annulii_layout)
    return TabPanel(child=full_histogram_layout, title="Histograms")
