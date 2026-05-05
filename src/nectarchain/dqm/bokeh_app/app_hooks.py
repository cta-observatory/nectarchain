import collections
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
from astropy.coordinates import SkyCoord

# bokeh imports
from bokeh.layouts import column, row
from bokeh.models import ColorBar, Label, Node, TabPanel
from bokeh.plotting import figure

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

# ctapipe imports
from ctapipe.visualization.bokeh import CameraDisplay
from ctapipe_io_nectarcam import constants

from nectarchain.dqm.bokeh_app.logging_config import setup_logger

base_dir = os.path.abspath(os.path.dirname(__file__))
labels_path = os.path.join(base_dir, "data", "labels.json")


NOTINDISPLAY = [
    "TRIGGER-.*",
    "PED-INTEGRATION-.*",
    "START-TIMES",
    "WF-.*",
    ".*PIXTIMELINE-.*",
]
TEST_PATTERN = "(?:% s)" % "|".join(NOTINDISPLAY)

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

logger = setup_logger()


def get_run_ids_for_camera(src, camera_code):
    """Get run ids for a given camera from database keys

    Parameters
    ----------
    src : DQMDB
        Object-oriented database defined in nectarchain.dqm.db_utils
        from ZODB and ZEO ClientStorage
    camera_code : str
        Code of the camera to filter the run ids for

    Returns
    -------
    list
        List containing the run ids for the given camera
    """

    all_database_keys = list(src.keys())
    run_ids_for_camera = []
    for key in all_database_keys:
        if f"NectarCAM{camera_code}" in key:
            run_ids_for_camera.append(key)

    logger.info(
        f"Successfully extracted run ids for camera {camera_code} from database keys"
    )

    run_ids_for_camera = sorted(run_ids_for_camera, reverse=True)
    return run_ids_for_camera


def get_available_cameras_from_db_keys(src):
    """Get available cameras from database keys

    Parameters
    ----------
    src : DQMDB
        Object-oriented database defined in nectarchain.dqm.db_utils
        from ZODB and ZEO ClientStorage

    Returns
    -------
    set
        Set containing the names of available cameras
    """

    all_database_keys = list(src.keys())
    available_cameras = set()
    for key in all_database_keys:
        if not re.match(TEST_PATTERN, key):
            camera_name = key.split("NectarCAM")[1].split("_")[0]
            available_cameras.add(camera_name)

    logger.info(
        "Successfully extracted available cameras"
        + f"from database keys: {available_cameras}"
    )

    return available_cameras


def get_rundata(src, runid):
    """Get run data to populate plots on the Bokeh displays

    Parameters
    ----------
    src : DQMDB
        Object-oriented database defined in nectarchain.dqm.db_utils
        from ZODB and ZEO ClientStorage
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'

    Returns
    -------
    dict
        Dictionary containing quantities extracted
        with nectarchain.dqm.start_dqm and stored into the database
    """

    run_data = src[runid]

    logger.info(f"Successfully extracted data for run {runid}")

    return run_data


def get_run_times(source):
    """Extract important time stamps for the provided run data

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`

    Returns
    -------
    run_start_time_dt : datetime.datetime
        Time of the start of the run in %Y-%m-%d %H:%M:%S format
    first_event_time_dt : datetime.datetime
        Time when the first event was recorded in %Y-%m-%d %H:%M:%S format
    last_event_time_dt : datetime.datetime
        Time when the last event was recorded in %Y-%m-%d %H:%M:%S format
    """

    run_start_time = int(source["START-TIMES"]["Run start time"].flatten()[0])
    run_start_time_dt = datetime.fromtimestamp(
        run_start_time, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    first_event_time = int(source["START-TIMES"]["First event"].flatten()[0])
    first_event_time_dt = datetime.fromtimestamp(
        first_event_time, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")
    last_event_time = int(source["START-TIMES"]["Last event"].flatten()[0])
    last_event_time_dt = datetime.fromtimestamp(
        last_event_time, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S")

    logger.info(
        f"Successfully extracted run times: "
        f"run start time {run_start_time_dt}, "
        f"first event time {first_event_time_dt}, "
        f"last event time {last_event_time_dt}"
    )

    return run_start_time_dt, first_event_time_dt, last_event_time_dt


def make_timelines(source, runid=None):
    """Make timeline plots for pixel quantities evolving with time

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.
        By default None

    Returns
    -------
    dict
        Nested dictionary containing line plots for the timelines
    """

    with open(labels_path, "r", encoding="utf-8") as file:
        y_axis_labels = json.load(file)["y_axis_labels_timelines"]

    timelines = collections.defaultdict(dict)
    for parentkey in source.keys():
        # Prepare timeline line plots only for pixel quantities evolving with time
        if re.match("(?:.*PIXTIMELINE-.*)", parentkey):
            for childkey in source[parentkey].keys():
                logger.info(
                    f"Run id {runid}, preparing plot for {parentkey}, {childkey}"
                )
                timelines[parentkey][childkey] = figure(title=childkey)
                evts = np.arange(len(source[parentkey][childkey]))
                timelines[parentkey][childkey] = figure(
                    title=childkey,
                    x_range=(0, np.max(evts) + 50),
                    y_range=(0, 1),
                    # A fraction is plotted:
                    # y-range values are between 0 and 1 because
                )
                timelines[parentkey][childkey].line(
                    x=evts,
                    y=source[parentkey][childkey],
                    line_width=3,
                )
    for parentkey in timelines.keys():
        for childkey in timelines[parentkey].keys():
            timelines[parentkey][childkey].xaxis.axis_label = "Event number"
            try:
                timelines[parentkey][childkey].yaxis.axis_label = y_axis_labels[
                    parentkey
                ]
            except ValueError:
                timelines[parentkey][childkey].yaxis.axis_label = ""
            except KeyError:
                timelines[parentkey][childkey].yaxis.axis_label = ""

            timelines[parentkey][childkey].xaxis.axis_label_text_font_size = "12pt"
            timelines[parentkey][childkey].yaxis.axis_label_text_font_size = "12pt"
            timelines[parentkey][childkey].xaxis.major_label_text_font_size = "10pt"
            timelines[parentkey][childkey].yaxis.major_label_text_font_size = "10pt"
            timelines[parentkey][childkey].xaxis.axis_label_text_font_style = "normal"
            timelines[parentkey][childkey].yaxis.axis_label_text_font_style = "normal"

    logger.info(f"Successfully created timeline plots for run {runid}")

    return dict(timelines)


def update_timelines(data, timelines, runid=None):
    """Reset each timeline previously created by `make_timelines`

    Parameters
    ----------
    data : dict
        Dictionary returned by `get_rundata`
    timelines : dict
        Nested dictionary containing line plots created by `make_timelines`
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.
        By default None

    Returns
    -------
    bokeh.models.TabPanel
        Updated TabPanel containing the bokeh layout for the timeline plots
    """

    # Make new timeline plots
    timelines = make_timelines(data, runid)

    list_timelines = [
        timelines[parentkey][childkey]
        for parentkey in timelines.keys()
        for childkey in timelines[parentkey].keys()
    ]

    layout_timelines = column(
        list_timelines,
        sizing_mode="scale_width",
    )

    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    return tab_timelines


def make_camera_displays(source, runid):
    """Make camera display plots using `make_camera_display`,
       `make_pixel_val_vs_id` and `make_pixel_vals_histo`

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    runid : str
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.

    Returns
    -------
    dict
        Nested dictionary containing display plots created by `make_camera_display`
    """

    displays = collections.defaultdict(dict)
    for parentkey in source.keys():
        if not re.match(TEST_PATTERN, parentkey):
            for childkey in source[parentkey].keys():
                logger.info(
                    f"Run id {runid}, preparing plot for {parentkey}, {childkey}"
                )
                camera_display = make_camera_display(
                    source, parent_key=parentkey, child_key=childkey
                )
                displays_to_show = [camera_display]

                if "BADPIX" not in parentkey:
                    camera_pixel_val_vs_id = make_pixel_val_vs_id(
                        source, parent_key=parentkey, child_key=childkey
                    )
                    displays_to_show.append(camera_pixel_val_vs_id)
                    camera_pixel_vals_histo = make_pixel_vals_histo(
                        source, parent_key=parentkey, child_key=childkey
                    )
                    displays_to_show.append(camera_pixel_vals_histo)

                displays[parentkey][childkey] = displays_to_show

    logger.info(f"Successfully created camera display plots for run {runid}")

    return dict(displays)


def update_camera_displays(data, displays, runid=None):
    """Reset each display previously created by `make_camera_displays`

    Parameters
    ----------
    data : dict
        Dictionary returned by `get_rundata`
    displays : dict
        Nested dictionary containing display plots
        created by `make_camera_displays`
    runid : str, optional
        Identifier for dictionary extracted from the database,
        containing the NectarCAM run number. Example: 'NectarCAM_Run6310'.
        By default None

    Returns
    -------
    bokeh.models.TabPanel
        Updated TabPanel containing the bokeh layout for the display plots
    """

    # Make new camera display plots
    displays = make_camera_displays(data, runid)

    camera_displays = [
        row(
            displays[parentkey][childkey][0].figure,
            displays[parentkey][childkey][1],
            displays[parentkey][childkey][2],
        )
        if len(displays[parentkey][childkey]) == 3
        else displays[parentkey][childkey][0].figure
        for parentkey in displays.keys()
        for childkey in displays[parentkey].keys()
    ]

    layout_camera_displays = column(
        camera_displays,
        sizing_mode="scale_width",
    )

    tab_camera_displays = TabPanel(
        child=layout_camera_displays, title="Camera displays"
    )

    return tab_camera_displays


def make_pixel_vals_histo(source, parent_key, child_key):
    """Make histograms of pixel values
       to fill the nested dict
       created by `make_camera_displays`
       along with the camera displays
       and the 1D plot of camera pixel values vs pixel id

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    parent_key : str
        Parent key to extract quantity from the dict
    child_key : str
        Child key to extract quantity from the dict

    Returns
    -------
    bokeh.plotting.figure
        figure containing the histogram of pixel values
    """

    image = np.nan_to_num(source[parent_key][child_key], nan=0.0)

    if "BADPIX" in parent_key:
        image = set_bad_pixels_cap_value(image)
        data_for_hist = image
    else:
        mask_high_gain, mask_low_gain = get_bad_pixels_position(
            source=source, image_shape=image.shape
        )
        data_for_hist = image[
            ~mask_low_gain if "LOW-GAIN" in parent_key else ~mask_high_gain
        ]

    # Use adaptive binning on full data to include outliers
    hist, bins = np.histogram(data_for_hist, bins="fd")
    min_val, max_val = 0.0, np.max(hist) + 10

    with open(labels_path, "r", encoding="utf-8") as file:
        colorbar_labels = json.load(file)["colorbar_labels_camera_display"]

    try:
        x_ax_label = colorbar_labels[parent_key]
    except ValueError:
        x_ax_label = ""
    except KeyError:
        x_ax_label = ""

    histo_values = figure(
        x_range=(np.min(bins) * 0.99, np.max(bins) * 1.01), y_range=(min_val, max_val)
    )

    # Calculate bar width based on actual bin widths to avoid overlapping or gaps
    bin_widths = bins[1:] - bins[:-1]
    bar_width = np.mean(bin_widths) * 0.95  # Use 95% of average bin width

    histo_values.vbar(
        x=((bins[1:] - bins[:-1]) / 2.0 + bins[:-1]),
        top=hist,
        width=bar_width,
        color="green",
        alpha=0.6,
    )

    frame_left = Node(target="frame", symbol="left", offset=5)
    frame_top = Node(target="frame", symbol="top", offset=5)

    stats = Label(
        x=frame_left,
        y=frame_top,
        anchor="top_left",
        text=f"Mean    = {np.mean(data_for_hist):.2f} \n"
        + f"Median = {np.median(data_for_hist):.2f} \n"
        + f"Std       = {np.std(data_for_hist):.2f}",
        padding=10,
        border_radius=5,
        border_line_color="green",
        border_line_width=2,
        background_fill_color="white",
    )

    histo_values.add_layout(stats)

    histo_values.xaxis.axis_label = x_ax_label
    histo_values.yaxis.axis_label = "Pixel count"

    histo_values.xaxis.axis_label_text_font_size = "12pt"
    histo_values.yaxis.axis_label_text_font_size = "12pt"
    histo_values.xaxis.major_label_text_font_size = "10pt"
    histo_values.yaxis.major_label_text_font_size = "10pt"
    histo_values.xaxis.axis_label_text_font_style = "normal"
    histo_values.yaxis.axis_label_text_font_style = "normal"

    return histo_values


def make_pixel_val_vs_id(source, parent_key, child_key):
    """Make 1D plot of camera pixel values vs pixel id
       to fill the nested dict
       created by `make_camera_displays`
       along with the camera displays and the histograms of pixel values

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    parent_key : str
        Parent key to extract quantity from the dict
    child_key : str
        Child key to extract quantity from the dict

    Returns
    -------
    bokeh.plotting.figure
        figure containing the 1D plot of camera pixel values vs pixel id
    """

    image = np.nan_to_num(source[parent_key][child_key], nan=0.0)
    if "BADPIX" in parent_key:
        image = set_bad_pixels_cap_value(image)
        min_val, max_val = 0.0, 1.0
    else:
        mask_high_gain, mask_low_gain = get_bad_pixels_position(
            source=source, image_shape=image.shape
        )
        min_val = (
            np.min(
                image[~mask_low_gain if "LOW-GAIN" in parent_key else ~mask_high_gain]
            )
            * 0.99
        )
        max_val = (
            np.max(
                image[~mask_low_gain if "LOW-GAIN" in parent_key else ~mask_high_gain]
            )
            * 1.01
        )

    with open(labels_path, "r", encoding="utf-8") as file:
        colorbar_labels = json.load(file)["colorbar_labels_camera_display"]

    try:
        y_ax_label = colorbar_labels[parent_key]
    except ValueError:
        y_ax_label = ""
    except KeyError:
        y_ax_label = ""

    scatter_value_vs_id = figure(
        background_fill_color="#ffffff",
        y_range=(min_val, max_val),
    )

    scatter_value_vs_id.scatter(
        x=np.arange(len(image)),
        y=image,
        color="blue",
        size=5,
        alpha=0.6,
    )

    scatter_value_vs_id.xaxis.axis_label = "Pixel id"
    scatter_value_vs_id.yaxis.axis_label = y_ax_label

    scatter_value_vs_id.xaxis.axis_label_text_font_size = "12pt"
    scatter_value_vs_id.yaxis.axis_label_text_font_size = "12pt"
    scatter_value_vs_id.xaxis.major_label_text_font_size = "10pt"
    scatter_value_vs_id.yaxis.major_label_text_font_size = "10pt"
    scatter_value_vs_id.xaxis.axis_label_text_font_style = "normal"
    scatter_value_vs_id.yaxis.axis_label_text_font_style = "normal"

    return scatter_value_vs_id


# TODO: some more explanation about the parent and child keys
# may help the user, if needed
def make_camera_display(source, parent_key, child_key):
    """Make camera display plot to fill the nested dict
       created by `make_camera_displays`
       along with the 1D plot of camera pixel values vs pixel id
       and the histograms of pixel values

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    parent_key : str
        Parent key to extract quantity from the dict
    child_key : str
        Child key to extract quantity from the dict

    Returns
    -------
    ctapipe.visualization.bokeh.CameraDisplay
        CameraDisplay filled with values for the selected quantity,
        and displayed with the geometry from ctapipe.instrument.CameraGeometry
    """

    # TODO: may want to check here to implement
    # the "on_pixel_clicked" function for pixels
    # ctapipe.readthedocs.io/en/stable/api/ctapipe.visualization.CameraDisplay.html

    image = np.nan_to_num(source[parent_key][child_key], nan=0.0)

    if "BADPIX" in parent_key:
        image = set_bad_pixels_cap_value(image)
    else:
        mask_high_gain, mask_low_gain = get_bad_pixels_position(
            source=source, image_shape=image.shape
        )
        # plotting by default range with 99.5% of all events, so that
        # outliers do not prevent us from seing the bulk of the data
        min_colorbar = np.nanquantile(
            image[~mask_low_gain if "LOW-GAIN" in parent_key else ~mask_high_gain],
            0.005,
        )
        max_colorbar = np.nanquantile(
            image[~mask_low_gain if "LOW-GAIN" in parent_key else ~mask_high_gain],
            0.995,
        )
        if max_colorbar == min_colorbar:
            # avoid problems with bokeh display
            max_colorbar *= 1.05
            min_colorbar *= 0.95
        image[mask_low_gain if "LOW-GAIN" in parent_key else mask_high_gain] = 0.0

    display = CameraDisplay(geometry=geom)
    try:
        display.image = image
    except ValueError as e:
        image = np.zeros(shape=display.image.shape)
        display.image = image
        logger.error(
            f"Exception '{e}', filled camera plot"
            + f" {parent_key}, {child_key} with zeros"
        )
    except KeyError as e:
        image = np.zeros(shape=constants.N_PIXELS)
        display.image = image
        logger.error(
            f"Exception '{e}', filled camera plot"
            + f" {parent_key}, {child_key} with zeros"
        )

    fig = display.figure
    pix_x, pix_y = geom.pix_x, geom.pix_y
    cam_coords = SkyCoord(x=pix_x, y=pix_y, frame=geom.frame)
    # add axis labels
    fig.xaxis.axis_label = f"x / {cam_coords.x.unit}"
    fig.yaxis.axis_label = f"y / {cam_coords.y.unit}"
    fig.xaxis.axis_label_text_font_size = "12pt"
    fig.xaxis.axis_label_text_font_style = "normal"
    fig.yaxis.axis_label_text_font_size = "12pt"
    fig.yaxis.axis_label_text_font_style = "normal"

    if "BADPIX" not in parent_key:
        display._color_mapper.low = min_colorbar
        display._color_mapper.high = max_colorbar
        # for pixels that are outside the colorbar range, like bad pixels,
        # set displayed color to white
        if not all(~mask_high_gain):
            display._color_mapper.low_color = "white"

    # add colorbar
    color_bar = ColorBar(
        color_mapper=display._color_mapper,
        padding=5,
    )
    fig.add_layout(color_bar, "right")
    color_bar.title_text_font_size = "14pt"
    color_bar.title_text_font_style = "normal"

    with open(labels_path, "r", encoding="utf-8") as file:
        colorbar_labels = json.load(file)["colorbar_labels_camera_display"]

    try:
        color_bar.title = colorbar_labels[parent_key]
    except ValueError:
        color_bar.title = ""
    except KeyError:
        color_bar.title = ""

    display.figure.title = child_key

    return display


def set_bad_pixels_cap_value(image):
    """Set cap value for the bad pixels to 1,
       to follow the colorbar definition

    Parameters
    ----------
    image : numpy.ndarray
        2D array extracted from the database,
        containing bad pixel values for the whole camera

    Returns
    -------
    numpy.ndarray
        The 2D image with the bad pixel values capped to 1
    """

    image[image > 1] = 1.0

    return image


def get_bad_pixels_position(source, image_shape):
    """Get the positions of the bad pixels
       in the camera as boolean masks

    Parameters
    ----------
    source : dict
        Dictionary returned by `get_rundata`
    image_shape : tuple
        Shape of the display image
        for the quantity called in `make_camera_display`

    Returns
    -------
    numpy.ndarray
        Boolean mask containing the positions of
        bad pixels in the camera for the High gain channel
    numpy.ndarray
        Boolean mask containing the positions of
        bad pixels in the camera for the Low gain channel
    """

    try:
        if "CAMERA-BADPIX-PED-PHY-OVEREVENTS-HIGH-GAIN" in source.keys():
            image_badpix_high_gain = source[
                "CAMERA-BADPIX-PED-PHY-OVEREVENTS-HIGH-GAIN"
            ]["CAMERA-BadPix-PED-PHY-OverEVENTS-HIGH-GAIN"]
            image_badpix_low_gain = source[
                "CAMERA-BADPIX-PED-PHY-OVEREVENTS-HIGH-GAIN"
            ]["CAMERA-BadPix-PED-PHY-OverEVENTS-HIGH-GAIN"]
        elif "CAMERA-BADPIX-PHY-OVEREVENTS-HIGH-GAIN" in source.keys():
            image_badpix_high_gain = source["CAMERA-BADPIX-PHY-OVEREVENTS-HIGH-GAIN"][
                "CAMERA-BadPix-PHY-OverEVENTS-HIGH-GAIN"
            ]
            image_badpix_low_gain = source["CAMERA-BADPIX-PHY-OVEREVENTS-HIGH-GAIN"][
                "CAMERA-BadPix-PHY-OverEVENTS-HIGH-GAIN"
            ]

        mask_bad_pixels_high_gain = image_badpix_high_gain >= 1.0
        # FIXME: bad pixels for High and Low gain may be the same
        # (although it may depend on the definition of bad pixel),
        # the mask defined below may be obsolete
        mask_bad_pixels_low_gain = image_badpix_low_gain >= 1.0
    except KeyError as e:
        mask_bad_pixels_high_gain = np.zeros(shape=constants.N_PIXELS, dtype=bool)
        mask_bad_pixels_low_gain = mask_bad_pixels_high_gain
        logger.error(f"Exception '{e}', bad pixels flag not found in the database")

    if image_shape != mask_bad_pixels_high_gain.shape:
        mask_bad_pixels_high_gain = np.zeros(shape=image_shape, dtype=bool)
        mask_bad_pixels_low_gain = mask_bad_pixels_high_gain
        logger.error(
            "Some modules not available for the run,"
            + " need to reset the shape of the bad pixels masks"
        )

    return mask_bad_pixels_high_gain, mask_bad_pixels_low_gain
