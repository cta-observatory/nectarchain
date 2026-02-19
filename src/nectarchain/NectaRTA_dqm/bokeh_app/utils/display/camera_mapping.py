# Author: Julian Hamo
# Contact mail: julian.hamo@ijclab.in2p3.fr

"""
This module stores the Bokeh webpage camera mapping maker for the RTA of NectarCAM.
"""

# imports
import logging
import time
import numpy as np

# Bokeh imports
from bokeh.models import Switch, Ellipse, HoverTool, TabPanel
from bokeh.layouts import column, gridplot

# Bokeh RTA imports
from ..utils_helpers import get_hillas_parameters

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization.bokeh import CameraDisplay

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())


__all__ = ["make_tab_camera_displays", "update_camera_display"]

logger = logging.getLogger(__name__)


def _make_camera_display_params(show_hillas=False, label="Show Hillas ellipse:"):
    """Create Switch to display Hillas ellipse.

    Parameters
    ----------
    show_hillas : bool, optional
        State of the Switch, default is False.
    label : string, optional
        Label title of the Switch.
        Default is "Show Hillas ellipse:".

    Returns
    -------
    out : Switch
        Bokeh Switch to check the display of the Hillas ellipse.

    """

    return Switch(active=show_hillas, label=label)


def update_hillas_ellipse(
    ellipse, file, parameterkeys, parameter_parentkeys, run_index=-1
):
    """Update the Hillas ellipse from new run.

    Parameters
    ----------
    ellipse :
        Bokeh ellipse representing the camera display.
    file : dict_like
        Data of the considered run.
    parameterkeys : dict
        Parent keys of the file to retrieve data.
    parameter_parentkeys : string
        Parent key for the parameters in the dictionary.
    index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.

    Returns
    -------
    out : None

    """

    ellipse.x, ellipse.y, ellipse.width, ellipse.height, ellipse.angle = (
        get_hillas_parameters(
            file,
            parameterkeys=parameterkeys,
            parameter_parentkeys=parameter_parentkeys,
            run_index=run_index,
        )
    )


def _make_camera_display(
    file,
    childkey,
    image_parentkey,
    parameter_parentkeys,
    parameterkeys,
    display_registry,
    n_pixels=1855,
    run_index=-1,
    title=None,
    show_hillas=False,
    label_colorbar="",
):
    """Create the hexagonal camera display
    based on the data from the input file.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkey : string
        Child key of the file to retrieve image data.
    image_parentkey : string
        Parent key of the file to retrieve image data.
    parameter_parentkeys : string
        Parent key of the file to retrieve parameter data.
    parameterkeys : dict
        Parent keys of the file to retrieve parameter data.
    display_registry : list
        Storage of all the displays for later update.
    n_pixels : int, optional
        Number of pixel on the camera.
        Default is 1855.
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.
    title : string, optional
        Title of the camera display.
        Default is ``None``, meaning it will be ``childkey``.
    show_hillas : bool, optional
        If ``True``, display the Hillas ellipse.
        Default is ``False``.
    label_colorbar : str, optional
        Label to display for the colorbar, default is empty.

    Returns
    -------
    display : CameraDisplay
        Display of the camera mapping.

    """

    if title is None:
        title = childkey
    image = file[image_parentkey][childkey]
    image = np.nan_to_num(image[run_index], nan=0.0)
    display = CameraDisplay(geometry=geom)
    try:
        # try to map the image to the display
        display.image = image
    except ValueError:
        logger.warning("Failed to make camera display: ValueError")
        # ValueError fail: map the image to 0s
        image = np.zeros(shape=display.image.shape)
        display.image = image
    except KeyError:
        logger.warning("Failed to make camera display: KeyError")
        # KeyError fail: assumes the map is n_pixel long
        image = np.zeros(shape=n_pixels)
        display.image = image

    # Hillas ellipse
    x, y, width, height, angle = get_hillas_parameters(
        file,
        parameterkeys=parameterkeys,
        parameter_parentkeys=parameter_parentkeys,
        run_index=run_index,
    )
    ellipse = Ellipse(
        x=x,
        y=y,
        width=width,
        height=height,
        angle=angle,
        fill_color=None,
        line_color="#40E0D0",
        line_width=2,
        line_alpha=1,
    )
    glyph = display.figure.add_glyph(ellipse)
    hovertool = [t for t in display.figure.tools if isinstance(t, HoverTool)][0]
    hovertool.renderers = [display.figure.renderers[0]]

    # Hillas ellipse always computed but not necessarily shown
    if not show_hillas:
        glyph.visible = False
    display._annotations.append(glyph)
    display.update()
    display.add_colorbar()
    display._color_bar.title = label_colorbar
    display.figure.title = title

    # To deal with using dict and not list?
    display._meta = {
        "type": "camera",
        "image_parentkey": image_parentkey,
        "parameter_parentkeys": parameter_parentkeys,
        "childkey": childkey,
        "parameterkeys": parameterkeys,
        "factory": "_make_camera_display",
    }
    display_registry.append(display)

    return display


def make_tab_camera_displays(
    file,
    childkeys,
    image_parentkeys,
    parameter_parentkeys,
    parameterkeys,
    display_registry,
    widgets,
    n_pixels=1855,
    run_index=None,
    titles=None,
    show_hillas=False,
    label_hillas="Show Hillas ellipse:",
    labels_colorbar=None,
):
    """Create the tab of the hexagonal camera displays
    based on the data from the input file.

    Parameters
    ----------
    file : dict_like
        Data of the considered run.
    childkeys : list
        Child keys of the file to retrieve image data.
    image_parentkeys : list or string
        Parent keys of the file to retrieve image data.
    parameter_parentkeys : slist or tring
        Parent keys of the file to retrieve parameter data.
    parameterkeys : list of dict
        Parent keys of the file to retrieve parameter data.
    display_registry : list
        Storage of all the displays for later update.
    widgets : list
        Storage of all interactive widgets for manual update.
    n_pixels : int, optional
        Number of pixel on the camera.
        Default is 1855.
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.
    titles : list of string, optional
        Titles of the camera displays.
        Default is ``None``, meaning it will be ``childkeys``.
    show_hillas : bool, optional
        If ``True``, display the Hillas ellipse.
        Default is ``False``.
    label_hillas : str
        Label to display over the Hillas widget.
    label_colorbar : str, optional
        Label to display for the colorbar, default is empty.

    Returns
    -------
    out : TabPanel
        Tab of the camera mappings.

    """

    displays = []
    if titles is None:
        titles = childkeys
    if labels_colorbar is None:
        labels_colorbar = {key: "" for key in childkeys.keys()}
    if isinstance(image_parentkeys, str):
        image_parentkeys = {key: image_parentkeys for key in childkeys.keys()}
    if isinstance(parameter_parentkeys, str):
        parameter_parentkeys = {key: parameter_parentkeys for key in childkeys.keys()}
    if isinstance(parameterkeys, dict):
        parameterkeys = {key: parameterkeys for key in childkeys.keys()}

    for key in childkeys.keys():
        args = {
            "file": file,
            "childkey": childkeys[key],
            "image_parentkey": image_parentkeys[key],
            "parameter_parentkeys": parameter_parentkeys[key],
            "parameterkeys": parameterkeys[key],
            "display_registry": display_registry,
            "n_pixels": n_pixels,
            "run_index": run_index,
            "title": titles[key],
            "show_hillas": show_hillas,
            "label_colorbar": labels_colorbar[key],
        }
        displays.append(_make_camera_display(**args).figure)

    hillas_switch = _make_camera_display_params(
        show_hillas=show_hillas, label=label_hillas
    )
    widgets["hillas_switch"] = hillas_switch

    def callback(attr, old, new):
        for display in displays:
            for r in display.renderers:
                if isinstance(r.glyph, Ellipse):
                    r.visible = not r.visible
                    hillas_flag = r.visible
            display.update()
        if hillas_flag:
            logger.info(f"Hillas ellipse displayed: {time.strftime('%H:%M:%S')}")
        else:
            logger.info(f"Hillas ellipse hidden: {time.strftime('%H:%M:%S')}")

    hillas_switch.on_change("active", callback)
    display_gridplot = gridplot(displays, ncols=2)

    display_layout = column(hillas_switch, display_gridplot)
    tab_camera_displays = TabPanel(child=display_layout, title="Camera displays")
    return tab_camera_displays


def update_camera_display(
    disp,
    childkey,
    image_parentkey,
    parameter_parentkeys,
    parameterkeys,
    current_file,
    run_index=-1,
):
    """Update camera display .image if available.

    Parameters
    ----------
    disp: CameraDisplay
        Display of the camera mapping.
    childkey : string
        Child key of the file to retrieve image data.
    image_parentkey : string
        Parent key of the file to retrieve image data.
    parameter_parentkeys : string
        Parent key of the file to retrieve parameter data.
    parameterkeys : dict
        Parent keys of the file to retrieve parameter data.
    current_file : dict_like
        Data of the considered run.
    run_index : int, optional
        A file is constituted of multiple events,
        select an event in the stored file.
        Default is -1, resulting in the latest event of the run.

    Returns
    -------
    out : None

    """

    try:
        update_hillas_ellipse(
            disp.figure.renderers[1].glyph,
            current_file,
            parameterkeys=parameterkeys,
            parameter_parentkeys=parameter_parentkeys,
            run_index=run_index,
        )
        imgds = current_file[image_parentkey][childkey]
        # try to index safely
        try:
            img = np.nan_to_num(np.asarray(imgds[run_index]), nan=0.0)
        except Exception:
            # fallback to last event
            img = np.nan_to_num(np.asarray(imgds[-1]), nan=0.0)
        # if disp has attribute image, set it
        try:
            disp.image = img
            # Some CameraDisplay objects require calling add_colorbar or refresh; ignore
            return
        except Exception as e:
            logger.warning(f"update_camera_display: failed {e}")
    except Exception as e:
        logger.warning(f"update_camera_display: failed {e}")
