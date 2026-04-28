from app_hooks import (
    get_available_cameras_from_db_keys,
    get_run_ids_for_camera,
    get_run_times,
    get_rundata,
    make_camera_displays,
    make_timelines,
    update_camera_displays,
    update_timelines,
)

# bokeh imports
from bokeh.layouts import column, row
from bokeh.models import Div, Select, TabPanel, Tabs
from bokeh.plotting import curdoc

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

from nectarchain.dqm.bokeh_app.logging_config import setup_logger
from nectarchain.dqm.db_utils import DQMDB

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

logger = setup_logger()


def get_layout_per_camera(source, runids, camera_code):
    def update(attr, old, new):
        """Callback that refreshes the dashboard whenever the selected run changes.

        Reset plots with helper functions for update
        and new data from `get_rundata`, and update the page_layout

        Parameters
        ----------
        attr : str
            Name of the Bokeh property that triggered the callback.
            For this widget it is always ``"value"``.
        old : str
            The previous value before the user made a new selection.
            Useful for diff-checking or logging.
        new : str
            The newly selected value (run identifier).
            This is the value needed to load fresh data.

        Returns
        -------
        None
            The function updates the global `page_layout` in place by replacing
            its second child with a new `Tabs` container built from the refreshed
            data sources.

        Notes
        -----
        * `new` and `run_select.value` are equivalent inside the callback,
        so you can use either.  Using `new` avoids an extra attribute lookup.
        * The callback must accept exactly three positional arguments because
        Bokeh always supplies `attr`, `old` and `new` when invoking
        `on_change` handlers.
        """

        runid = new
        logger.info(f"Requested to display information for run: {runid}")
        source = get_rundata(db, runid)

        tab_camera_displays = update_camera_displays(source, displays, runid)
        tab_timelines = update_timelines(source, timelines, runid)
        run_start_time_dt, first_event_time_dt, last_event_time_dt = get_run_times(
            source
        )

        run_times_string = Div(
            text=f"""
            <div style="
                background-color: #f0f8ff;
                border-radius: 10px;
                padding: 10px;
                width: fit-content;
                font-size: 14px;
            ">
                <p>Run start time: {run_start_time_dt}</p>
                <p>First event recorded at: {first_event_time_dt}</p>
                <p>Last event recorded at: {last_event_time_dt}</p>
            </div>
            """
        )

        # Combine panels into tabs
        tabs = Tabs(
            tabs=[tab_camera_displays, tab_timelines], sizing_mode="scale_width"
        )

        page_layout.children[1] = tabs
        page_layout.children[0].children[1] = run_times_string

        logger.info("Updated layouts and TabPanel objects for tabs.")

    # First, get the run id with the most populated result dictionary
    # On the full DB, this takes an awful lot of time, and saturates the RAM on the host
    # VM (gets OoM killed)
    # run_dict_lengths = [len(db[r]) for r in runids]
    # runid = runids[np.argmax(run_dict_lengths)]
    # runid = "NectarCAMQM_Run6310"
    runid = runids[0]
    logger.info(f"We will start with run {runid}, in camera {camera_code}")

    logger.info("Defining Select")
    # runid_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
    # run_select = Select(value=runid, title="NectarCAM run number", options=runids)
    run_select = Select(
        value=runid,
        title=f"NectarCAM {camera_code} run number",
        options=runids,
        css_classes=["select"],
    )

    logger.info(f"Getting data for run {run_select.value}")
    source = get_rundata(db, run_select.value)
    displays = make_camera_displays(source, runid)
    timelines = make_timelines(source, runid)

    run_start_time_dt, first_event_time_dt, last_event_time_dt = get_run_times(source)
    run_times_string = Div(
        text=f"""
        <div style="
            background-color: #f0f8ff;
            border-radius: 10px;
            padding: 10px;
            width: fit-content;
            font-size: 14px;
        ">
            <p>Run start time: {run_start_time_dt}</p>
            <p>First event recorded at: {first_event_time_dt}</p>
            <p>Last event recorded at: {last_event_time_dt}</p>
        </div>
        """
    )

    controls = row(run_select, run_times_string)

    # # TEST:
    # attr = 'value'
    # old = runid
    # new = runids[1]
    # update_camera_displays(attr, old, new)
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

    list_timelines = [
        timelines[parentkey][childkey]
        for parentkey in timelines.keys()
        for childkey in timelines[parentkey].keys()
    ]

    layout_camera_displays = column(
        camera_displays,
        sizing_mode="scale_width",
    )

    layout_timelines = column(
        list_timelines,
        sizing_mode="scale_width",
    )

    # Create different tabs
    tab_camera_displays = TabPanel(
        child=layout_camera_displays, title="Camera displays"
    )
    tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

    # Combine panels into tabs
    tabs = Tabs(
        tabs=[tab_camera_displays, tab_timelines],
    )

    # TODO: may want to add a list to the logging of all created tabs,
    # to keep track of what is being displayed
    logger.info(
        f"Created layouts and TabPanel objects for tabs, for camera {camera_code}"
    )

    page_layout = column([controls, tabs], sizing_mode="scale_width")

    run_select.on_change("value", update)

    return page_layout, run_select


logger.info("Opening connection to ZODB")
db = DQMDB(read_only=True).root

available_cameras_in_db_keys = get_available_cameras_from_db_keys(db)
available_cameras_in_db = [
    f"NectarCAM {cam}" if cam != "QM" else "NectarCAM Qualification Model"
    for cam in available_cameras_in_db_keys
]
runs_for_available_cameras = {
    cam: get_run_ids_for_camera(db, cam) for cam in available_cameras_in_db_keys
}

page_layouts_per_camera = {}
run_selects_per_camera = {}
tab_panels_for_layout = []

for cam, runs in runs_for_available_cameras.items():
    logger.info(f"Camera {cam} has {len(runs)} runs in the database")
    page_layout, run_select = get_layout_per_camera(db, runs, cam)

    page_layouts_per_camera[cam] = page_layout
    run_selects_per_camera[cam] = run_select

    tab_panels_for_layout.append(TabPanel(child=page_layout, title=f"NectarCAM {cam}"))

tabs_for_layout = Tabs(tabs=tab_panels_for_layout, sizing_mode="scale_width")


# Add to the Bokeh document
curdoc().add_root(tabs_for_layout)
curdoc().title = "NectarCAM Data Quality Monitoring web app"
