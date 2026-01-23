from app_hooks import (
    create_all_camera_display_layouts,
    get_run_times,
    get_rundata,
    make_camera_displays,
    make_timelines,
    update_camera_displays,
    update_timelines,
)

# bokeh imports
from bokeh.layouts import column, gridplot, row
from bokeh.models import Div, Select, TabPanel, Tabs
from bokeh.plotting import curdoc

# ctapipe imports
from ctapipe.coordinates import EngineeringCameraFrame
from ctapipe.instrument import CameraGeometry

from nectarchain.dqm.db_utils import DQMDB

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

# FIXME: followed some instructions to possibly inject some CSS into the Bokeh app,
# using the css_div... Does not seem to work, though...

# Read the CSS file
with open(
    "/opt/cta/nectarchain/src/nectarchain/dqm/bokeh_app/static/styles.css", "r"
) as file:
    css_content = file.read()

# Inject the CSS into the document
css_div = Div(text=f"<style>{css_content}</style>", width=0, height=0)


def update(attr, old, new):
    """Callback that refreshes the dashboard whenever the selected run changes.

    Reset plots with helper functions for update
    and new data from `get_rundata`, and update the page_layout

    Parameters
    ----------
    attr : str
        Name of the Bokeh property that triggered the callback.
        For this widget it is always ``"value"``.
    old : int
        The previous value of `run_select.value` before the user made a new
        selection. Useful for diff-checking or logging.
    new : int
        The newly selected value (the run identifier).
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
    source = get_rundata(db, runid)

    tab_camera_displays = update_camera_displays(source, displays, runid)
    tab_timelines = update_timelines(source, timelines, runid)

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

    # Combine panels into tabs
    tabs = Tabs(tabs=[tab_camera_displays, tab_timelines], sizing_mode="scale_width")

    page_layout.children[1] = tabs
    page_layout.children[0].children[1] = run_times_string


def update_timelines(attr, old, new):
    runid = run_select.value
    new_rundata = get_rundata(db, runid)

    # Reset each timeline
    for k in timelines.keys():
        for kk in timelines[k].keys():
            timelines[k][kk].line(x=0, y=0)

    for parentkey in db[runid].keys():
        if re.match("(?:.*PIXTIMELINE-.*)", parentkey):
            for childkey in db[runid][parentkey].keys():
                print(f"Run id {runid} Preparing plot for {parentkey}, {childkey}")
                timelines[parentkey][childkey] = figure(title=f"{childkey}_{runid}")
                evts = np.arange(len(new_rundata[parentkey][childkey]))
                timelines[parentkey][childkey].line(
                    x=evts,
                    y=new_rundata[parentkey][childkey],
                    line_width=3,
                )


print("Opening connection to ZODB")
db = DQMDB(read_only=True).root
print("Getting list of run numbers")
runids = sorted(list(db.keys()), reverse=True)

# First, get the run id with the most populated result dictionary
# On the full DB, this takes an awful lot of time, and saturates the RAM on the host
# VM (gets OoM killed)
# run_dict_lengths = [len(db[r]) for r in runids]
# runid = runids[np.argmax(run_dict_lengths)]
runid = "NectarCAM_Run6310"
print(f"We will start with run {runid}")

print("Defining Select")
# runid_input = NumericInput(value=db.root.keys()[-1], title="NectarCAM run number")
# run_select = Select(value=runid, title="NectarCAM run number", options=runids)
run_select = Select(
    value=runid, title="NectarCAM run number", options=runids, css_classes=["select"]
)

print(f"Getting data for run {run_select.value}")
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

# controls = row(run_select)
controls = column(css_div, run_select)

# # TEST:
# attr = 'value'
# old = runid
# new = runids[1]
# update_camera_displays(attr, old, new)

# ncols = 3
# camera_displays = [
#     displays[parentkey][childkey].figure
#     for parentkey in displays.keys()
#     for childkey in displays[parentkey].keys()
# ]

list_timelines = [
    timelines[parentkey][childkey]
    for parentkey in timelines.keys()
    for childkey in timelines[parentkey].keys()
]

# layout_camera_displays = gridplot(
#     camera_displays,
#     ncols=ncols,
# )
layout_camera_displays = create_all_camera_display_layouts(displays)

layout_timelines = gridplot(
    list_timelines,
    ncols=2,
)

# Create different tabs
tab_camera_displays = TabPanel(
    child=column(*layout_camera_displays), title="Camera displays"
)
tab_timelines = TabPanel(child=layout_timelines, title="Timelines")

# Combine panels into tabs
tabs = Tabs(
    tabs=[tab_camera_displays, tab_timelines],
)

page_layout = column([row(controls, run_times_string), tabs], sizing_mode="scale_width")

run_select.on_change("value", update)

# Add to the Bokeh document
curdoc().add_root(page_layout)
curdoc().title = "NectarCAM Data Quality Monitoring web app"
