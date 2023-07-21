import numpy as np
from ZODB import DB

# bokeh imports
from bokeh.layouts import layout, row
from bokeh.models import Select
from bokeh.plotting import curdoc
from bokeh.io import output_file, save

from ctapipe.instrument import CameraGeometry
from ctapipe.coordinates import EngineeringCameraFrame

geom = CameraGeometry.from_name("NectarCam-003")
geom = geom.transform_to(EngineeringCameraFrame())

test_dict = {'run1': {'mykey1': {'mysubkey1': np.random.normal(size=geom.n_pixels),
                                 'mysubkey2': np.random.normal(size=geom.n_pixels)},
                      'mykey2': {'mysubkey1': np.random.normal(size=geom.n_pixels),
                                 'mysubkey2': np.random.normal(size=geom.n_pixels)}
                      },
             'run2': {'mykey1': {'mysubkey1': np.random.normal(size=geom.n_pixels),
                                 'mysubkey2': np.random.normal(size=geom.n_pixels)},
                      'mykey2': {'mysubkey1': np.random.normal(size=geom.n_pixels),
                                 'mysubkey2': np.random.normal(size=geom.n_pixels)}
                      }
             }
# Renders the second image incomplete
test_dict['run1']['mykey2']['mysubkey2'][10:20] = np.nan


def test_make_camera_display():
    from ..bokeh_app import make_camera_display
    for runid in list(test_dict.keys()):
        for parentkey in test_dict[runid].keys():
            for childkey in test_dict[runid][parentkey].keys():
                make_camera_display(test_dict[runid], parentkey, childkey)


def test_bokeh(tmp_path):
    from ..bokeh_app import make_camera_displays, update_camera_displays, \
        get_rundata

    db = DB(None)
    conn = db.open()
    root = conn.root()
    runids = sorted(list(test_dict.keys()))

    # Fill in-memory DB
    for runid in runids:
        root[runid] = test_dict[runid]

    runid = runids[-1]
    run_select = Select(value=runid, title='NectarCAM run number', options=runids)

    source = get_rundata(root, run_select.value)
    displays = make_camera_displays(root, source, runid)

    run_select.on_change('value', update_camera_displays)

    controls = row(run_select)

    ncols = 3
    plots = [displays[parentkey][childkey].figure for parentkey in displays.keys() for
             childkey in displays[parentkey].keys()]
    curdoc().add_root(layout([[controls],
                              [[plots[x:x + ncols] for x in
                                range(0, len(plots), ncols)]]],
                             sizing_mode='scale_width'
                             )
                      )
    curdoc().title = 'NectarCAM Data Quality Monitoring web app'

    output_path = tmp_path / "test.html"
    output_file(output_path)
    save(curdoc(), filename=output_path)
