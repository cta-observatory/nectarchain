"""Manual test: embed MongoExplorer as one tab in a multi-tab layout.

Run with:
    bokeh serve manual_test_mongo.py --show
or from the bokeh_app directory:
    python manual_test_mongo.py
"""

from bokeh.io import curdoc
from bokeh.models import Div, TabPanel, Tabs
from mongodb_explorer import MongoExplorer

# MongoExplorer uses defaults from env vars (MONGO_URI, MONGO_DB, MONGO_COLLECTION)
# or falls back to the hardcoded defaults in mongodb_explorer.py.
# Pass explicit arguments to override.
explorer = MongoExplorer()

another_tab = TabPanel(title="Another Tab", child=Div(text="Here will be dqm tabs"))
tabs = Tabs(tabs=[another_tab, explorer.panel], sizing_mode="stretch_both")
curdoc().add_root(tabs)
