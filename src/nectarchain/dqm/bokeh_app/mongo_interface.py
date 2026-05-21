"""
Bokeh server app — MongoDB explorer
====================================
Auto-discovers fields in a MongoDB collection, builds appropriate filter
controls for each field type, and displays matching documents in a DataTable.

Run with:
    bokeh serve main.py --show

Configuration: edit the three constants below or set environment variables
    MONGO_URI, MONGO_DB, MONGO_COLLECTION
"""

import os
from datetime import datetime, date

from pymongo import MongoClient
import pandas as pd

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    DateFormatter,
    Div,
    NumberFormatter,
    RangeSlider,
    DatetimeRangeSlider,
    Select,
    StringFormatter,
    TableColumn,
    TextInput,
    Toggle,
)

# ── Configuration ────────────────────────────────────────────────────────────

MONGO_URI        = os.getenv("MONGO_URI",        "mongodb://192.168.30.104:27017")
MONGO_DB         = os.getenv("MONGO_DB",         "test")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "runconfig")

# Maximum number of documents fetched for display (keep UI responsive)
MAX_DOCS = 5000
# Threshold: <= this many distinct values → Select, otherwise dual TextInput
NUMERIC_SELECT_THRESHOLD = 15
_debounce_handle = None
DEBOUNCE_MS = 400  # wait 400 ms after last slider move before querying

def get_collection():
    client     = MongoClient(MONGO_URI)
    db         = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    return collection

def _infer_fields(collection, sample_size: int = 5000) -> dict:
    """
    Sample documents and return a dict:
        field_name -> {"type": "numeric"|"date"|"string"|"bool", "values": [...]}
    Skips the internal _id field.
    """

    docs   = list(collection.find({}, {"_id": 0}).limit(sample_size))
    if not docs:
        return {}

    # Collect all values per field across the sample
    raw: dict[str, list] = {}
    for doc in docs:
        for k, v in doc.items():
            raw.setdefault(k, [])
            if v is not None:
                raw[k].append(v)

    fields = {}
    for name, values in raw.items():
        if name == "firstvar" or name == "secondvar" or name == "thirdvar":
            continue
        if not values:
            fields[name] = {"type": "string", "values": []}
            continue

        # Determine dominant type
        type_counts = {"numeric": 0, "date": 0, "bool": 0, "string": 0}
        for v in values:
            if isinstance(v, bool):
                type_counts["bool"] += 1
            elif isinstance(v, (int, float)):
                type_counts["numeric"] += 1
            elif isinstance(v, (datetime, date)):
                type_counts["date"] += 1
            else:
                type_counts["string"] += 1

        dominant = max(type_counts, key=type_counts.get)
        fields[name] = {"type": dominant, "values": values}

    return fields



# ── NumericControl helper ─────────────────────────────────────────────────────
 
class NumericRangeControl:
    """
    Compound control for a high-cardinality numeric field.
 
    Layout (inside the sidebar column):
        [Toggle: "Exact value ±10%"]   ← inactive by default
        [TextInput: exact value    ]   ← disabled until toggle active
        ──────────────────────────
        [TextInput: min            ]   ← active by default
        [TextInput: max            ]
 
    When the toggle is OFF  → range mode (min/max inputs); query also includes
                               docs where the field is missing/null.
    When the toggle is ON   → exact mode (single value ±10%); only docs that
                               have the field and match the tolerance are returned.
    """
 
    TOLERANCE = 0.10   # ±10 %
 
    def __init__(self, name: str, lo: float, hi: float, on_change_cb):
        self.name = name
        self.lo   = lo
        self.hi   = hi
 
        # ── Toggle ──────────────────────────────────────────────────────────
        self.toggle = Toggle(
            label=f"{name}: exact value ±10 %",
            active=False,
            sizing_mode="stretch_width",
            button_type="default",
        )
 
        # ── Exact-value input (disabled until toggle is ON) ──────────────
        self.exact_input = TextInput(
            title="Exact value",
            value="",
            disabled=True,
            sizing_mode="stretch_width",
        )
 
        # ── Range inputs (active by default) ────────────────────────────
        self.min_input = TextInput(
            title=f"{name}  —  min",
            value=str(lo),
            sizing_mode="stretch_width",
        )
        self.max_input = TextInput(
            title="max",
            value=str(hi),
            sizing_mode="stretch_width",
        )
 
        # Wire internal toggle → enable/disable sub-widgets
        def _on_toggle(attr, old, new):
            is_exact = bool(new)
            self.exact_input.disabled = not is_exact
            self.min_input.disabled   = is_exact
            self.max_input.disabled   = is_exact
            on_change_cb(attr, old, new)
 
        self.toggle.on_change("active", _on_toggle)
        self.exact_input.on_change("value", on_change_cb)
        self.min_input.on_change("value",   on_change_cb)
        self.max_input.on_change("value",   on_change_cb)
 
    # Convenience: return all Bokeh widgets for layout
    def widgets(self):
        return [self.toggle, self.exact_input, self.min_input, self.max_input]
 
    def mongo_filter(self) -> dict | None:
        """
        Return a MongoDB filter fragment for this field, or None if no filter
        should be applied.
        """
        if self.toggle.active:
            # ── Exact ±10 % mode ─────────────────────────────────────────
            raw = self.exact_input.value.strip()
            if not raw:
                return None          # no value entered → no filter
            try:
                val = float(raw)
            except ValueError:
                return None
            tol   = abs(val) * self.TOLERANCE
            lo    = val - tol
            hi    = val + tol
            return {self.name: {"$gte": lo, "$lte": hi}}
        else:
            # ── Range mode (include missing/null docs) ───────────────────
            try:
                lo = float(self.min_input.value.strip())
            except ValueError:
                lo = self.lo
            try:
                hi = float(self.max_input.value.strip())
            except ValueError:
                hi = self.hi
 
            # If the range covers the full sample extent → no filter needed
            # (keeps missing-field docs naturally included)
            if lo <= self.lo and hi >= self.hi:
                return None
 
            # Keep docs that satisfy the range OR that have the field missing
            return {self.name: {"$gte": lo, "$lte": hi}}
            # Note: "also return docs with field missing" is applied in
            # _build_query by wrapping with $or: [{filter}, {field: {$exists: false}}]
 








def _make_control(name: str, meta: dict):
    """Return the most appropriate Bokeh widget for the field."""
    ftype  = meta["type"]
    values = meta["values"]
    if ftype == "numeric" and values:
        distinct = list(dict.fromkeys(values))
        if len(distinct) <= NUMERIC_SELECT_THRESHOLD:
            # ── Low-cardinality: Select ──────────────────────────────────
            # Format options: keep ints as ints, floats as floats
            def _fmt(v):
                return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
            options = ["Any", "(missing)"] + [_fmt(v) for v in distinct]
            return Select(
                title=name,
                options=options,
                value="Any",
                sizing_mode="stretch_width",
            )
        else:
            # ── High-cardinality: dual TextInput with toggle ─────────────
            # (returned as NumericRangeControl; wired to update() below)
            lo = min(float(v) for v in distinct)
            hi = max(float(v) for v in distinct)
            return NumericRangeControl(name, lo, hi, on_change_cb=update)


    if ftype == "date" and values:
        dates = []
        for v in values:
            if isinstance(v, datetime):
                dates.append(v)
            elif isinstance(v, date):
                dates.append(datetime(v.year, v.month, v.day))
        if dates:
            lo = min(dates)
            hi = max(dates)
            if lo == hi:
                from datetime import timedelta
                hi = lo + timedelta(days=1)
            return DatetimeRangeSlider(
                title=name,
                start=lo, end=hi,
                value=(lo, hi),
                sizing_mode="stretch_width",
            )

    if ftype == "bool":
        return Select(
            title=name,
            options=["Any", "True", "False"],
            value="Any",
            sizing_mode="stretch_width",
        )

    # string  — free-text search
    return TextInput(
        title=f"{name} contains",
        value="",
        sizing_mode="stretch_width",
    )




def _make_columns(FIELD_META) -> list[TableColumn]:
    cols = []
    for fname, fmeta in FIELD_META.items():
        ftype = fmeta["type"]
        if ftype == "numeric":
            fmt = NumberFormatter(format="0,0.##")
        elif ftype == "date":
            fmt = DateFormatter(format="%Y-%m-%d__%H:%M:%S")
        else:
            fmt = StringFormatter()
        cols.append(TableColumn(field=fname, title=fname, formatter=fmt))
    return cols



# ── Query builder ─────────────────────────────────────────────────────────────

def _build_query(FIELD_META, controls: dict) -> dict:
    """Translate current widget values into a MongoDB filter dict."""
    query: dict = {}

    for fname, widget in controls.items():
        ftype = FIELD_META[fname]["type"]

        # ── High-cardinality numeric (NumericRangeControl) ────────────────
        if isinstance(widget, NumericRangeControl):
            frag = widget.mongo_filter()
            if frag is None:
                # No constraint → include everything (even missing field)
                continue
            if widget.toggle.active:
                # Exact mode: strict match, no missing-field inclusion
                query.update_runconfig_tab(frag)
            else:
                # Range mode: also include docs where the field is absent
                field_filter = frag[fname]   # e.g. {"$gte": lo, "$lte": hi}
                query["$or"] = query.get("$or", []) + [
                    {fname: field_filter},
                    {fname: {"$exists": False}},
                    {fname: None},
                    ]
                # field_filter = frag[fname]
                # query[fname] = {
                #     "$or": [
                #         field_filter,
                #         {"$exists": False},
                #         {"$eq": None},
                #     ]
                # }
            continue
 
        # ── Low-cardinality numeric (Select) ─────────────────────────────
        if ftype == "numeric":
            val = widget.value
            if val == "Any":
                pass   # no filter
            # elif val == "(missing)":
            #     query[fname] = {"$or": [{"$exists": False}, {"$eq": None}]}
            elif val == "(missing)":
                query["$or"] = query.get("$or", []) + [
                    {fname: {"$exists": False}},
                    {fname: None},
                    ]
            else:
                try:
                    query[fname] = {"$eq": float(val)}
                except ValueError:
                    pass
            continue
 
        # ── Date ─────────────────────────────────────────────────────────
        if ftype == "date":
            lo_ms, hi_ms = widget.value   # milliseconds since epoch
            if lo_ms > widget.start or hi_ms < widget.end:
                lo_dt = datetime.utcfromtimestamp(lo_ms / 1000)
                hi_dt = datetime.utcfromtimestamp(hi_ms / 1000)
                query[fname] = {"$gte": lo_dt, "$lte": hi_dt}
            continue
 
        # ── Bool ─────────────────────────────────────────────────────────
        if ftype == "bool":
            if widget.value == "True":
                query[fname] = {"$eq": True}
            elif widget.value == "False":
                query[fname] = {"$eq": False}
            continue
 
        # ── String / TextInput ────────────────────────────────────────────
        txt = widget.value.strip()
        if txt:
            query[fname] = {"$regex": txt, "$options": "i"}
 
    return query



def update_runconfig_tab(attr, old, new):
    global _debounce_handle
    if _debounce_handle is not None:
        try:
            curdoc().remove_timeout_callback(_debounce_handle)
        except ValueError:
            pass  # already fired, safe to ignore
    _debounce_handle = curdoc().add_timeout_callback(_do_update, DEBOUNCE_MS)


def _do_update():
    global _debounce_handle
    _debounce_handle = None

    query  = _build_query(FIELD_META, controls)
    cursor = collection.find(query, {"_id": 0}).limit(MAX_DOCS)
    docs   = list(cursor)

    query = _build_query(FIELD_META, controls)
    cursor = collection.find(query, {"_id": 0}).limit(MAX_DOCS)
    docs   = list(cursor)
    df = pd.DataFrame(docs)

     
    if not docs:
        # update div text to show no results, 
        status_div.text = "No documents match the current filters."
        # and clear the table (keep columns so user can adjust filters and see results)
        source.data = {f: [] for f in FIELD_META}
    else:
        total = collection.count_documents(query)
        shown = len(df)
        if total > shown :
            status_div.text = (
                f"<b>{shown}</b> documents shown"
                + (f" (of {total} matching — increase MAX_DOCS to see more)" )
            )
        else:
            status_div.text = f"All <b>{total}</b> matching documents shown."
    


        # Ensure all expected columns exist (some docs may lack optional fields)
        for fname in FIELD_META:
            if fname not in df.columns:
                df[fname] = None

        # Convert datetime columns to ms-since-epoch so Bokeh DateFormatter works
        for fname, fmeta in FIELD_META.items():
            if fmeta["type"] == "date" and fname in df.columns:
                df[fname] = pd.to_datetime(df[fname], errors="coerce")

        # finally update the source with the new data
        source.data = {fname: df[fname].tolist() for fname in FIELD_META if fname in df.columns}



# ── Layout ────────────────────────────────────────────────────────────────────
def make_layout(FIELD_META):
    header = Div(
        text=f"<h2 style='margin:0'>MongoDB explorer — <code>{MONGO_DB}.{MONGO_COLLECTION}</code></h2>",
        sizing_mode="stretch_width",
    )
    sidebar_children = []
    for widget in controls.values():
        if isinstance(widget, NumericRangeControl):
            sidebar_children.extend(widget.widgets())
        else:
            sidebar_children.append(widget)
    sidebar = column(
        *sidebar_children,
        # width=280,
        sizing_mode="stretch_height",
        styles={"overflow-y": "auto", "max-height": "90vh", "padding-right": "8px"},
    )
    status_div = Div(
    text="",
    styles={"font-size": "13px", "color": "#555", "margin-bottom": "6px"},
    sizing_mode="stretch_width",
    )
    # ── DataTable setup ───────────────────────────────────────────────────────────
    source     = ColumnDataSource(data={f: [] for f in FIELD_META})
    table_cols = _make_columns(FIELD_META)
    data_table = DataTable(
        source=source,
        columns=table_cols,
        sizing_mode="stretch_both",
        # height=600,
        # autosize_mode="force_fit",
    )
    main_area = column(
        status_div,
        data_table,
        sizing_mode="stretch_both",
    )

    layout = column(
    header,
    row(sidebar, main_area, sizing_mode="stretch_both"),
    sizing_mode="stretch_both",
    )
    return layout, source, status_div


# ── Connect & introspect ──────────────────────────────────────────────────────
collection = get_collection()
FIELD_META = _infer_fields(collection)
# ── Build controls ────────────────────────────────────────────────────────────
controls: dict = {}   # field_name -> bokeh widget
for fname, fmeta in FIELD_META.items():
    controls[fname] = _make_control(fname, fmeta)
# ── Wire simple controls (NumericRangeControl wires itself in __init__) ───────
for widget in controls.values():
    if not isinstance(widget, NumericRangeControl):
        widget.on_change("value", update_runconfig_tab) 
        # update_runconfig_tab will have the signature (attr, old, new) as required by Bokeh
        # with attr being the widget value here. 
layout, source, status_div = make_layout(FIELD_META)
update_runconfig_tab(None, None, None)   # initial data load

curdoc().add_root(layout)
curdoc().title = f"{MONGO_DB}.{MONGO_COLLECTION} explorer"
