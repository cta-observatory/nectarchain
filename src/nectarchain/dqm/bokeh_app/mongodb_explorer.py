# mongodb_explorer.py
"""
TabPanel for a Bokeh server app — MongoDB explorer
====================================
Auto-discovers fields in a MongoDB collection, builds appropriate filter
controls for each field type, and displays matching documents in a DataTable.
"""

from datetime import date, datetime

import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DataTable,
    DateFormatter,
    DatetimeRangeSlider,
    Div,
    NumberFormatter,
    Select,
    StringFormatter,
    Switch,
    TableColumn,
    TextInput,
)
from pymongo import MongoClient

# Threshold: <= this many distinct values → Select, otherwise dual TextInput
NUMERIC_SELECT_THRESHOLD = 15


def get_collection(dburl, dbname, collname):
    client = MongoClient(dburl)
    db = client[dbname]
    collection = db[collname]
    return collection


def _infer_fields(collection, sample_size: int = 5000) -> dict:
    """
    Sample documents and return a dict:
        field_name -> {"type": "numeric"|"date"|"string"|"bool", "values": [...]}
    Skips the internal _id field.
    """

    docs = list(collection.find({}, {"_id": 0}).limit(sample_size))
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
        [Switch: "Exact value ±10%"]   ← inactive by default
        [TextInput: exact value    ]   ← disabled until toggle active
        ──────────────────────────
        [TextInput: min            ]   ← active by default
        [TextInput: max            ]

    When the switch is OFF  → range mode (min/max inputs); query also includes
                               docs where the field is missing/null.
    When the switch is ON   → exact mode (single value ±10%); only docs that
                               have the field and match the tolerance are returned.
    """

    TOLERANCE = 0.10  # ±10 %

    def __init__(self, name: str, lo: float, hi: float):
        self.name = name
        self.lo = lo
        self.hi = hi

        # ── Switch ──────────────────────────────────────────────────────────
        self.toggle = Switch(
            label=f"{name}: exact value ±10 %",
            active=False,
            sizing_mode="stretch_width",
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
            search_by_exact_value = bool(new)
            self.exact_input.disabled = not search_by_exact_value
            self.min_input.disabled = search_by_exact_value
            self.max_input.disabled = search_by_exact_value
            # on_change_cb(attr, old, new)

        # Wire switch input
        self.toggle.on_change("active", _on_toggle)

    # Convenience: return all Bokeh widgets for layout
    def widgets(self):
        return [self.toggle, self.exact_input, self.min_input, self.max_input]

    def mongo_filter(self) -> dict | None:
        """
        Return a MongoDB filter fragment for this field, or None if no filter
        should be applied.
        """
        if self.toggle.active:
            # ── Exact ±x % mode ─────────────────────────────────────────
            raw = self.exact_input.value.strip()
            if not raw:
                return None  # no value entered → no filter
            try:
                val = float(raw)
            except ValueError:
                return None
            tol = abs(val) * self.TOLERANCE
            lo = val - tol
            hi = val + tol
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
    ftype = meta["type"]
    values = meta["values"]
    if ftype == "numeric" and values:
        distinct = list(set(values))
        if len(distinct) <= NUMERIC_SELECT_THRESHOLD:
            # ── Low-cardinality: Select ──────────────────────────────────
            # Format options: keep ints as ints, floats as floats
            def _fmt(v):
                return (
                    str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
                )

            options = ["Any", "(missing)"] + [_fmt(v) for v in distinct]
            return Select(
                title=name,
                options=options,
                value="Any",
                sizing_mode="stretch_width",
            )
        else:
            # ── High-cardinality: dual TextInput with toggle ─────────────
            # (returned as NumericRangeControl
            lo = min(float(v) for v in distinct)
            hi = max(float(v) for v in distinct)
            return NumericRangeControl(name, lo, hi)

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
                start=lo,
                end=hi,
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
                query.update(frag)
            else:
                # Range mode: also include docs where the field is absent
                field_filter = frag[fname]  # e.g. {"$gte": lo, "$lte": hi}
                query["$or"] = query.get("$or", []) + [
                    {fname: field_filter},
                    {fname: {"$exists": False}},
                    {fname: None},
                ]
            continue

        # ── Low-cardinality numeric (Select) ─────────────────────────────
        if ftype == "numeric":
            val = widget.value
            if val == "Any":
                pass  # no filter
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
            lo_ms, hi_ms = widget.value  # milliseconds since epoch
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


class MongoExplorer:
    DEBOUNCE_MS = 400

    def __init__(self, uri: str, db: str, collection: str, max_docs: int = 5000):
        self.db_name = db
        self.coll_name = collection
        self.max_docs = max_docs
        self._debounce = None
        self._error = None
        self.collection = None

        try:
            self._client = MongoClient(uri)
            # PyMongo uses lazy connections — accessing the db/collection just
            # creates references.  Force the first actual network call here
            # (e.g. _infer_fields → collection.find) so that connection errors
            # are caught in this try/except block.
            self.collection = self._client[db][collection]
            self.FIELD_META = _infer_fields(self.collection)
            self.controls = {
                fname: _make_control(fname, fmeta)
                for fname, fmeta in self.FIELD_META.items()
            }

            # Wire controls
            for widget in self.controls.values():
                if isinstance(widget, NumericRangeControl):
                    for w in [widget.exact_input, widget.min_input, widget.max_input]:
                        w.on_change("value", self._schedule_update)
                else:
                    widget.on_change("value", self._schedule_update)
        except Exception as e:
            self._error = str(e)
            self.collection = None
            self._client = None
            self.FIELD_META = {}
            self.controls = {}

        self.source, self.status_div, self.panel = self._make_panel()
        if self._error is None:
            self._do_update()  # initial load

    def _schedule_update(self, attr, old, new):
        if self.collection is None:
            return
        if self._debounce is not None:
            try:
                curdoc().remove_timeout_callback(self._debounce)
            except ValueError:
                pass
        self._debounce = curdoc().add_timeout_callback(
            self._do_update, self.DEBOUNCE_MS
        )

    def _do_update(self):
        if self.collection is None:
            return
        self._debounce = None
        query = _build_query(self.FIELD_META, self.controls)
        cursor = self.collection.find(query, {"_id": 0}).limit(self.max_docs)
        docs = list(cursor)
        df = pd.DataFrame(docs)

        if not docs:
            self.status_div.text = "No documents match the current filters."
            self.source.data = {f: [] for f in self.FIELD_META}
        else:
            total = self.collection.count_documents(query)
            shown = len(df)
            if total > shown:
                self.status_div.text = (
                    f"<b>{shown}</b> documents shown"
                    f" (of {total} matching — increase max_docs to see more)"
                )
            else:
                self.status_div.text = f"All <b>{total}</b> matching documents shown."

            for fname in self.FIELD_META:
                if fname not in df.columns:
                    df[fname] = None
            for fname, fmeta in self.FIELD_META.items():
                if fmeta["type"] == "date" and fname in df.columns:
                    df[fname] = pd.to_datetime(df[fname], errors="coerce")

            self.source.data = {
                fname: df[fname].tolist()
                for fname in self.FIELD_META
                if fname in df.columns
            }

    def _make_panel(self):
        from bokeh.models import TabPanel

        if self._error is not None:
            error_div = Div(
                text=(
                    "<div style='color: #cc0000; padding: 20px;'>"
                    "<h3>&#9888;&#65039; MongoDB Connection Error</h3>"
                    "<p>Could not connect to "
                    f"<b>{self.db_name}.{self.coll_name}</b>:</p>"
                    "<pre style='background:#f5f5f5;padding:10px;"
                    f"overflow-x:auto;'>{self._error}</pre>"
                    "<p><em>Other tabs are unaffected.</em></p>"
                    "</div>"
                ),
                sizing_mode="stretch_width",
            )
            source = ColumnDataSource(data={})
            status_div = Div(text="")
            panel = TabPanel(child=error_div, title=f"{self.db_name}.{self.coll_name}")
            return source, status_div, panel

        header = Div(
            text=f"<h2 style='margin:0'>{self.db_name}.{self.coll_name} —   "
            f"<code>DB explorer </code></h2>",
            sizing_mode="stretch_width",
        )

        sidebar_children = []
        for widget in self.controls.values():
            if isinstance(widget, NumericRangeControl):
                sidebar_children.extend(widget.widgets())
            else:
                sidebar_children.append(widget)

        sidebar = column(
            *sidebar_children,
            sizing_mode="stretch_height",
            styles={"overflow-y": "auto", "max-width": "15%", "padding-right": "4px"},
        )

        status_div = Div(
            text="",
            styles={"font-size": "13px", "color": "#555", "margin-bottom": "6px"},
            sizing_mode="stretch_width",
        )

        source = ColumnDataSource(data={f: [] for f in self.FIELD_META})
        table_cols = _make_columns(self.FIELD_META)
        data_table = DataTable(
            source=source,
            columns=table_cols,
            sizing_mode="stretch_both",
        )

        layout = column(
            header,
            row(
                sidebar,
                column(status_div, data_table, sizing_mode="stretch_both"),
                sizing_mode="stretch_both",
            ),
            sizing_mode="stretch_both",
        )

        panel = TabPanel(child=layout, title=f"{self.db_name}.{self.coll_name}")
        return source, status_div, panel
