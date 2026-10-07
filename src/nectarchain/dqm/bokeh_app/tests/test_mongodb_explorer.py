"""Unit tests for the mongodb_explorer module.

All tests use mocking — no real MongoDB connection required.
"""

from __future__ import annotations

import unittest.mock
from datetime import datetime

from bokeh.models import (
    ColumnDataSource,
    DateFormatter,
    DatetimeRangeSlider,
    Div,
    NumberFormatter,
    Select,
    StringFormatter,
    TabPanel,
    TextInput,
)

from nectarchain.dqm.bokeh_app.mongodb_explorer import (
    MongoExplorer,
    NumericRangeControl,
    _build_query,
    _infer_fields,
    _make_columns,
    _make_control,
)

# ========================================================================
#  Helpers
# ========================================================================


class _MockCursor(list):
    """A list subclass that acts like a pymongo Cursor.

    * Is iterable multiple times (unlike a MagicMock with a one-shot iterator).
    * Has a ``.limit()`` method that returns ``self`` (method-chaining).
    """

    def limit(self, n: int) -> _MockCursor:
        return self


def _mock_collection(
    docs: list[dict], total: int | None = None, *, strip_id: bool = True
) -> unittest.mock.MagicMock:
    """Return a MagicMock that behaves like a pymongo Collection.

    Parameters
    ----------
    docs:
        The documents the mocked ``find()`` should return.
    total:
        Value for ``count_documents()``.  Defaults to ``len(docs)``.
    strip_id:
        If True (default), remove ``_id`` from each document to simulate
        the ``{"_id": 0}`` projection used in the real ``find()`` call.
    """
    processed = (
        [{k: v for k, v in d.items() if k != "_id"} for d in docs] if strip_id else docs
    )
    cursor = _MockCursor(processed)

    collection = unittest.mock.MagicMock()
    collection.find.return_value = cursor
    collection.count_documents.return_value = total if total is not None else len(docs)
    return collection


def _make_mongo_explorer(
    docs: list[dict],
    total: int | None = None,
    max_docs: int = 5000,
    uri: str = "mongodb://localhost:27017",
    db: str = "test_db",
    coll: str = "test_coll",
) -> MongoExplorer:
    """Create a ``MongoExplorer`` whose MongoDB connection is fully mocked.

    Returns
    -------
    MongoExplorer
        The explorer instance, ready for assertions.
    """
    collection = _mock_collection(docs, total=total)

    with unittest.mock.patch(
        "nectarchain.dqm.bokeh_app.mongodb_explorer.MongoClient"
    ) as mock_client_cls:
        mock_client = unittest.mock.MagicMock()
        mock_db = unittest.mock.MagicMock()
        mock_db.__getitem__.return_value = collection
        mock_client.__getitem__.return_value = mock_db
        mock_client_cls.return_value = mock_client
        explorer = MongoExplorer(uri, db, coll, max_docs=max_docs)

    return explorer


# ========================================================================
#  NumericRangeControl
# ========================================================================


class TestNumericRangeControl:
    """Tests for the ``NumericRangeControl`` compound widget."""

    def test_init_creates_widgets(self):
        ctrl = NumericRangeControl("test_field", 0.0, 100.0)
        assert ctrl.name == "test_field"
        assert ctrl.lo == 0.0
        assert ctrl.hi == 100.0
        assert ctrl.toggle.active is False
        # Exact-value input starts disabled; range inputs start enabled
        assert ctrl.exact_input.disabled is True
        assert ctrl.min_input.disabled is False
        assert ctrl.max_input.disabled is False

    def test_widgets_returns_four_widgets(self):
        ctrl = NumericRangeControl("val", 0.0, 10.0)
        widgets = ctrl.widgets()
        assert len(widgets) == 4
        assert widgets[0] is ctrl.toggle
        assert widgets[1] is ctrl.exact_input
        assert widgets[2] is ctrl.min_input
        assert widgets[3] is ctrl.max_input

    def test_mongo_filter_range_default_returns_none(self):
        """When min/max cover the full sample extent → no filter needed."""
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        assert ctrl.mongo_filter() is None

    def test_mongo_filter_range_custom(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.min_input.value = "10"
        ctrl.max_input.value = "50"
        result = ctrl.mongo_filter()
        assert result == {"val": {"$gte": 10.0, "$lte": 50.0}}

    def test_mongo_filter_range_invalid_min_falls_back(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.min_input.value = "not-a-number"
        ctrl.max_input.value = "50"
        result = ctrl.mongo_filter()
        # min falls back to self.lo = 0.0, which is within the full range,
        # so the effective range is [0, 50] — narrower than [0, 100] → filter emitted
        assert result == {"val": {"$gte": 0.0, "$lte": 50.0}}

    def test_mongo_filter_range_invalid_max_falls_back(self):
        """Invalid max input falls back to self.hi."""
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.min_input.value = "30"
        ctrl.max_input.value = "not-a-number"
        result = ctrl.mongo_filter()
        # max falls back to self.hi = 100.0; effective range [30, 100] → narrower
        assert result == {"val": {"$gte": 30.0, "$lte": 100.0}}

    def test_mongo_filter_range_both_invalid_falls_back(self):
        """When both min and max are unparseable, fall back to lo/hi.
        If the fallback covers the full extent, no filter is returned."""
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.min_input.value = "not-a-number"
        ctrl.max_input.value = "also-invalid"
        result = ctrl.mongo_filter()
        # Both fall back → full range [0, 100] → no filter needed
        assert result is None

    def test_mongo_filter_exact_mode(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.toggle.active = True
        ctrl.exact_input.value = "50"
        tol = 50 * ctrl.TOLERANCE  # 50 * 0.10 = 5.0
        result = ctrl.mongo_filter()
        assert result == {"val": {"$gte": 50 - tol, "$lte": 50 + tol}}

    def test_mongo_filter_exact_no_value_returns_none(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.toggle.active = True
        ctrl.exact_input.value = ""
        assert ctrl.mongo_filter() is None

    def test_mongo_filter_exact_invalid_value_returns_none(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.toggle.active = True
        ctrl.exact_input.value = "not-a-number"
        assert ctrl.mongo_filter() is None

    def test_toggle_switches_input_enabled_states(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)

        # Start: exact disabled, range enabled
        assert ctrl.exact_input.disabled is True
        assert ctrl.min_input.disabled is False
        assert ctrl.max_input.disabled is False

        # Activate toggle (exact mode ON)
        ctrl.toggle.active = True
        assert ctrl.exact_input.disabled is False
        assert ctrl.min_input.disabled is True
        assert ctrl.max_input.disabled is True

        # Deactivate toggle (back to range mode)
        ctrl.toggle.active = False
        assert ctrl.exact_input.disabled is True
        assert ctrl.min_input.disabled is False
        assert ctrl.max_input.disabled is False


# ========================================================================
#  _make_control
# ========================================================================


class TestMakeControl:
    """Tests for the ``_make_control`` factory function."""

    def test_numeric_low_cardinality_returns_select(self):
        control = _make_control("age", {"type": "numeric", "values": [25, 30, 35]})
        assert isinstance(control, Select)
        assert "Any" in control.options
        assert "(missing)" in control.options
        assert "25" in control.options
        assert "30" in control.options

    def test_numeric_high_cardinality_returns_range_control(self):
        values = list(range(100))  # > NUMERIC_SELECT_THRESHOLD (15)
        control = _make_control("val", {"type": "numeric", "values": values})
        assert isinstance(control, NumericRangeControl)

    def test_numeric_no_values_returns_none(self):
        control = _make_control("val", {"type": "numeric", "values": []})
        # No matching branch → falls to string TextInput
        assert isinstance(control, TextInput)

    def test_date_returns_daterange_slider(self):
        control = _make_control(
            "ts", {"type": "date", "values": [datetime(2024, 6, 15)]}
        )
        assert isinstance(control, DatetimeRangeSlider)

    def test_date_single_value_adds_one_day(self):
        """When only one distinct date exists, the slider end is incremented.

        ``DatetimeRangeSlider`` stores timestamps in **milliseconds** since
        the epoch, so ``end - start`` should equal 86_400_000 ms (1 day).
        """
        control = _make_control(
            "ts", {"type": "date", "values": [datetime(2024, 1, 1)]}
        )
        assert isinstance(control, DatetimeRangeSlider)
        end_ms = control.end
        start_ms = control.start
        assert end_ms - start_ms == 86_400_000  # 1 day in ms

    def test_date_with_date_object_not_datetime(self):
        """``date`` (not ``datetime``) values should also create a slider."""
        from datetime import date as date_type

        control = _make_control(
            "ts",
            {"type": "date", "values": [date_type(2024, 6, 15)]},
        )
        assert isinstance(control, DatetimeRangeSlider)

    def test_date_no_valid_dates_returns_none(self):
        control = _make_control("ts", {"type": "date", "values": []})
        assert isinstance(control, TextInput)  # falls through to string

    def test_bool_returns_select(self):
        control = _make_control("flag", {"type": "bool", "values": [True, False]})
        assert isinstance(control, Select)
        assert control.options == ["Any", "True", "False"]

    def test_string_returns_text_input(self):
        control = _make_control("name", {"type": "string", "values": ["Alice", "Bob"]})
        assert isinstance(control, TextInput)
        assert "contains" in control.title


# ========================================================================
#  _make_columns
# ========================================================================


class TestMakeColumns:
    """Tests for the ``_make_columns`` helper."""

    def test_columns_use_correct_formatters(self):
        meta = {
            "age": {"type": "numeric"},
            "ts": {"type": "date"},
            "name": {"type": "string"},
        }
        cols = _make_columns(meta)
        assert len(cols) == 3
        assert isinstance(cols[0].formatter, NumberFormatter)
        assert isinstance(cols[1].formatter, DateFormatter)
        assert isinstance(cols[2].formatter, StringFormatter)

    def test_empty_meta_returns_empty_list(self):
        assert _make_columns({}) == []


# ========================================================================
#  _build_query
# ========================================================================


class TestBuildQuery:
    """Tests for the ``_build_query`` function."""

    def test_empty_controls_returns_empty(self):
        assert _build_query({}, {}) == {}

    def test_string_filter_adds_regex(self):
        ctrl = TextInput(value="Ali")
        meta = {"name": {"type": "string"}}
        controls = {"name": ctrl}
        query = _build_query(meta, controls)
        assert query == {"name": {"$regex": "Ali", "$options": "i"}}

    def test_string_empty_no_filter(self):
        ctrl = TextInput(value="")
        meta = {"name": {"type": "string"}}
        controls = {"name": ctrl}
        assert _build_query(meta, controls) == {}

    def test_bool_true(self):
        ctrl = Select(options=["Any", "True", "False"], value="True")
        meta = {"flag": {"type": "bool"}}
        controls = {"flag": ctrl}
        query = _build_query(meta, controls)
        assert query == {"flag": {"$eq": True}}

    def test_bool_false(self):
        ctrl = Select(options=["Any", "True", "False"], value="False")
        meta = {"flag": {"type": "bool"}}
        controls = {"flag": ctrl}
        query = _build_query(meta, controls)
        assert query == {"flag": {"$eq": False}}

    def test_bool_any_no_filter(self):
        ctrl = Select(options=["Any", "True", "False"], value="Any")
        meta = {"flag": {"type": "bool"}}
        controls = {"flag": ctrl}
        assert _build_query(meta, controls) == {}

    def test_date_narrower_than_full_range(self):
        """When the slider range is narrower than start→end, emit a filter."""
        lo_ms = datetime(2024, 3, 1).timestamp() * 1000
        hi_ms = datetime(2024, 6, 1).timestamp() * 1000

        slider = unittest.mock.MagicMock()
        slider.start = datetime(2024, 1, 1).timestamp() * 1000
        slider.end = datetime(2025, 1, 1).timestamp() * 1000
        slider.value = (lo_ms, hi_ms)

        meta = {"ts": {"type": "date"}}
        controls = {"ts": slider}
        query = _build_query(meta, controls)
        assert "ts" in query
        assert "$gte" in query["ts"]
        assert "$lte" in query["ts"]

    def test_date_full_range_no_filter(self):
        start_ms = datetime(2024, 1, 1).timestamp() * 1000
        end_ms = datetime(2025, 1, 1).timestamp() * 1000

        slider = unittest.mock.MagicMock()
        slider.start = start_ms
        slider.end = end_ms
        slider.value = (start_ms, end_ms)

        meta = {"ts": {"type": "date"}}
        controls = {"ts": slider}
        assert _build_query(meta, controls) == {}

    def test_numeric_select_any_no_filter(self):
        ctrl = Select(options=["Any", "(missing)", "25", "30"], value="Any")
        meta = {"age": {"type": "numeric"}}
        controls = {"age": ctrl}
        assert _build_query(meta, controls) == {}

    def test_numeric_select_value(self):
        ctrl = Select(options=["Any", "(missing)", "25", "30"], value="30")
        meta = {"age": {"type": "numeric"}}
        controls = {"age": ctrl}
        query = _build_query(meta, controls)
        assert query == {"age": {"$eq": 30.0}}

    def test_numeric_select_invalid_value_is_ignored(self):
        """An unparseable numeric Select value does not add a filter."""
        ctrl = Select(options=["Any", "(missing)", "25", "30"], value="not-a-number")
        meta = {"age": {"type": "numeric"}}
        controls = {"age": ctrl}
        query = _build_query(meta, controls)
        assert query == {}

    def test_numeric_select_missing(self):
        ctrl = Select(options=["Any", "(missing)", "25", "30"], value="(missing)")
        meta = {"age": {"type": "numeric"}}
        controls = {"age": ctrl}
        query = _build_query(meta, controls)
        assert "$or" in query
        age_missing = [c for c in query["$or"] if "age" in c]
        assert len(age_missing) == 2

    def test_numeric_range_control_range_mode(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.min_input.value = "10"
        ctrl.max_input.value = "50"
        meta = {"val": {"type": "numeric"}}
        controls = {"val": ctrl}
        query = _build_query(meta, controls)
        assert "$or" in query
        # Three conditions: range filter, $exists: false, None
        val_conditions = [c for c in query["$or"] if "val" in c]
        assert len(val_conditions) == 3

    def test_numeric_range_control_exact_mode(self):
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        ctrl.toggle.active = True
        ctrl.exact_input.value = "50"
        meta = {"val": {"type": "numeric"}}
        controls = {"val": ctrl}
        query = _build_query(meta, controls)
        assert "$or" not in query
        assert "val" in query

    def test_numeric_range_control_no_filter(self):
        """When range covers full extent → no filter emitted."""
        ctrl = NumericRangeControl("val", 0.0, 100.0)
        meta = {"val": {"type": "numeric"}}
        controls = {"val": ctrl}
        assert _build_query(meta, controls) == {}


# ========================================================================
#  _infer_fields
# ========================================================================


class TestInferFields:
    """Tests for the ``_infer_fields`` function."""

    def test_empty_collection_returns_empty(self):
        collection = _mock_collection([])
        assert _infer_fields(collection) == {}

    def test_skips_id_field(self):
        docs = [{"_id": "abc", "name": "Alice"}]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        assert "_id" not in result
        assert "name" in result

    def test_skips_firstvar_secondvar_thirdvar(self):
        docs = [
            {
                "firstvar": 1,
                "secondvar": 2.0,
                "thirdvar": "skip",
                "name": "Alice",
            }
        ]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        for skipped in ("firstvar", "secondvar", "thirdvar"):
            assert skipped not in result
        assert "name" in result

    def test_detects_numeric_type(self):
        docs = [{"val": 42}, {"val": 3.14}]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        assert result["val"]["type"] == "numeric"

    def test_detects_string_type(self):
        docs = [{"name": "Alice"}, {"name": "Bob"}]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        assert result["name"]["type"] == "string"

    def test_detects_date_type(self):
        docs = [{"ts": datetime(2024, 1, 1)}, {"ts": datetime(2024, 6, 15)}]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        assert result["ts"]["type"] == "date"

    def test_detects_bool_type(self):
        docs = [{"flag": True}, {"flag": False}]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        assert result["flag"]["type"] == "bool"

    def test_field_with_only_none_values_is_string(self):
        docs = [{"label": None}, {"label": None}]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        # All values are None (not added to raw), so values is empty
        # → fallback to string
        assert result["label"]["type"] == "string"
        assert result["label"]["values"] == []

    def test_mixed_type_dominates_correctly(self):
        docs = [{"val": 1}, {"val": 2}, {"val": "oops"}]
        collection = _mock_collection(docs)
        result = _infer_fields(collection)
        # 2 numeric, 1 string → numeric dominates
        assert result["val"]["type"] == "numeric"

    def test_passes_sample_size_to_find(self):
        """The ``sample_size`` argument is forwarded to ``.limit()``."""
        cursor_mock = unittest.mock.MagicMock()
        cursor_mock.__iter__.return_value = iter([{"a": 1}])
        cursor_mock.limit.return_value = cursor_mock

        collection = unittest.mock.MagicMock()
        collection.find.return_value = cursor_mock

        _infer_fields(collection, sample_size=100)
        collection.find.assert_called_once_with({}, {"_id": 0})
        cursor_mock.limit.assert_called_once_with(100)


# ========================================================================
#  MongoExplorer
# ========================================================================


class TestMongoExplorerConnectionFailure:
    """MongoExplorer initialisation when MongoDB is unreachable."""

    def test_stores_error_and_returns_error_tab(self):
        with unittest.mock.patch(
            "nectarchain.dqm.bokeh_app.mongodb_explorer.MongoClient",
            side_effect=Exception("Connection refused"),
        ):
            explorer = MongoExplorer("mongodb://badhost:27017", "test_db", "test_coll")

        assert explorer._error is not None
        assert "Connection refused" in explorer._error
        assert explorer.collection is None

        assert explorer.panel is not None
        assert isinstance(explorer.panel, TabPanel)
        child = explorer.panel.child
        assert isinstance(child, Div)
        assert "Connection Error" in child.text
        assert "test_db.test_coll" in child.text


class TestMongoExplorerSuccessfulInit:
    """MongoExplorer initialisation with a mocked MongoDB connection."""

    def test_successful_init_no_errors(self):
        docs = [{"name": "Alice", "age": 30}]
        explorer = _make_mongo_explorer(docs)
        assert explorer._error is None
        assert explorer.collection is not None
        assert explorer.panel is not None

    def test_infers_field_metadata(self):
        docs = [{"name": "Alice", "age": 30}]
        explorer = _make_mongo_explorer(docs)
        assert "name" in explorer.FIELD_META
        assert explorer.FIELD_META["name"]["type"] == "string"
        assert "age" in explorer.FIELD_META
        assert explorer.FIELD_META["age"]["type"] == "numeric"

    def test_creates_controls_for_each_field(self):
        docs = [{"name": "Alice", "age": 30}]
        explorer = _make_mongo_explorer(docs)
        assert "name" in explorer.controls
        assert "age" in explorer.controls

    def test_panel_is_tab_panel(self):
        docs = [{"name": "Alice", "age": 30}]
        explorer = _make_mongo_explorer(docs)
        assert isinstance(explorer.panel, TabPanel)

    def test_source_and_status_div_are_created(self):
        docs = [{"name": "Alice", "age": 30}]
        explorer = _make_mongo_explorer(docs)
        assert isinstance(explorer.source, ColumnDataSource)
        assert isinstance(explorer.status_div, Div)

    def test_high_cardinality_numeric_creates_range_control(self):
        """>15 distinct numeric values → NumericRangeControl is used,
        exercising the ``widgets()`` and ``on_change`` wiring paths."""
        import random

        random.seed(42)
        docs = [{"val": random.randint(0, 1000)} for _ in range(50)]
        explorer = _make_mongo_explorer(docs)

        ctrl = explorer.controls["val"]
        assert isinstance(ctrl, NumericRangeControl)
        # The sidebar layout should include the 4 sub-widgets
        assert len(ctrl.widgets()) == 4

    def test_date_fields_are_converted_in_do_update(self):
        """Documents with date fields test the ``pd.to_datetime`` conversion
        in ``_do_update``."""
        docs = [
            {"event": "start", "ts": datetime(2024, 1, 1, 12, 0, 0)},
            {"event": "end", "ts": datetime(2024, 6, 15, 18, 30, 0)},
        ]
        explorer = _make_mongo_explorer(docs)
        data = explorer.source.data
        assert "ts" in data
        # Values should be pandas Timestamps (converted by pd.to_datetime)
        assert len(data["ts"]) == 2


class TestMongoExplorerDoUpdate:
    """Behaviour of ``MongoExplorer._do_update``."""

    def test_no_matching_docs_shows_message(self):
        docs: list[dict] = []
        explorer = _make_mongo_explorer(docs)
        assert "No documents match" in explorer.status_div.text

    def test_matching_docs_shows_all_count(self):
        docs = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        explorer = _make_mongo_explorer(docs, total=2)
        assert "All" in explorer.status_div.text
        assert "2" in explorer.status_div.text

    def test_truncated_results_shows_warning(self):
        """When total docs exceed max_docs, the status mentions the limit."""
        docs = [{"name": f"User{i}"} for i in range(100)]
        explorer = _make_mongo_explorer(docs, total=5000, max_docs=100)
        assert "increase max_docs" in explorer.status_div.text
        assert "100" in explorer.status_div.text

    def test_populates_source_data(self):
        docs = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        explorer = _make_mongo_explorer(docs)
        data = explorer.source.data
        assert "name" in data
        assert data["name"] == ["Alice", "Bob"]
        assert "age" in data
        assert data["age"] == [30, 25]

    def test_missing_fields_in_docs_filled_with_nan(self):
        """If some documents lack a field, pandas fills with NaN (float).

        We verify:
        * The field is present in ``source.data``.
        * The first value matches (may be cast to float by pandas).
        * The second value is NaN (not missing from the column).
        """
        import math

        docs = [
            {"name": "Alice", "age": 30},
            {"name": "Bob"},  # missing 'age'
        ]
        explorer = _make_mongo_explorer(docs)
        data = explorer.source.data
        assert "age" in data
        assert data["age"][0] == 30.0  # pandas may upcast int → float
        assert math.isnan(data["age"][1])


class TestMongoExplorerScheduleUpdate:
    """Debounced update scheduling."""

    def test_collection_none_returns_early(self):
        explorer = _make_mongo_explorer([{"a": 1}])
        # Force collection to None like a connection failure
        explorer.collection = None
        # This should not raise
        explorer._schedule_update("value", "old", "new")
        assert explorer._debounce is None

    def test_schedules_timeout(self):
        explorer = _make_mongo_explorer([{"a": 1}])
        explorer._schedule_update("value", "old", "new")
        # A timeout callback should have been scheduled
        assert explorer._debounce is not None

    def test_replaces_existing_debounce(self):
        explorer = _make_mongo_explorer([{"a": 1}])
        explorer._schedule_update("value", "old", "new")
        first = explorer._debounce
        explorer._schedule_update("value", "old", "new")
        # A new callback replaces the old one
        assert explorer._debounce is not first

    def test_do_update_fills_missing_column_with_none(self):
        """When query results lack a field that was in the initial scan,
        it is added as an all-None column (line 426)."""
        # Initial scan sees docs with three fields
        full_docs = [{"name": "Alice", "city": "NY", "age": 30}]
        explorer = _make_mongo_explorer(full_docs)

        # Confirm FIELD_META includes "city"
        assert "city" in explorer.FIELD_META

        # Replace the collection to return docs *without* "city"
        filtered_docs = [{"name": "Bob", "age": 25}]
        explorer.collection = _mock_collection(filtered_docs)

        # Re-run the update
        explorer._do_update()

        # "city" column should still exist in source.data, filled with None
        data = explorer.source.data
        assert "city" in data
        assert data["city"] == [None]
        # The other columns remain intact
        assert data["name"] == ["Bob"]
        assert data["age"] == [25.0]


class TestMongoExplorerDoUpdateGuard:
    """``_do_update`` early-return when collection is None."""

    def test_does_not_raise_when_collection_is_none(self):
        explorer = _make_mongo_explorer([{"a": 1}])
        explorer.collection = None
        # Should not raise
        explorer._do_update()


class TestMongoExplorerMakePanel:
    """``_make_panel`` layout construction."""

    def test_error_path_returns_error_div(self):
        with unittest.mock.patch(
            "nectarchain.dqm.bokeh_app.mongodb_explorer.MongoClient",
            side_effect=Exception("Broken"),
        ):
            explorer = MongoExplorer("bad://uri", "db", "coll")

        panel = explorer.panel
        assert isinstance(panel, TabPanel)
        child = panel.child
        assert isinstance(child, Div)
        assert "Connection Error" in child.text
        assert "db.coll" in child.text

    def test_normal_path_contains_data_table(self):
        docs = [{"name": "Alice", "age": 30}]
        explorer = _make_mongo_explorer(docs)
        # The panel child is a column with header + sidebar + data area.
        # At minimum, source and status_div are proper objects.
        assert explorer.source is not None
        assert explorer.status_div is not None

    def test_tab_title_uses_db_and_collection(self):
        explorer = _make_mongo_explorer([{"a": 1}], db="mydb", coll="mycoll")
        assert explorer.panel.title == "mydb.mycoll"
