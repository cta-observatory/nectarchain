import datetime
import logging
import os
import sqlite3
from enum import Flag, auto
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate
from tqdm import tqdm

from .datautils import (
    FindFile,
    FindFiles,
    GetDAQTimeFromTime,
    GetDBNameFromTime,
    GetDefaultDataPath,
    GetFirstLastEventTime,
    to_datetime,
)

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    filename=f"{os.environ.get('NECTARCHAIN_LOG', '/tmp')}/{os.getpid()}/"
    f"{Path(__file__).stem}_{os.getpid()}.log",
    handlers=[logging.getLogger("__main__").handlers],
)
log = logging.getLogger(__name__)


class DBInfosFlag(Flag):
    """Flag enum indicating the granularity of database information.

    Attributes
    ----------
    CAMERA : auto
        Information at the camera level.
    DRAWER : auto
        Information at the drawer (module) level.
    PIXEL : auto
        Information at the pixel level.
    """

    CAMERA = auto()
    DRAWER = auto()
    PIXEL = auto()


class DictInfos:
    """Dictionary-like container that promotes keys to instance attributes.

    New entries can be added via the ``[]`` operator; once entered they
    become accessible as member attributes.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the container with an empty internal dictionary."""
        self.infos = dict()

    def __setitem__(self, key, value):
        """Set ``key`` in the internal dict and as an instance attribute."""
        self.infos[key] = value
        setattr(self, key, value)

    def __getitem__(self, key):
        """Return the value associated with *key*."""
        try:
            return self.infos[key]
        except Exception:
            raise AttributeError

    def __contains__(self, key):
        """Return ``True`` if *key* is present in the container."""
        return key in self.infos

    def set_time(self, t):
        """Propagate *t* to all stored value objects that support ``set_time``."""
        for v in self.infos.values():
            try:
                v.set_time(t)
            except Exception:
                pass


class DBCameraInfos(DictInfos):
    """Dictionary of table-level information for a single camera telescope."""

    def __init__(self, tel, df=None, *args, **kwargs):
        """Initialise with telescope ID and optional dataframe.

        Parameters
        ----------
        tel : int
            Telescope ID.
        df : pd.DataFrame or None, optional
            Associated data.
        """
        super().__init__(*args, **kwargs)
        self.tel = tel
        self.df = df


class DBTableInfos(DictInfos):
    """Dictionary of column-level information for a single database table."""

    def __init__(self, table_name, df=None, *args, **kwargs):
        """Initialise with table name and optional dataframe.

        Parameters
        ----------
        table_name : str
            Name of the database table.
        df : pd.DataFrame or None, optional
            Associated data.
        """
        super().__init__(self, *args, **kwargs)
        self.table_name = table_name
        self.df = df


class DBColumnInfos(DictInfos):
    """Dictionary of information for a single database column."""

    def __init__(self, column_name, df=None, *args, **kwargs):
        """Initialise with column name and optional dataframe.

        Parameters
        ----------
        column_name : str
            Name of the database column.
        df : pd.DataFrame or None, optional
            Associated data.
        """
        super().__init__(self, *args, **kwargs)
        self.column_name = column_name
        self.df = df


class CameraArray(np.ndarray):
    """Subclass of :class:`~numpy.ndarray` carrying information about camera axes.

    Tracks axes whose size matches the number of camera elements (modules
    or pixels).
    """

    _info_axis = None
    _nElements = -1

    def __new__(cls, a):
        """Create a new ``CameraArray`` by viewing *a* as this subclass."""
        obj = np.asarray(a).view(cls)
        return obj

    def __init__(self, a, mod_axis=None):
        """
        init function

        Parameters
        ----------

        a : np.array
            An input numpy array that contains information on the module level
        mod_axis : int or list
            Axis position (int or list of int) of the module info.
            If None, it will take all axis that has a 265 size.
        """
        if self._info_axis is None:
            self._info_axis = list()

        if mod_axis is None:
            self._guess_axis()
        else:
            try:
                for ax in mod_axis:
                    self.infos_axis.append(ax)
            except Exception:
                # must be a single value
                self.infos_axis.append(mod_axis)

    @property
    def nElements(self):
        """Number of camera elements (modules or pixels) represented by this array."""
        return self._nElements

    @property
    def info_axis(self):
        """Axis indices whose size equals ``nElements``, auto-guessed if needed."""
        self._guess_axis()
        return self._info_axis

    def prod(self, a, **kwargs):
        """Calls ~numpy.array.prod

        This custom documentation for ~numpy.array.prod is necessary, otherwise
        Sphinx autodoc fails in importing the `CameraArray` class."""
        return super().prod(a, **kwargs)

    def _guess_axis(self):
        """Identify array axes whose size matches ``nElements`` and store them."""
        if self._info_axis is None:
            self._info_axis = list()
        if not self._info_axis:
            for axis, s in enumerate(self.shape):
                if s == self.nElements:
                    self._info_axis.append(axis)
                if not self._info_axis:
                    raise ValueError(
                        f"Received Array does not have any module like axis"
                        f"(An axis with {self.nElements} is expected)"
                    )


class ModuleArray(CameraArray):
    """Array with 265 elements, one per NectarCAM drawer (module)."""

    # _info_axis = list()
    _nElements = 265

    def __init__(self, a, mod_axis=None):
        """
        init function

        Parameters
        ----------

        a : np.array
            An input numpy array that contains information on the module level
        mod_axis : int or list
            Axis position (int or list of int) of the module info.
            If None, it will take all axis that has a 265 size.
        """
        super().__init__(a=a, mod_axis=mod_axis)

    def to_pixel(self):
        """
        This will create a new instanciation at each call
        """
        c = None
        for a in self.info_axis:
            c = np.repeat(self if c is None else c, 7, axis=a)

        return PixelArray(c)


class PixelArray(CameraArray):
    """Array with 1855 elements, one per NectarCAM pixel."""

    # _info_axis = list()
    _nElements = 1855

    def __init__(self, a, mod_axis=None):
        """
        init function

        Parameters
        ----------

        a : np.array
            An input numpy array that contains information on the module level
        mod_axis : int or list
            Axis position (int or list of int) of the module info.
            If None, it will take all axis that has a 265 size.
        """
        super().__init__(a=a, mod_axis=mod_axis)


class DBCameraElementInfos:
    """Time-dependent camera-element information with interpolation support.

    Stores a dataframe of per-element values indexed by time and provides
    interpolation to arbitrary query times.
    """

    def __init__(self, name, orig_df, nElements=None, t_ref=None, verbose=False):
        """Initialise with a *name*, the original DataFrame and number of elements."""
        self.name = name
        self.nElements = nElements
        self.df = self._reorganize_dataframe(orig_df)
        self.t_ref = self._define_t_ref() if t_ref is None else t_ref
        self.interpolator = self._create_interpolator()
        self.interpolation_done = False
        self._current_data = None
        self._current_time = None
        self.verbose = verbose
        # needed : ?
        self.table_datas = self.df.to_numpy().T
        self.table_times = self.df.index.to_numpy()

    def _interpolate_data(self, t):
        """Interpolate stored data at time *t* (datetime-like or astropy time)."""
        dt = (pd.to_datetime(to_datetime(t)) - self.t_ref) / np.timedelta64(1, "s")
        return self.interpolator(dt)

    def at(self, t):
        """
        Return interpolation for a given time or a list of time
        This will not store the result internally
        """
        return self._interpolate_data(t)

    @property
    def data(self):
        """
        Get the data for the current time.
        Do the interpolation if not already done
        Don't do the interpolation if not needed
        """
        if not self.interpolation_done:
            log.info(f"{__class__.__name__} {self.time = }")
            self._current_data = self._interpolate_data(self.time)
            self.interpolation_done = True
        return self._current_data

    @property
    def time(self):
        """Current query time used for interpolation."""
        return self._current_time

    @time.setter
    def time(self, t):
        """Set the current query time; resets interpolation state."""
        if self._current_time != t:
            self._current_time = t
            self.interpolation_done = False

    def set_time(self, t):
        """Set the current evaluation time and reset the interpolation cache.

        Parameters
        ----------
        t : float
            The time value.
        """
        self.time = t

    @property
    def times(self):
        """All time-stamps stored in the internal dataframe."""
        return self.df.index.to_numpy()

    @property
    def datas(self):
        """All data values stored in the internal dataframe (transposed)."""
        return self.df.to_numpy().T

    def _create_interpolator(self):
        """
        Create interpolator
        Add an option to choose interpolation method ?
        """
        x_interp = (self.df.index - self.t_ref).to_numpy() / np.timedelta64(1, "s")
        y_interp = self.df.to_numpy().T
        return interpolate.interp1d(
            x_interp, y_interp, axis=-1, assume_sorted=True, fill_value="extrapolate"
        )
        # for Akima1DInterpolator(x_interp,y_interp,axis=-1)

    def _define_t_ref(self):
        """Return the earliest time-stamp in the dataframe as the reference time."""
        return self.df.index[0]

    def _reorganize_dataframe(self, orig_df):
        """Convert a long-format dataframe into a wide-format time series.

        Parameters
        ----------
        orig_df : pd.DataFrame
            Input dataframe with columns ``drawer``, ``value`` and a
            time-stamp index.

        Returns
        -------
        pd.DataFrame
            Re-organised dataframe with one column per drawer.
        """
        # dataframe expected :
        # already filtered for the correct camera
        # expected column : drawer, value
        # index : time

        # First re-organize the data in a dictionnary so
        # that all time entry are represented
        datas = dict()
        for index, row in tqdm(
            orig_df.iterrows(), total=len(orig_df), desc=f"Read {self.name} info"
        ):
            # for index, row in orig_df.iterrows():
            t = index
            elem_id = self._get_pandas_element_id(row)
            val = row[self.name]
            if t not in datas:
                values = np.empty(self.nElements)
                values.fill(np.nan)
                datas[t] = values
            datas[t][elem_id] = val

        # Then re-orgaisze the information such that we have data as 2d numpy array
        new_times = list()
        new_values = list()
        # for k,v in tqdm(datas.items(),desc=f'Reorganize {self.name} info'):
        for k, v in datas.items():
            new_times.append(k)
            new_values.append(v)
        new_values = np.array(new_values)

        ds = dict()
        # for m in tqdm(range(self.nElements),desc=f'Assign {self.name} infos'):
        for m in range(self.nElements):
            ds[f"module_{m}"] = new_values[:, m]

        df = pd.DataFrame(ds, index=new_times)
        df.index = pd.to_datetime(df.index)
        df.interpolate("time", inplace=True)
        df.sort_index(ascending=True, inplace=True)

        return df


class DBModuleInfos(DBCameraElementInfos):
    """Time-dependent information for each of the 265 modules (drawers)."""

    def __init__(self, name, *args, **kwargs):
        """Initialise with the module name.

        Parameters
        ----------
        name : str
            Module identifier.
        """
        super().__init__(name=name, nElements=265, *args, **kwargs)

    def _get_pandas_element_id(self, row):
        """Extract the module (drawer) index from a DataFrame row."""
        return int(row["drawer"])

    def at(self, t):
        """
        Return interpolation for a given time or a list of time
        This will not store the result internally
        """
        return ModuleArray(self._interpolate_data(t))

    @property
    def data(self):
        """
        Get the data for the current time.
        Do the interpolation if not already done
        Don't do the interpolation if not needed
        """
        return ModuleArray(super().data)

    @property
    def datas(self):
        """
        Get the complete datas as they are stored in db.
        """
        return ModuleArray(super().datas)

    # def to_pixels(self,time=None):
    #     # get the interpolated value and transform the module to pixels
    #     print("NOT YET IMPLEMENTED")
    #     pass


class DBPixelInfos(DBCameraElementInfos):
    """Time-dependent information for each of the 1855 pixels."""

    def __init__(self, *args, **kwargs):
        """Initialise with 1855 pixel elements."""
        super().__init__(nElements=1855, *args, **kwargs)

    def _get_pandas_element_id(self, row):
        """Compute the pixel index from ``drawer`` (module) and ``channel`` columns."""
        val = 7 * row["drawer"] + row["channel"]
        return int(val)

    def at(self, t):
        """
        Return interpolation for a given time or a list of time
        This will not store the result internally
        """
        return PixelArray(self._interpolate_data(t))

    @property
    def data(self):
        """
        Get the data for the current time.
        Do the interpolation if not already done
        Don't do the interpolation if not needed
        """
        return PixelArray(super().data)

    @property
    def datas(self):
        """
        Get the complete datas as they are stored in db.
        """
        return PixelArray(super().datas)


class DBSimpleInfos(DBCameraElementInfos):
    """Time-dependent information for a single camera-level scalar value."""

    def __init__(self, *args, **kwargs):
        """Initialise with a single element."""
        super().__init__(nElements=1, *args, **kwargs)

    def _get_pandas_element_id(self, row):
        """Always return 0 since there is a single camera-level element."""
        return 0

    def at(self, t):
        """
        Return interpolation for a given time or a list of time
        This will not store the result internally
        """
        return self._interpolate_data(t)[0]

    @property
    def data(self):
        """
        Get the data for the current time.
        Do the interpolation if not already done
        Don't do the interpolation if not needed
        """
        return super().data[0]

    @property
    def datas(self):
        """All stored data as a 1-D array (single camera-level value)."""
        return super().datas[0]  # self.df.to_numpy().T


class SQLiteDB:
    """Interface to one or more SQLite database files containing camera monitoring data.

    Parameters
    ----------
    dbfilename : str or list of str
        Path (or list of paths) to SQLite database files.
    tmin : datetime-like, optional
        Minimum time (UTC) for selecting entries.
    tmax : datetime-like, optional
        Maximum time (UTC) for selecting entries.
    verbose : bool, optional
        Enable verbose logging.
    """

    def __init__(self, dbfilename, tmin=None, tmax=None, verbose=False, **kwargs):
        """Initialise the SQLite database interface.

        Parameters
        ----------
        dbfilename : str or list of str
            Path (or list of paths) to SQLite database files.
        tmin : datetime-like, optional
            Minimum time (UTC) for selecting entries.
        tmax : datetime-like, optional
            Maximum time (UTC) for selecting entries.
        verbose : bool, optional
            Enable verbose logging.
        """
        self.dbfilenames = set()
        self.dbs = dict()
        self.table_infos = dict()
        self.verbose = verbose
        self.tmin = tmin
        self.tmax = tmax
        self.add_db(dbfilename)

    @property
    def tmin(self):
        """Minimum time (UTC) used for selecting entries from the database."""
        return self._tmin

    @property
    def tmax(self):
        """Maximum time (UTC) used for selecting entries from the database."""
        return self._tmax

    @tmin.setter
    def tmin(self, t):
        """
        minimum time used for selecting in DB (if exist)
        As time in DB is in utc, the datetime given must be in UTC as well
        the code accept datetime and astropy.time as input
        """
        self._tmin = to_datetime(t)

    @tmax.setter
    def tmax(self, t):
        """
        maximum time used for selecting in DB (if exist)
        As time in DB is in utc, the datetime given must be in UTC as well
        the code accept datetime and astropy.time as input
        """
        self._tmax = to_datetime(t)

    def add_db(self, dbfilename):
        """
        Add one or multiple db to the class
        Accept str, list, set ,tuple of string as input
        """
        if isinstance(dbfilename, str):
            self.dbfilenames.add(dbfilename)
        elif (
            isinstance(dbfilename, list)
            or isinstance(dbfilename, set)
            or isinstance(dbfilename, tuple)
        ):
            self.dbfilenames.update(dbfilename)
        else:
            raise ValueError(
                f"dbfilename is of {type(dbfilename)} which is not understood"
            )
        self._load_infos()

    def get_table_names(self):
        """Return the set of table names loaded from the SQLite files."""
        return {t for t in self.table_infos.keys()}

    def get_available_tables(self):
        """Alias for :meth:`get_table_names`."""
        return self.get_table_names()

    @staticmethod
    def get_tables_infos_from_sqlitefile(db):
        """Return ``{table_name: {column_names}}`` for every table in *db*."""
        cursor = db.cursor()
        cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        db_tables = {t[1] for t in tables}
        infos = dict()
        for t in db_tables:
            cursor = db.execute(f"SELECT * FROM {t}")
            names = {description[0] for description in cursor.description}
            infos[t] = names
        return infos

    @staticmethod
    def _merge_dict(dict_merge, dict_input):
        """Merge *dict_input* into *dict_merge*, combining values as sets."""
        for k, v in dict_input.items():
            if k not in dict_merge:
                dict_merge[k] = set()
            dict_merge[k] |= v

    def _get_sqlfile_connection(self, dbfilename):
        """
        Open a read-only SQLite connection to *dbfilename*, or ``None`` on failure.
        """
        db = None
        try:
            sqlite3filename = f"file:{dbfilename}?mode=ro"
            db = sqlite3.connect(sqlite3filename, uri=True)
        except sqlite3.Error as error:
            db = None
            if self.verbose:
                log.error(
                    f"Can't open the sqlite for file [{dbfilename}] (error:"
                    f" {error})"
                )
        return db

    def _load_infos(self):
        """(Re)load table information from all registered SQLite files."""
        # clear infos and sql db as we'll read everything again
        self.table_infos.clear()
        self.dbs.clear()
        load_now = len(self.dbfilenames) < 500
        for dbfilename in tqdm(sorted(self.dbfilenames)):
            try:
                db = self._get_sqlfile_connection(dbfilename)
                current_table_infos = self.get_tables_infos_from_sqlitefile(db)
                self.dbs[dbfilename] = (
                    self._get_sqlfile_connection(dbfilename) if load_now else None
                )
                self._merge_dict(self.table_infos, current_table_infos)
            except Exception as err:
                if self.verbose:
                    log.error(err)

    def _aggregate_dataframes(self, df, df_list):
        """
        Concatenate *df_list* into *df*, or return the concatenation if *df* is
        ``None``.
        """
        if df is None:
            df = pd.concat(df_list)
        else:
            df = pd.concat(
                [
                    df,
                ]
                + df_list
            )
        return df

    def get_table(self, table_name):
        """
        Get a pandas dataframe from a given table name.
        The access is perhaps not safe as one can have some problem
        """

        if table_name not in self.table_infos:
            raise ValueError(
                f"[{table_name}] not in the sqlite file [{self.dbfilenames}]"
            )

        dfs = list()
        df = None
        log.info(f"Loading sqlite file for table [{table_name}]")
        for entry, (dbname, db) in enumerate(tqdm(sorted(self.dbs.items()))):
            if db is None:
                db = self._get_sqlfile_connection(dbname)
            if db is None:
                continue
            table_infos = self.get_tables_infos_from_sqlitefile(db)
            if table_name not in table_infos:
                continue
            time_name = "time"
            has_time = time_name in table_infos[table_name]
            condition = f"SELECT * FROM {table_name} "
            has_tmin = self.tmin is not None
            has_tmax = self.tmax is not None
            if has_time and (has_tmin or has_tmax):
                condition += " WHERE "
                if has_tmin:
                    condition += (
                        f"{time_name}>=datetime({self.tmin.timestamp()}, 'unixepoch') "
                    )
                if has_tmin and has_tmax:
                    condition += " AND "
                if has_tmax:
                    condition += (
                        f"{time_name}<=datetime({self.tmax.timestamp()}, 'unixepoch') "
                    )
            if has_time:
                condition += f" ORDER BY {time_name} ASC "
            if self.verbose:
                log.info(f"condition: [{condition}]")
            parse_dates = time_name if has_time else None
            d = pd.read_sql(condition, db, parse_dates=parse_dates)
            if "id" in d.columns:
                d.drop(columns=["id"], inplace=True)
            if has_time:
                d[time_name] = d[time_name].dt.tz_localize(tz="utc")
                d.set_index(time_name, inplace=True)

            if len(d) == 0:
                continue

            dfs.append(d)

        if len(dfs) > 0:
            df = self._aggregate_dataframes(df, dfs)
            dfs.clear()

        df.sort_index(ascending=True, inplace=True)

        return df

    def show_available_infos(self):
        """Log all table names and their columns known from the SQLite files."""
        for table_name, table_info in sorted(self.table_infos.items()):
            log.info(f"Table [{table_name}]:")
            for info in sorted(table_info):
                log.info(f"\t- {info}")


class DBInfos(DictInfos):
    """High-level interface to NectarCAM monitoring database information.

    Discovers SQLite files for a given run or time window, loads table
    metadata, and organises time-dependent information per camera.
    """

    def __init__(self, verbose=False, *args, **kwargs):
        """Initialise the database information interface.

        Parameters
        ----------
        verbose : bool, optional
            Enable verbose logging. Passed through to :class:`SQLiteDB`.
        """
        super().__init__(*args, **kwargs)
        self.tel = dict()
        self._current_time = None
        self.db = SQLiteDB(**kwargs)
        self.verbose = verbose
        # self.loaded_tables = list()

    @staticmethod
    def init_from_run(run, path=None, dbpath=None, verbose=False):
        """Create a :class:`DBInfos` instance from a run number.

        Parameters
        ----------
        run : int
            Run number.
        path : str or None, optional
            Path to the run data files.
        dbpath : str or None, optional
            Path to the SQLite database files.
        verbose : bool, optional
            Enable verbose logging.
        """
        # print(dir())
        # print(dir())
        # if "GetFirstLastEventTime" not in dir():
        #    raise NameError("GetFirstLastEvent is not defined.
        # The import likely failed or was not found.
        # 'init_from_run' function can't be used")

        # find the first and last event time
        begin_time, end_time = GetFirstLastEventTime(run, path=path)
        begin_time = to_datetime(begin_time)
        end_time = to_datetime(end_time)
        if path is None:
            path = GetDefaultDataPath()
        if dbpath is None:
            dbpath = path
        return DBInfos.init_from_time(begin_time, end_time, dbpath, verbose=verbose)

    @staticmethod
    def init_from_time(begin_time, end_time, dbpath=None, verbose=False):
        """Create a :class:`DBInfos` instance from a time window.

        Parameters
        ----------
        begin_time : datetime-like
            Start time (UTC).
        end_time : datetime-like
            End time (UTC).
        dbpath : str or None, optional
            Path to the SQLite database files.
        verbose : bool, optional
            Enable verbose logging.
        """
        begin_time = to_datetime(begin_time)
        end_time = to_datetime(end_time)
        t = GetDAQTimeFromTime(begin_time)
        db_files = list()
        # Let's do something very ugly so that we can use tqdm in the search
        times2search = list()
        while t <= GetDAQTimeFromTime(end_time):
            times2search.append(t)
            t = t + datetime.timedelta(seconds=86400)

        filelist_lookup = None
        if len(times2search) > 20:
            # Create a lookup to avoid searching all the time among a lot of files
            filelist_lookup = {os.path.basename(p): p for p in FindFiles("*", dbpath)}

        for t in tqdm(times2search):
            dbname = GetDBNameFromTime(t)
            if filelist_lookup is not None:
                db_file = filelist_lookup.get(dbname)
            else:
                db_file = FindFile(dbname, dbpath)
            if db_file and os.path.exists(db_file):
                db_files.append(db_file)
                if verbose:
                    log.info(f"Adding [{db_file}] to the list")
            else:
                if verbose:
                    log.error(f"Can't find file [{dbname}]")
            t = t + datetime.timedelta(seconds=86400)

        log.info(f"There is {len(db_files)} files to be read")
        db_infos = DBInfos(
            dbfilename=db_files, tmin=begin_time, tmax=end_time, verbose=verbose
        )
        return db_infos

    def get_available_tables(self):
        """Return the set of table names available in the SQLite files."""
        return self.db.get_available_tables()

    def show_available_infos(self):
        """Log all available table names and their columns."""
        self.db.show_available_infos()

    def set_time(self, t):
        """
        Set the current query time and propagate it to all loaded information objects.
        """
        t = to_datetime(t)
        super().set_time(t)
        for v in self.tel.values():
            try:
                v.set_time(t)
            except Exception:
                pass

    def show_available_tables(self):
        """Log all available table names."""
        log.info("Available tables:")
        for t in self.get_available_tables():
            log.info(f"\t{t}")

    def show_loaded_infos(self):
        """Log all loaded information (tables and columns) per camera."""
        log.info("Loaded infos:")
        for k, v in self.infos.items():
            log.info(f"\t{k}")
            log.info(f"{v = }")
            for e in v:
                log.info(f"\t\t-{v}")
        for tel, info in self.tel.items():
            log.info(f"Camera: {tel}")
            for table, elements in info.infos.items():
                log.info(f"\t{table}:")
                for elem in elements.infos.keys():
                    log.info(f"\t\t- {elem}")

    def _fix_specific_colname(self, df, table_name):
        """Apply table-specific column-name fixes (placeholder for known tables)."""
        if table_name == "monitoring_dtc_channels":
            pass

    #            df.rename(columns={'channel':'drawer'},inplace=True)
    # display(df)

    def _fixcolname(self, df):
        """
        Normalise column names: ``camera_id`` → ``camera``, ``pixel`` → ``channel``.
        """
        df.rename(columns={"camera_id": "camera", "pixel": "channel"}, inplace=True)

    # def _fixtime(self,df):
    #    df['time'] = pd.to_datetime(df['time'])

    def _get_info_flag(self, df):
        """Determine the :class:`DBInfosFlag` for *df* based on its columns."""
        has_camera = "camera" in df or "camera_id" in df
        has_drawer = "drawer" in df
        has_pixel = "channel" in df or "pixel" in df
        # warning : monitoring_dtc_channels
        # it has camera and channel but it's not pixels
        flag = DBInfosFlag(0)
        if has_camera:
            flag |= DBInfosFlag.CAMERA
        if has_drawer:
            flag |= DBInfosFlag.DRAWER
        if has_pixel and has_drawer:
            flag |= DBInfosFlag.PIXEL
        return flag

    def connect(self, *args):
        """Load one or more tables from the database into memory.

        Parameters
        ----------
        *args : str
            Table names to load. Pass ``"*"`` or nothing to load all available tables.
        """
        # args example "monitoring_channel_currents"
        tables_to_load = set()
        available_tables = self.get_available_tables()
        if (len(args) == 1 and args[0] == "*") or len(args) == 0:
            # Load everything that is available
            # db_to_load = self.db_tables
            tables_to_load = available_tables
        else:
            # Load what is asked by the user
            for a in args:
                if a in available_tables:
                    tables_to_load.add(a)
                else:
                    log.warning(f"Don't know table [{a}] --> Skip !")

        # Now for each tables, load information
        # for table_name in tqdm(tables_to_load):
        for table_name in tables_to_load:
            try:
                # for table_name in (pbar := tqdm(tables_to_load)):
                #    pbar.set_description(f"Processing {table_name}")
                log.info(f"Loading information from table [{table_name}]")
                df = self.db.get_table(table_name)
                self._fix_specific_colname(df, table_name)
                self._fixcolname(df)
                # self._fixtime(df)
                flags = self._get_info_flag(df)

                if flags & DBInfosFlag.CAMERA:
                    cameras = set(df["camera"])
                    for camera in cameras:
                        df_sel = df[df["camera"] == camera]
                        if camera not in self.tel:
                            self.tel[camera] = DBCameraInfos(tel=camera, df=df_sel)
                        if table_name not in self.tel[camera]:
                            self.tel[camera][table_name] = DBTableInfos(
                                table_name=table_name, df=df_sel
                            )

                        # Pixel level information like pixel HV
                        cols = {c for c in df_sel.columns}
                        cols2ignore = {"camera", "id", "drawer", "channel"}
                        cols = cols.difference(cols2ignore)

                        for col_name in cols:
                            try:
                                if (
                                    flags & DBInfosFlag.DRAWER
                                    and flags & DBInfosFlag.PIXEL
                                ):
                                    self.tel[camera][table_name][
                                        col_name
                                    ] = DBPixelInfos(
                                        name=col_name,
                                        orig_df=df_sel,
                                        verbose=self.verbose,
                                    )
                                elif flags & DBInfosFlag.DRAWER:
                                    # Module level information like FEB Temperature
                                    self.tel[camera][table_name][
                                        col_name
                                    ] = DBModuleInfos(
                                        name=col_name,
                                        orig_df=df_sel,
                                        verbose=self.verbose,
                                    )
                                    # print("Implement me")
                                else:
                                    # Camera level information like UCTS
                                    self.tel[camera][table_name][
                                        col_name
                                    ] = DBSimpleInfos(
                                        name=col_name,
                                        orig_df=df_sel,
                                        verbose=self.verbose,
                                    )
                                    # self.tel[camera][table_name][col_name] = DBInfos()
                                    # print("Implement me")
                            except Exception as err:
                                log.error(
                                    f"Reading column [{col_name}]"
                                    f" from table [{table_name}]"
                                    f" yield exception [{err}]"
                                )
                                log.error(
                                    "\t==> Consider specializing the function for those"
                                    "data"
                                )

                else:
                    cols = {c for c in df.columns}
                    cols2ignore = {"camera", "id", "drawer", "channel"}
                    cols = cols.difference(cols2ignore)
                    if table_name not in self.infos:
                        # better use __setitem__ ?
                        self[table_name] = DBTableInfos(table_name=table_name, df=df)
                    for col_name in cols:
                        # better use __setitem__ ?
                        self[table_name][col_name] = DBSimpleInfos(
                            name=col_name, orig_df=df
                        )
            except Exception as err:
                print(f"Problem Loading Table [{table_name}] --> Error: {err}")

    def Connect(self, *args):
        """Alias for :meth:`connect` (provided for backward compatibility)."""
        return self.connect(self, *args)


if __name__ == "__main__":
    log.error(
        "DBHandler is not meant to be run ==> You have likely done something " "wrong !"
    )
