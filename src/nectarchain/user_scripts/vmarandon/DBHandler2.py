try:
    import datetime
    import os
    import sqlite3
    from collections.abc import Iterable
    from enum import Flag, auto

    import astropy
    import numpy as np
    import pandas as pd
    from scipy import interpolate
    from tqdm import tqdm

except ImportError as e:
    print(e)
    raise SystemExit

try:
    from Utils import GetDefaultDataPath
except ImportError:
    print("Can't find Utils module --> Create GetDefaultDataPath")

    def GetDefaultDataPath():
        return os.environ.get(
            "NECTARCAMDATA", "/Users/vm273425/Programs/NectarCAM/data"
        )


try:
    from DataUtils import GetFirstLastEventTime
except ImportError as err:
    print(
        "Can't find the GetFirstLastEventTime function --> Deactivate some functionality"
    )
    print(f"[{err}]")

try:
    from Utils import GetDAQTimeFromTime, GetDBNameFromTime
except ImportError as err:
    print("Can't find the some function --> Deactivate some functionality")
    print(f"[{err}]")


from IPython.display import HTML, display


def to_datetime(t):
    if t is None:
        t_corr = None
    elif isinstance(t, datetime.datetime):
        # print(f"t is datetime: {t}")
        if t.tzinfo is None:
            ## Assume this is actually utc
            t_corr = t.replace(tzinfo=datetime.timezone.utc)
        else:
            t_corr = t
    elif isinstance(t, astropy.time.core.Time):
        # print(f"t is astropy: {t}")
        t_corr = t.utc.to_datetime(timezone=datetime.timezone.utc)
    elif isinstance(t, Iterable):
        # print(f"t is iterable: {t}")
        t_corr = list(map(to_datetime, t))
    else:
        raise ValueError(
            f"tmin (type: {type(t)}) is not of type datetime --> Problem !"
        )

    if isinstance(t, np.ndarray):  # Convert to ndarray if this was given
        t_corr = np.array(t_corr)

    return t_corr


def FindFile(filename, path):
    for dirpath, _, filenames in os.walk(path):
        if filename in filenames:
            # print(dirpath,filename)
            return os.path.join(dirpath, filename)


class DBInfosFlag(Flag):
    CAMERA = auto()
    DRAWER = auto()
    PIXEL = auto()


class DictInfos:
    # New entry to the dict can be added by using the [] operator
    # Once entered, they can be entered by name as they become member of the class
    def __init__(self, *args, **kwargs):
        self.infos = dict()

    def __setitem__(self, key, value):
        # print(f"{self.__class__.__name__}> __setitem__: {key = } {value = }")
        self.infos[key] = value
        setattr(self, key, value)

    def __getitem__(self, key):
        # print(f"{self.__class__.__name__}> __getitem__: {key = }")
        try:
            return self.infos[key]
        except Exception:
            raise AttributeError

    def __contains__(self, key):
        # print(f"{self.__class__.__name__}> __contains__: {key = }")
        return key in self.infos

    def set_time(self, t):
        # print(f"{self.__class__.__name__}> set_time: {t}")
        for v in self.infos.values():
            try:
                v.set_time(t)
            except Exception:
                pass


##
# db.tel[0]
class DBCameraInfos(DictInfos):
    def __init__(self, tel, df=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tel = tel
        self.df = df


class DBTableInfos(DictInfos):
    def __init__(self, table_name, df=None, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.table_name = table_name
        self.df = df


class DBColumnInfos(DictInfos):
    def __init__(self, column_name, df=None, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.column_name = column_name
        self.df = df


class CameraArray(np.ndarray):
    _info_axis = None
    _nElements = -1

    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def __init__(self, a, mod_axis=None):
        """
        init function
        Parameters:
        - a : An input numpy array that contains information on the module level
        - mod_axis : Axis position (int or list of int) of the module info. If None, it will take all axis that has a 265 size.
        """
        if self._info_axis is None:
            self._info_axis = list()

        if mod_axis is None:
            self._guess_axis()
        else:
            try:
                for ax in mod_axis:
                    self.infos_axis.append(ax)
            except Exception as err:
                ## must be a single value
                self.infos_axis.append(mod_axis)

    @property
    def nElements(self):
        return self._nElements

    @property
    def info_axis(self):
        self._guess_axis()
        return self._info_axis

    def _guess_axis(self):
        if self._info_axis is None:
            self._info_axis = list()
        if not self._info_axis:
            for axis, s in enumerate(self.shape):
                if s == self.nElements:
                    self._info_axis.append(axis)
                if not self._info_axis:
                    raise ValueError(
                        f"Received Array does not have any module like axis (An axis with {self.nElements} is expected)"
                    )


class ModuleArray(CameraArray):
    # _info_axis = list()
    _nElements = 265

    def __init__(self, a, mod_axis=None):
        """
        init function
        Parameters:
        - a : An input numpy array that contains information on the module level
        - mod_axis : Axis position (int or list of int) of the module info. If None, it will take all axis that has a 265 size.
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
    # _info_axis = list()
    _nElements = 1855

    def __init__(self, a, mod_axis=None):
        """
        init function
        Parameters:
        - a : An input numpy array that contains information on the module level
        - mod_axis : Axis position (int or list of int) of the module info. If None, it will take all axis that has a 265 size.
        """
        super().__init__(a=a, mod_axis=mod_axis)

        # if self._info_axis is None:
        #     self.info_axis = list()

        # if mod_axis is None:
        #     self._guess_axis()
        # else:
        #     try:
        #         for ax in mod_axis:
        #             self.infos_axis.append(ax)
        #     except Exception as err:
        #         ## must be a single value
        #         self.infos_axis.append(mod_axis)


# class ModuleArray(np.ndarray):
#     """
#     Helper class that is a numpy array but for NectarCAM Module infos
#     """
#     _modules_axis = list()

#     def __new__(cls, a):
#         obj = np.asarray(a).view(cls)
#         return obj

#     def _guess_axis(self):
#         if not self._modules_axis:
#             for axis, s in enumerate(self.shape):
#                 if s == 265:
#                     self._modules_axis.append(axis)
#                 if not self._modules_axis:
#                     raise ValueError("Received Array does not have any module like axis (An axis with 265 is expected)")


#     def __init__(self, a, mod_axis=None):
#         """
#         init function
#         Parameters:
#         - a : An input numpy array that contains information on the module level
#         - mod_axis : Axis position (int or list of int) of the module info. If None, it will take all axis that has a 265 size.
#         """
#         #print("module array init")
#         #self._modules_axis = None

#         if mod_axis is None:
#             self._guess_axis()
#         else:
#             try:
#                 for ax in mod_axis:
#                     self._modules_axis.append(ax)
#             except Exception as err:
#                 ## must be a single value
#                 self._modules_axis.append(mod_axis)

#     @property
#     def modules_axis(self):
#         self._guess_axis()
#         return self._modules_axis


#     def to_pixel(self):
#         """
#         This will create a new instanciation at each call
#         """
#         c = None
#         for a in self.modules_axis:
#             print(f"{self.modules_axis = }")
#             print(f"{a = }")
#             c = np.repeat(self if c is None else c,7,axis=a)

#         return PixelArray(c)

# class PixelArray(np.ndarray):
#     """
#     Helper class that is a numpy array but for NectarCAM pixel infos
#     """
#     _modules_axis = list()

#     def __new__(cls, a):
#         obj = np.asarray(a).view(cls)
#         return obj

#     def __init__(self, a, pix_axis=None):
#         """
#         init function
#         Parameters:
#         - a : An input numpy array that contains information on the pixel level
#         - pix_axis : Axis position (int or list of int) of the pixel info. If None, it will take all axis that has a 1855 size.
#         """
#         self.modules_axis = list()
#         print("HERE",type(a))
#         print("THERE",a)
#         if pix_axis is None:
#             for axis, s in enumerate(a.shape):
#                 if s == 1855:
#                     self.modules_axis.append(axis)
#             if not self.modules_axis:
#                 raise ValueError("Received Array does not have any pixel like axis (An axis with 1855 is expected)")
#         else:
#             try:
#                 for ax in pix_axis:
#                     self.modules_axis.append(ax)
#             except Exception as err:
#                 ## must be a single value
#                 self.modules_axis.append(pix_axis)


class DBCameraElementInfos:
    def __init__(self, name, orig_df, nElements=None, t_ref=None, verbose=False):
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
        # print(f'{pd.to_datetime(t) = }')
        # print(f'{pd.to_datetime(to_datetime(t)) = }')
        # print(f'{self.t_ref}')
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
            print(f"DBCameraElementInfos> {self.time = }")
            self._current_data = self._interpolate_data(self.time)
            self.interpolation_done = True
        return self._current_data

    @property
    def time(self):
        return self._current_time

    @time.setter
    def time(self, t):
        # print(f"time.setter: {t = }")
        # print(f"time.setter: {self._current_time = }")
        if self._current_time != t:
            self._current_time = t
            self.interpolation_done = False

    def set_time(self, t):
        # print(f"set_time> {t = }")
        self.time = t

    # @property
    # def table_datas(self):
    #     return self.datas

    # @property
    # def table_times(self):
    #     return self.table_times

    @property
    def times(self):
        return self.df.index.to_numpy()

    @property
    def datas(self):
        # print("DBCameraElementInfos> datas")
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
        return self.df.index[0]

    def _reorganize_dataframe(self, orig_df):
        ## dataframe expected :
        ## already filtered for the correct camera
        ## expected column : drawer, value
        ## index : time

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
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, nElements=265, *args, **kwargs)

    def _get_pandas_element_id(self, row):
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
    def __init__(self, *args, **kwargs):
        super().__init__(nElements=1855, *args, **kwargs)

    def _get_pandas_element_id(self, row):
        val = 7 * row["drawer"] + row["channel"]
        # print(f'DBPixelInfos._get_pandas_element_id> {type(val) = }')
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
    def __init__(self, *args, **kwargs):
        super().__init__(nElements=1, *args, **kwargs)

    def _get_pandas_element_id(self, row):
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
        return super().datas[0]  # self.df.to_numpy().T


class SQLiteDB:
    def __init__(self, dbfilename, tmin=None, tmax=None, verbose=False, **kwargs):
        self.dbfilenames = set()
        self.dbs = dict()
        self.table_infos = dict()
        self.verbose = verbose
        self.tmin = tmin
        self.tmax = tmax
        self.add_db(dbfilename)

    @property
    def tmin(self):
        return self._tmin

    @property
    def tmax(self):
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
        # print(f'add_db: {dbfilename}')
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
        return {t for t in self.table_infos.keys()}

    def get_available_tables(self):
        return self.get_table_names()

    @staticmethod
    def get_tables_infos_from_sqlitefile(db):
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
        for k, v in dict_input.items():
            if k not in dict_merge:
                dict_merge[k] = set()
            dict_merge[k] |= v

    def _load_infos(self):
        # clear infos and sql db as we'll read everything again
        self.table_infos.clear()
        self.dbs.clear()
        for dbfilename in sorted(self.dbfilenames):
            try:
                # print(f"loading {dbfilename}")
                sqlite3filename = f"file:{dbfilename}?mode=ro"
            except sqlite3.Error as error:
                print(f"Can't open the sqlite for file [{dbfilename}]")
                continue
            if self.verbose:
                print(f"Add file [{sqlite3filename}]")
            try:
                db = sqlite3.connect(sqlite3filename, uri=True)
                self.dbs[dbfilename] = db
                current_table_infos = self.get_tables_infos_from_sqlitefile(db)
                self._merge_dict(self.table_infos, current_table_infos)
            except Exception as err:
                print(err)

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

        for dbname, db in sorted(self.dbs.items()):
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
                    condition += f"{time_name} >= datetime({self.tmin.timestamp()}, 'unixepoch') "
                if has_tmin and has_tmax:
                    condition += " AND "
                if has_tmax:
                    condition += f"{time_name} <= datetime({self.tmax.timestamp()}, 'unixepoch') "
            if has_time:
                condition += f" ORDER BY {time_name} ASC "
            if self.verbose:
                print(f"condition: [{condition}]")
            parse_dates = time_name if has_time else None
            d = pd.read_sql(condition, db, parse_dates=parse_dates)
            if "id" in d.columns:
                d.drop(columns=["id"], inplace=True)
            if has_time:
                d[time_name] = d[time_name].dt.tz_localize(tz="utc")
                d.set_index(time_name, inplace=True)
            dfs.append(d)
            # dfs.append(pd.read_sql(condition,db,parse_dates=True))
        if len(dfs) > 1:
            dfs = [d for d in dfs if len(d) > 0]
        ## now concatenate the pandas
        df = pd.concat(dfs)
        df.sort_index(ascending=True, inplace=True)
        # if 'time' in df.columns:
        #    df.sort_values(by='time',inplace=True,ignore_index=True)
        # df = pd.concat(dfs)
        # if 'time' in df.columns:
        #    df.sort_values(by='time',inplace=True,ignore_index=True)
        return df

    def show_available_infos(self):
        for table_name, table_info in sorted(self.table_infos.items()):
            print(f"Table [{table_name}]:")
            for info in sorted(table_info):
                print(f"\t- {info}")


class DBInfos(DictInfos):
    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tel = dict()
        self._current_time = None
        self.db = SQLiteDB(**kwargs)
        self.verbose = verbose
        # self.loaded_tables = list()

    @staticmethod
    def init_from_run(run, path=None, dbpath=None, verbose=False):
        # print(dir())
        # if "GetFirstLastEventTime" not in dir():
        #    raise NameError("GetFirstLastEvent is not defined. The import likely failed or was not found. 'init_from_run' function can't be used")

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
        # from datetime import datetime
        # import datetime
        begin_time = to_datetime(begin_time)
        end_time = to_datetime(end_time)
        t = GetDAQTimeFromTime(begin_time)
        db_files = list()
        while t <= GetDAQTimeFromTime(end_time):
            # print(f"GetDBNameFromTime(t): {GetDBNameFromTime(t)}")
            db_file = FindFile(GetDBNameFromTime(t), dbpath)
            if db_file:
                db_files.append(db_file)
                print(f"Adding [{db_file}] to the list")
            else:
                print(f"Can't find file [{db_file}]")
            t = t + datetime.timedelta(seconds=86400)

        # print(f'{len(db_files) = }')
        db_infos = DBInfos(
            dbfilename=db_files, tmin=begin_time, tmax=end_time, verbose=verbose
        )
        return db_infos

    def get_available_tables(self):
        return self.db.get_available_tables()

    def show_available_infos(self):
        self.db.show_available_infos()

    def set_time(self, t):
        t = to_datetime(t)
        super().set_time(t)
        for v in self.tel.values():
            try:
                v.set_time(t)
            except Exception:
                pass

    def show_available_tables(self):
        print("Available tables:")
        for t in self.get_available_tables():
            print(f"\t{t}")

    def show_loaded_infos(self):
        print("Loaded infos:")
        for k, v in self.infos.items():
            print(f"\t{k}")
            print(f"{v = }")
            for e in v:
                print(f"\t\t-{v}")
        for tel, info in self.tel.items():
            print(f"Camera: {tel}")
            for table, elements in info.infos.items():
                print(f"\t{table}:")
                for elem in elements.infos.keys():
                    print(f"\t\t- {elem}")

    # def show_available_infos(self,table=None):
    #    if table is None:
    #        table = self.get_available_tables()
    #    if type(table) is not list: table = [ table ]
    #    for t in table:
    #        cursor = self.db.execute(f'select * from {t}')
    #        names = [description[0] for description in cursor.description]
    #        print(f'{t}:')
    #        for n in names:
    #            print(f'\t{n}')

    # def show_loaded_tables(self):
    #     print("Loaded tables:")
    #     for t in self.get_loaded_tables():
    #         print(f"\t{t}")

    # def show_loaded_infos(self):
    #     self.show_available_infos(self.get_loaded_tables())

    def _fix_specific_colname(self, df, table_name):
        if table_name == "monitoring_dtc_channels":
            pass

    #            df.rename(columns={'channel':'drawer'},inplace=True)
    # display(df)

    def _fixcolname(self, df):
        df.rename(columns={"camera_id": "camera", "pixel": "channel"}, inplace=True)

    # def _fixtime(self,df):
    #    df['time'] = pd.to_datetime(df['time'])

    def _get_info_flag(self, df):
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
        ## args example "monitoring_channel_currents"
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
                    print(f"Don't know table [{a}] --> Skip !")

        ## Now for each tables, load information
        # for table_name in tqdm(tables_to_load):
        for table_name in tables_to_load:
            # for table_name in (pbar := tqdm(tables_to_load)):
            #    pbar.set_description(f"Processing {table_name}")
            print(f"Loading information from table [{table_name}]")
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
                            if flags & DBInfosFlag.DRAWER and flags & DBInfosFlag.PIXEL:
                                self.tel[camera][table_name][col_name] = DBPixelInfos(
                                    name=col_name, orig_df=df_sel, verbose=self.verbose
                                )
                            elif flags & DBInfosFlag.DRAWER:
                                # Module level information like FEB Temperature
                                self.tel[camera][table_name][col_name] = DBModuleInfos(
                                    name=col_name, orig_df=df_sel, verbose=self.verbose
                                )
                                # print("Implement me")
                            else:
                                # Camera level information like UCTS
                                self.tel[camera][table_name][col_name] = DBSimpleInfos(
                                    name=col_name, orig_df=df_sel, verbose=self.verbose
                                )
                                # self.tel[camera][table_name][col_name] = DBInfos()
                                # print("Implement me")
                        except Exception as err:
                            print(
                                f"Reading column [{col_name}] from table [{table_name}] yield exception [{err}]"
                            )
                            print(
                                f"\t==> Consider specializing the function for those data"
                            )

            else:
                cols = {c for c in df.columns}
                cols2ignore = {"camera", "id", "drawer", "channel"}
                cols = cols.difference(cols2ignore)
                if table_name not in self.infos:
                    # self.infos[table_name] = DBTableInfos(table_name=table_name)
                    # better use __setitem__ ?
                    self[table_name] = DBTableInfos(table_name=table_name, df=df)
                for col_name in cols:
                    # self.infos[table_name][col_name] = DBSimpleInfos(name=col_name,orig_df=df)
                    # better use __setitem__ ?
                    self[table_name][col_name] = DBSimpleInfos(
                        name=col_name, orig_df=df
                    )

    def Connect(self, *args):
        return self.connect(self, *args)


if __name__ == "__main__":
    print("DBHandler is not meant to be run ==> You have likely done something wrong !")
