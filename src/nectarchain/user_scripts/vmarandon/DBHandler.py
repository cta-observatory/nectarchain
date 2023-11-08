try:
    import sys
    import os
    import sqlite3
    import numpy as np
    import time
    import datetime
    import pandas as pd
    from enum import Enum, Flag, auto


    from enum import Enum
    from pathlib import Path

    from Utils import GetDefaultDataPath

except ImportError as e:
    print(e)
    raise SystemExit


from IPython.display import display, HTML


def FindFile(filename,path):
    for (dirpath, _ , filenames) in os.walk(path):
        if filename in filenames:
            #print(dirpath,filename)
            return os.path.join(dirpath,filename)       
  
        
# class DBInfosType(Enum):
#     Unkown = 0
#     Array = 1
#     Camera = 2
#     Drawer = 4
#     Pixel = 8

class DBInfosFlag(Flag):
    CAMERA = auto()
    DRAWER = auto()
    PIXEL = auto()

def get_info_flag(df):
    has_camera = "camera" in df or "camera_id" in df
    has_drawer = "drawer" in df
    has_pixel  = "channel" in df or "pixel" in df

    flag = DBInfosFlag(0)
    if has_camera:
        flag |= DBInfosFlag.CAMERA
    if has_drawer:
        flag |= DBInfosFlag.DRAWER
    if has_pixel:
        flag |= DBInfosFlag.PIXEL
    return flag

# def get_db_data_type(df):
#     has_camera = "camera" in df or "camera_id" in df
#     has_drawer = "drawer" in df
#     has_pixel  = "channel" in df or "pixel" in df
#     if has_camera:
#         if has_drawer and has_pixel:
#             return DBInfosType.Pixel
#         elif has_drawer:
#             return DBInfosType.Drawer
#         elif not has_drawer and has_pixel:
#             #print(f"Weird type: {has_camera = } {has_drawer = } {has_pixel = }")
#             raise ValueError("Invalid type of DB --> Check ({has_camera = } {has_drawer = } {has_pixel = })")
#         else:
#             return DBInfosType.Camera     
#     else:
#         return DBInfosType.Array
    

class DB:
    def __init__(self,dbname="",path=None,time_start=None,time_end=None):
        self.dbname = dbname #self.__getdbname__(path,dbname)
        self.db = None
        self.db_tables = set()
        
        self.infos = dict()
        self.telinfos = dict()
        self.tel = self.telinfos ## to be consistent with the r0, r1 access
        self.time_start = time_start
        self.time_end = time_end
        if self.dbname is not None:
            self.set_db(dbname,path if path is not None else GetDefaultDataPath() )
    
    def show_tables(self):
        print("Available tables:")
        for t in self.db_tables:
            print(f'\t{t}')

    def __get_selection_condition__(self,table):
        #df1b = pd.read_sql(f"SELECT * FROM 'monitoring_drawer_temperatures' WHERE time>datetime({int(test_time.timestamp())}, 'unixepoch')", con=sqlite3.connect(db_url))
        time_cond = ""
        if self.time_start is not None:
            if isinstance( self.time_start, datetime.datetime ):
                time_cond = f" WHERE time >= datetime({self.time_start.timestamp()}, 'unixepoch') "
            else:
                print(f"WARNING> {self.time_start} of type {type(self.time_start)} is of a non handled type ==> Won't be used (please correct code)")

        
        if self.time_end is not None:
            if isinstance( self.time_end, datetime.datetime ):
                link_word = " WHERE " if not time_cond else " AND "
                time_cond = f" {link_word} time <= datetime({self.time_end.timestamp()}, 'unixepoch') "
            else:
                print(f"WARNING> {self.time_end} of type {type(self.time_end)} is of a non handled type ==> Won't be used (please correct code)")

        cond = f"SELECT * FROM {table} {time_cond} ORDER BY time ASC"
        return cond

    def __getdbname__(self,path,dbname):
        ## first try path/dbname
        path = path if path is not None else ""
        fulldbname = os.path.join(path,dbname)
        print(f'{fulldbname = }')
        try:
            dbfile = Path(fulldbname)
            if dbfile.is_file():
                return fulldbname
        except NameError:
            pass

        
        fulldbname = FindFile(dbname,path)
        print(f'After FindFile: {fulldbname}')
        try:
            dbfile = Path(fulldbname)
            if dbfile.is_file():
                return fulldbname
        except NameError:
            pass
   
    def __fixcolname__(self,df):
        df.rename(columns={'camera_id': 'camera', 'pixel': 'channel'}, inplace=True)
    
    def __fixtime__(self,df):
        df['time'] = pd.to_datetime(df['time'])
        
    def set_db(self,dbname,path):
        
        print(f"dbname: {dbname} path: {path}")
        self.dbname = self.__getdbname__(path,dbname)

        if self.dbname is None:
            self.db = None
            self.db_tables = set()
            raise FileNotFoundError(f"Unknown file [{dbname}] in path [{path}]")
        
        try:
            ## retrieve the list of tables:
            self.db = sqlite3.connect(self.dbname)
            cursor = self.db.cursor()
            cursor.execute("SELECT * FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            self.db_tables = { t[1] for t in tables }

        except sqlite3.Error as error:
            #print("Error while connecting to sqlite", error)
            raise FileNotFoundError(f"Error connecting to sqlite for file [{dbname}] in path [{path}]")


    def SetDB(self,dbname,path):
        return self.set_db(dbname,path)
    
    def Connect(self,*args):

        db_to_load = set()

        if (len(args) == 1 and args[0] == "*")  or len(args) == 0:
            db_to_load = self.db_tables
        else :
             for a in args:
                if a in self.db_tables:
                    db_to_load.add( a )
                else:
                    print(f"Don't know table [{a}] --> Skip !")

        for table in db_to_load:
            #cond = f"SELECT * FROM {table} ORDER BY time ASC"
            cond = self.__get_selection_condition__(table)
            print(f"Table {table} --> [{cond}]")
            print(f"Loading Table [{table}]")
            df = pd.read_sql(cond,self.db)
            self.__fixcolname__(df)
            self.__fixtime__(df)
            flags = get_info_flag(df)
            if flags & DBInfosFlag.CAMERA:
                if flags & DBInfosFlag.DRAWER and flags & DBInfosFlag.PIXEL:
                    start_time = time.time()
                    #print(f"Table: {table} --> Pixel Level Information Table are bot handled yet !")
                    #print(f"Table: {table} --> Drawer Level Information Table are bot handled yet !")
                    cameras = set( df['camera'] )
                    drawers = list( set( df['drawer'] ) )
                    channels = list( set( df['channel'] ) )
                    min_pix = np.min(drawers)*7 + np.min(channels)
                    max_pix = np.max(drawers)*7 + np.max(channels)
                    for cam in cameras:
                        times = df[ df['camera']== cam ]['time']
                        times = [ d.to_pydatetime() for d in times]
                        times = list( set( times ) )
                        times.sort()
                        #print(type(times[0]))
                        for pix in range(min_pix,max_pix+1):
                            drw  = pix // 7
                            chan = pix % 7
                            #print(f"CT{cam} Pix: {pix} Drw: {drw} chan: {chan}")
                            df_sel = df[ (df['camera'] == cam) & (df['drawer'] == drw) & (df['channel'] == chan) ]
                            #display(df_sel)
                            if len(df_sel) > 0:
                                if cam not in self.telinfos:
                                    #self.telinfos[cam] = dict()
                                    self.telinfos[cam] = DBCameraInfos(tel=cam)
                                if table not in self.telinfos[cam]:
                                    #print(f"THERE: {table}")
                                    self.telinfos[cam][table] = DBCameraElementInfos(times)
                                #print("HERE")
                                #print(type(self.telinfos[cam][table]))
                                #print(type(self.telinfos[cam][table][pix]))
                                #self.telinfos[cam][table][pix] = DBInfos( df_sel.copy().reset_index(inplace=True) )
                                self.telinfos[cam][table][pix] = DBInfos( df_sel )
                    end_time = time.time()
                    print(f"Time to fill pixel structures for table [{table}]: {end_time-start_time}")
                elif flags & DBInfosFlag.DRAWER:
                    start_time = time.time()
                    #print(f"Table: {table} --> Drawer Level Information Table are bot handled yet !")
                    cameras = set( df['camera'] )
                    drawers = set( df['drawer'] )
                    for cam in cameras:
                        #times = df[ df['camera']== cam ]['time'] #.unique() )
                        times = df[ df['camera']== cam ]['time']
                        times = [ d.to_pydatetime() for d in times]
                        times = list( set( times ) )
                        times.sort()
                        for drw in drawers:
                            df_sel = df[ (df['camera'] == cam) & (df['drawer'] == drw) ]
                            if len(df_sel) > 0:
                                if cam not in self.telinfos:
                                    #self.telinfos[cam] = dict() # VIM : REPLACE HERE BY THE DBCAMERAINFOS CLASS
                                    self.telinfos[cam] = DBCameraInfos(tel=cam)
                                if table not in self.telinfos[cam]:
                                    self.telinfos[cam][table] = DBCameraElementInfos(times)
                                #self.telinfos[cam][table][drw] = DBInfos( df_sel.copy().reset_index(inplace=True))
                                self.telinfos[cam][table][drw] = DBInfos( df_sel )
                    end_time = time.time()
                    print(f"Time to fill drawer structure for table [{table}]: {end_time-start_time}")
                else:
                    start_time = time.time()
                    ## List of camera inside the range:
                    cameras = set( df['camera'] )
                    for cam in cameras:
                        df_sel = df[ df['camera'] == cam ]
                        if cam not in self.telinfos:
                            #self.telinfos[cam]= dict() # VIM : REPLACE HERE BY THE DBCAMERAINFOS CLASS
                            self.telinfos[cam] = DBCameraInfos(tel=cam)
                        self.telinfos[cam][table] = DBInfos(df_sel)                   
                    end_time = time.time()
                    print(f"Time to fill camera structure for table [{table}]: {end_time-start_time}")
            else:
                start_time = time.time()
                self.infos[table] = DBInfos(df)
                end_time = time.time()
                print(f"Time to fill structure for table [{table}]: {end_time-start_time}")
                setattr(self,table, self.infos[table])

    def advance(self,time):
        for name,info in self.infos.items():
            info.__advance_to__(time)
        for cam, name_info in self.telinfos.items():
            for name, info in name_info.items():
                info.__advance_to__(time)

    def rewind(self):
        for name,info in self.infos.items():
            info.rewind()
        for cam, name_info in self.telinfos.items():
            for name, info in name_info.items():
                info.rewind()


    def __str__(self) -> str:
        str_info = "DB Available Infos:\n"
        for name, info in self.infos.items():
            str_info += f"{name}:\n"
            str_info += f"{info.__str__(False)}"

        if len(self.telinfos) > 0:
            str_info += "\nDB Available Telescope Level Infos:\n"
        for cam, name_infos in self.telinfos.items():
            str_info += f"tel[{cam}]\n"
            for name, info in name_infos.items():
                str_info += f"{name}:\n"
                str_info += f"{info.__str__(False)}"
                
        return str_info


class DBCameraInfos:
    ### class of a dictionnary with a telescope id
    def __init__(self,tel):
        self.tel = tel
        self.infos = dict() 

    def __setitem__(self, key, value):
        #setattr(self, key, value)
        #print(f"DBCameraElementInfos> __setitem__: {key = } {value = }")
        self.infos[key] = value
        setattr(self,key,value)

    def __getitem__(self,key):
        return self.infos[key]
    
    def __contains__(self,key):
        return key in self.infos
    
    def items(self):
        return self.infos.items()

## add access to attribute by [] and use set attr
# setitem and getitem

class DBCameraElementInfos:
    def __init__(self,times=None):
        self.elementinfos = dict()
        self.times = times
        self.index = 0

    
    def __advance_to__(self,time):
        ## check if there is a need to update things before going for the long loop
        update = True
        

        if self.times is not None:
            current_index = self.index
            try:
                # print("samere")
                # print(type(self.times.iloc[self.index]),type(time))
                # print(type(self.times[self.index]),type(time))
                # print("time:", time)
                # print(f"time[{self.index}]:",self.times[self.index])
                # print(self.times[self.index] < time)
                # print(self.times[self.index+1] < time)
                #print(f'VIM INFO> {self.index = }' )
                while self.times[self.index] < time and self.times[self.index+1] < time:
                    self.index += 1
            except IndexError as err:
                print(f'Exception Intercepted: {err}')
                ## likely at the end of the file
                pass

            if self.index != current_index:
                update = True
            else:
                update = False
                
        if update:
            for elemid, infos in self.elementinfos.items():
                infos.__advance_to__(time)

    def advance(self,time):
        self.__advance_to__(time)

    def rewind(self):
        self.index = 0
        for drawer, infos in self.elementinfos.items():
            infos.rewind()

    def __setitem__(self, key, value):
        #setattr(self, key, value)
        #print(f"DBCameraElementInfos> __setitem__: {key = } {value = }")
        self.elementinfos[key] = value

    def __getitem__(self,key):
        return self.elementinfos[key]

    def __iter__(self):
        ## Note : Iterating this way can be dangerous... Desynchronisation might happen if sometimes a module was there or not in a run
        ## better use time and advance function
        for drawer, infos in self.elementinfos.items():
            infos.__iter__()
        return self

    def __nextentry__(self):
        for elemid, infos in self.elementinfos.items():
            infos.__nextentry__()
        if not self.time_consistent():
            print("Warning> information might be de-synchronised between drawers !")
        return self

    def time_consistent(self):
        times = list()
        for drawer, infos in self.elementinfos.items():
            times.append( infos.current_time )
        ## check if all are equals:
        iterator = iter(times)
        try:
            first = next(iterator)
        except StopIteration:
            return True
        return all(first == x for x in iterator)
        
    def __str__(self,header=True):
        str_info = ""
        if header:
            str_info += f"Available info for {len(self.elementinfos)} items:\n"
        else:
            str_info += f'For {len(self.elementinfos)} items\n'
        for k,v  in self.elementinfos.items():
            str_info += v.__str__(False)
            break
        return str_info


class DBInfos:

    def __init__(self,df=None,time=None):
        #print("DBInfos: __init__")
        self.infos = df
        self.index = -1
        self.time_index = -1
        self.current_time = None
        self.next_time = None
        self.__setdata__(df,time)
        #print("DBInfos: end __init__")

        
    def __setdata__(self,df,time=None):
        #print("__setdata__")
        self.infos = df
        if self.infos is not None:
            #print("searching index")
            self.time_index = self.infos.columns.get_loc('time')
            #print(f"index found: {self.time_index}")
            self.index = 0
            #print("before loadtime")
            self.__loadtime__()
            #print("before loaddata")
            self.__loaddata__()
            if time is not None:
                self.__advance_to__(time)

    def __iter__(self):
        #print("__iter__")
        self.index = -1
        #print(self.index)
        return self

    def __nextentry__(self):
        print("__nextentry__")
        #if self.infos is not None:
        self.index += 1
        self.__loaddata__()
        self.__loadtime__()
        #if self.infos is not None and self.index < len(self.infos):

    def __loadtime__(self):
        #print("HERE")
        try:
            self.current_time = self.infos.iloc[self.index,self.time_index]
        except KeyError as err:
            print(f"except key error : {err}")
            self.current_time = None
        #print("THERE")
        try:
            #print(f"{self.time_index = } {self.index = }")
            self.next_time = self.infos.iloc[self.index + 1,self.time_index]
        except KeyError as err:
            #print(f"except key error : {err} (Likely end of file)")
            self.next_time = None
        except IndexError as err:
            #print(f"index error : {err} (Likely end of file)")
            self.next_time = None
        except Exception as err:
            print(f"Unknown Exception catched : {err}")
            display(self.infos)
            raise err
        #print("DONE")


    def __loaddata__(self):
        cur_row = self.infos.iloc[self.index]
        for name in self.infos.columns:
            setattr(self, name, cur_row[name] )

    def rewind(self):
        if self.infos is not None:
            self.index = 0
            self.__loaddata__()
            self.__loadtime__()

    def __next__(self):
        #print("__next__")
        try:
            self.__nextentry__()
        except ValueError as err:
            #print(f"__next__: Intercepted Value error [{err}]")
            raise StopIteration
        except KeyError as err:
            #print(f"__next__: Intercepted Key error [{err}]")
            raise StopIteration
        except AttributeError as err:
            #print(f"__next__: Intercepted Attribute error [{err}]")
            raise StopIteration
        else:
            return self
   
    def __advance_to__(self,time):
        #print("__advance__")
        current_index = self.index
        try:
            #while self.infos.iloc[self.index,self.time_index] < time and self.infos.iloc[self.index+1,self.time_index] < time:
            #while self.current_time < time and self.next_time < time:
            while self.current_time < time and (self.next_time is not None and self.next_time < time) :
                self.index += 1
                self.__loadtime__()
        except KeyError as err:
            print(f"__next__: Intercepted Key error [{err}] --> eof ?")
        except TypeError as err:
            print(f'{self.current_time = } {time = } {self.next_time = } ==> {err}')
            raise TypeError(err)
        
        if current_index != self.index:
            # We changed entry --> Load the new values
            #print(f"DB Now at : {self.infos.iloc[self.index,self.time_index]}")
            self.__loaddata__()

    def __len__(self):
        return(len(self.infos))

    
    def __str__(self,header=True):
        str_info = ""
        if header:
            str_info += "Available info:\n"
        for col in self.infos.columns:
            str_info += f'\t{col}\n'
        return str_info
    

    


if __name__ == "__main__":
    print("DBHandler is not meant to be run ==> You have likely done something wrong !")