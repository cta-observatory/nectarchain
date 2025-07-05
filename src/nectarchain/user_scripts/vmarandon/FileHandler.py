try:
    import sys
    import os
    import lz4.frame
    import gzip
    import bz2
    import lzma
    import pickle
    from glob import glob

    from enum import Enum

    from ctapipe_io_nectarcam import NectarCAMEventSource
    try:
        from ctapipe_io_nectarcam import BlockNectarCAMEventSource
    except ImportError as err:
        print(err)
    from traitlets.config import Config

    from DBHandler import DB
    from Utils import FindFile, GetDAQDateFromTime, FindFiles, GetBlockListFromURL, GetRunURL, GetDefaultDataPath, GetDBNameFromTime
    from tqdm import tqdm

except ImportError as e:
    print(e)
    raise SystemExit

class OpenMode(Enum):
    READ = 1
    WRITE = 2

class TimeInfo:
    def __init__(self,data,time,nexttime=None):
        self.data = data
        self.time = time
        self.nexttime = nexttime

class SingleFile:
    def __init__(self,mode,name,run,path="",data_block=-1):
        self._mode = mode
        self._name = name
        self._file = None
        self._data = None
        self._indexLength = 4
        self._end_of_file_reached = False
        
        # Info for read mode
        #print(f"{data_block = }")
        if self._mode == OpenMode.READ:
            self._fileList = self.__get_readfilelist(name,run,path,data_block)
            if len(self._fileList) == 0:
                # if _fileList is empty, then I should search if a file exist in the path given
                # as it could be the root path
                #print(f"{path = }")
                #print(f"{name = }")
                #print(f"{run = }")
                fullpath = FindFile( self.__create_filename(name,run,"",index=0) ,path)
                dirpath = os.path.dirname(fullpath)
                self._fileList = self.__get_readfilelist(name,run,dirpath,data_block)

            self._currentFileIndex = 0
        else:
            index = data_block if data_block != -1 else 0
            self._fileList = [self.__create_filename(name,run,path,index=index)]
            self._currentFileIndex = 0

        #print(f"path: {path}")
        self.__open_file__()

        #self._path = self.__create_filename(name,run,path)
        #self.__open_file__() #will open the file
        setattr(self, self._name, None)

    def get_directory(self):
        if len(self._fileList)>0 :
            dirname = os.path.dirname(self._fileList[self._currentFileIndex])
        else:
            dirname = ""
        return dirname
    
    def __get_readfilelist(self,name,run,path,data_block):
        
        if data_block >=0:
            str_index = f'{data_block}'.zfill(self._indexLength)
        else:
            str_index = '*'

        filename = f'run{run}.{name}.{str_index}.pickle.lz4'
        #filename = f'run{run}.{name}.{str_index}.pickle.gz'
        full_path = os.path.join(path,filename)
        file_list = glob(full_path)
        file_list.sort()
        return file_list
        

    def __create_filename(self,name,run,path,index):
        str_index = f'{index}'.zfill(self._indexLength)
        filename = f'run{run}.{name}.{str_index}.pickle.lz4'
        #filename = f'run{run}.{name}.{str_index}.pickle.gz'
        full_path = os.path.join(path,filename)
        return full_path
    
    def __open_file__(self):
        open_mode = 'wb' if self._mode == OpenMode.WRITE else 'rb'
        self._file = lz4.frame.open( self._fileList[self._currentFileIndex], open_mode, compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC, block_size=lz4.frame.BLOCKSIZE_MAX4MB)
        #self._file = gzip.open( self._fileList[self._currentFileIndex], open_mode )
        #self._file = mgzip.open( self._fileList[self._currentFileIndex], open_mode, thread=4 )

    def __close_file__(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        #print("SingleFile> __enter__")
        #ttysetattr etc goes here before opening and returning the file object
        #self.__open_file__()
        return self
    
    def __exit__(self, type, value, traceback):
        #print("SingleFile> __exit__")
        #Exception handling here
        self._file.close()

    def __iter__(self):
        #print("SingleFile> __iter__")
        if self._mode == OpenMode.READ:
            return self
        else:
            raise 
        ## only in read mode. Exception should be raised otherwise
        return self

    def dump(self,data,time=None,nexttime=None):
        if self._mode == OpenMode.WRITE:
            pickle.dump( TimeInfo(data=data,time=time,nexttime=nexttime) ,self._file)
        else:
            print("SingleFile> You are not in write mode, you can't dump")
            ## Raise exception instead
    
    def rewind(self):
        ## close and re-open ?
        if self._mode == OpenMode.READ:
            self.__close_file__()
            self._currentFileIndex = 0
            self.__open_file__()
        else:
            ## Throw exception instead
            print("SingleFile> Can't rewind if not in READ Mode")

    def get_entry(self,entry_id):
        print("SingleFile> get_entry function not yet implemented")
        pass
    
    def __next__(self):
        #print("SingleFile> __next__")
        ## Only for the read mode
        if self._mode == OpenMode.READ:
            try:
                #print("HERE")
                self._data = pickle.load(self._file) ## catch the exception that should be raised here and return it as the end of __next__
                setattr(self, self._name, self._data)
                return self
            except EOFError as err:
                self.__close_file__()
                self._currentFileIndex += 1
                if self._currentFileIndex < len( self._fileList ):
                    #print(f"Previous file over --> Opening next file [{self._fileList[self._currentFileIndex]}]")
                    self.__open_file__()
                    ## then load the first event !
                    self.__next__()
                else:
                    raise StopIteration
        else:
            print("You should not be here !")
            ## Raise exception

    def __advance_to__(self,time):
        if self._mode == OpenMode.READ:
            try:
                if not self._end_of_file_reached:
                    #print("start of __advance_to__")
                    if self._data is None:
                        self.__next__()
                    ## If no event loaded, self._data is None
                    ## if we go too far, the __next__() call will close the file... I don't know what is in memory in this case..... 
                    #print(f'cond: {self._data.time < time} {self._data.time = } {time = } {self._data.nexttime = }')
                    nexttime_comparison = self._data.nexttime > time if self._data.nexttime is not None else True
                    
                    while self._data.time < time and nexttime_comparison:
                        #print("FROM SYNC")
                        self.__next__()

            except StopIteration as err:
                #print("==> End of file reached")
                self._end_of_file_reached = True
            #print("end of __advance_to__")
        else:
            print("the function __advance_to__ is only for the READ mode of FileHandler")


class DataDumper:
    def __init__(self,run,path="",data_block=-1):
        self.dictinfos = dict()
        self.path = path
        self.runnumber = run
        self.data_block=data_block

    def __getitem__(self,key):
        if key not in self.dictinfos:
            self.dictinfos[key] = SingleFile(mode=OpenMode.WRITE,name=key,run=self.runnumber,path=self.path,data_block=self.data_block)
        return self.dictinfos[key]

    def __enter__(self):
        #print("DataDumper> __enter__")
        #ttysetattr etc goes here before opening and returning the file object
        return self

    def __exit__(self, type, value, traceback):
        #print("DataDumper> __exit__")
        #Exception handling here
        for k, v in self.dictinfos.items():
            #print(f"Closing file [{k}] ")
            v.__exit__(type,value,traceback)


class DataReader:
    def __init__(self,run,path=None,data_block=-1):
        self._runnumber = run
        self._path = path if path is not None else GetDefaultDataPath()
        self._dictinfos = dict()
        self._data_block = data_block
        self._slaveinfos = dict()
        self._time = None
        self.db = None

    def get_data_directory(self):
        dirname = ""
        for k, v in self._dictinfos.items():
            dirname = v.get_directory()
            if dirname:
                break
        return dirname
    
    def __strip_filename(self,file,prefix,suffix):
        #name = file.split(".")[1]
        name = os.path.basename(file).split(".")[1]
        return name
        ## old style
        # if sys.version_info.major >= 3 and sys.version_info.minor >= 9:
        #     name = file.removeprefix(prefix).removesuffix(suffix)
        #     print(f"__strip_filename: {name = }")
        #     return name
        # else :
        #     print("here")
        #     f = file[ len(prefix): ]
        #     f = f[ : len(suffix) ]
        #     return f
        
    def Connect(self,*args):

        Valid = True
        if (len(args) == 1 and args[0] == "*")  or len(args) == 0:
            expected_datas = ("calibration","count","dl0","dl1","dl2","index","mon","muon","nectarcam","pointing","r0","r1","simulation","trigger")
            #prefix = os.path.join(self._path, f'run{self._runnumber}.')
            prefix = f'run{self._runnumber}.'
            suffix = ".pickle.lz4"
            #suffix = ".pickle.gz"
            #files = glob(prefix + '*' + suffix)
            #print(files)
            files = FindFiles(prefix + '*' + suffix,self._path)
            #print(f'{files = }')
            list_file = list()
            for file in files:
                #list_file.append( file.removeprefix(prefix).removesuffix(suffix) ) 
                #print(file)
                file_type =  self.__strip_filename(file,prefix,suffix)
                if file_type not in expected_datas:
                    print(f"WARNING> {file_type} will be loaded but might not be part of the RawData... Are you sure ?")
                
                list_file.append( file_type )
                
            args = list_file

            if not args:
                Valid = False

        for name in args:
            if name not in self._dictinfos and type(name) == str:
                try:
                    print(f"Opening file [{name}]",end=' ')
                    self._dictinfos[name] = SingleFile(mode=OpenMode.READ,name=name,run=self._runnumber,path=self._path,data_block=self._data_block)
                    print("--> Done !")
                    Valid &= True
                except FileNotFoundError:
                    #print(f"Can't find the file [{name}] for run [{self._runnumber}]") 
                    print("--> Failed !")
                    Valid &= False
                except TypeError:
                    #print(f"Can't find the file [{name}] for run [{self._runnumber}]") 
                    print("--> Failed !")
                    Valid &= False
        return Valid

    def ConnectSlave(self,*args):
        for name in args:
            if name not in self._dictinfos and name not in self._slaveinfos and isinstance(name,str):
                try:
                    ## try first local, then the self._path, then dirac ?
                    print(f"Opening slave file [{name}]",end=' ')
                    self._slaveinfos[name] = SingleFile(mode=OpenMode.READ,name=name,run=self._runnumber,path=self._path,data_block=-1)
                    print("--> Done !")
                except FileNotFoundError:
                    #print(f"Can't find the file [{name}] for run [{self._runnumber}]") 
                    print("--> Failed !")
        
    def ConnectDB(self,db=None,dbname=None,dbpath=None,tables=None,time_min=None,time_max=None):
        ## try dbpath, try locally, try self._path, then go to dirac if exist ? alternate to path of the data
        #def __init__(self,dbname="",path="",time_start=None,time_end=None):

        print(f"Data Path is : {self._path}")

        if db is None and dbname is None and time_min is not None:
            #dbname = "nectarcam_monitoring_db_" + GetDAQDateFromTime(time_min) + ".sqlite"
            dbname = GetDBNameFromTime(time_min)

        # if dbname is not given and dbpath is not given, try to open the one present in the directory of the data
        #GetDAQDateFromTime

        if db is None and dbname is None:
            data_directory = self.get_data_directory()
            #print(data_directory)
            db_files = FindFiles("*.sqlite",data_directory)
            #print(db_files)
            if len(db_files)>0:
                dbname = os.path.basename(db_files[0])
                dbpath = data_directory
                #print(f"DB to be loaded : {dbname}")

                if len(db_files)>1:
                    print("There is more than one DB file in [{data_directory}]... weird !")
                

        
        if db is None: # Try to use the run directory
            try:
                db = DB(dbname=dbname,path=dbpath,time_start=time_min,time_end=time_max)
            except FileNotFoundError as error:
                try:
                    print(f"Not found --> Try with other in path {self._path}")
                    db = DB(dbname=dbname,path=self._path,time_start=time_min,time_end=time_max)
                except FileNotFoundError as error:
                    print("Not found --> Try with DIRAC (need to be implemented)")
                    raise error

        if tables is None:
            db.Connect()
        elif isinstance(tables,str):
            db.Connect(tables)
        elif isinstance(tables, list) or isinstance(tables, set) or isinstance(tables, tuple):
            db.Connect(*tables)
        else:
            print(f"I don't know what to do with variable tables of type {type(tables)}")

        self.db = db


    def Contains(self,name):
        return name in self._dictinfos or name in self._slaveinfos


    def __iter__(self):
        #print("DataReader> __iter__")
        return self
        
    def __next__(self):
        #print("DataReader> __next__")
        for k,v in self._dictinfos.items():
            v.__next__()
            #print(f"type v : {type(v)}")
            setattr(self, k, getattr(v,k).data )
            #print(f"type self.k: {type(getattr(self,k))}")
            self._time = v._data.time 
            #print(f"{v._data.time = }")

        self.__sync_slave__(self._time)
        return self
    
        ## Do a rewind at the end to be able to loop again on the files ? (and still raise the end iteration)
    def __sync_slave__(self,time):
        #print(f"sync_slave [{time}]")
        for k,v in self._slaveinfos.items():
            try:
                v.__advance_to__(self._time)
                setattr(self, k, getattr(v,k).data )
            except StopIteration:
                pass
        if self.db is not None:
            try:
                self.db.advance(self._time)
            except StopIteration:
                pass


    def __rewind_slave__(self):
        for v in self._slaveinfos.values():
            v.rewind()
        if self.db is not None:
           self.db.rewind()

    def rewind(self):
        for v in self._dictinfos.values():
            v.rewind()
        self.__rewind_slave__()

    def load_entry(self,entry):
        self.rewind()
        count = 0
        for e in self:
            count += 1
            if count > entry:
                break
        self.__rewind_slave__()
        self.__sync_slave__(self._time)
        return self


class FileHandler:
    def __init__(self,run,path=""):
        self.dictinfos = dict()
        self.runNumber = run
        self.dataPath = path
    
    def Contains(self,name):
        return name in self.dictinfos

    def Connect(self,name):
        fileName = f'run{self.runNumber}.{name}.pickle.lz4'
        #fileName = f'run{self.runNumber}.{name}.pickle.gz'
        fullPath = os.path.join(self.dataPath,fileName)
        print(f'Opening file [{fullPath}]')
        try:
            #with lzma.open( fullPath, 'rb' ) as f :
            #with bz2.BZ2File( fullPath, 'rb' ) as f :
            with lz4.frame.open( fullPath, 'rb' ) as f :
            #with gzip.open( fullPath, 'rb' ) as f :
            #with mgzip.open( fullPath, 'rb',thread=4 ) as f :
                x = pickle.load(f)
                if name not in self.dictinfos:
                    self.dictinfos[name] = x
                    setattr(self, name, x)
            return True
        except EnvironmentError: 
            print(f"Can't find file [{fullPath}]")
            return False

    def GetDatas(self):
        return self.dictinfos.values()

    def ValidDatas(self):
        it = iter( self.dictinfos.values() )
        the_len = len(next(it))
        if not all(len(l) == the_len for l in it):
            print("Not all data have same length ! ==> PROBLEM ")
            for k,v in self.dictinfos.items():
                print(f"[{k}] --> {len(v)} entries")
            return False
        else:
            return True
        
    def Print(self):
        print('Available infos:')
        for k, v in self.dictinfos.items():
            print(f'\t{k}')








## Retrieve the data with the standard way of ctapipe_io_nectarcam
def GetNectarCamEvents(run,path=None,applycalib=True,*args,**kwargs):

    if path is None:
        path = GetDefaultDataPath()
    
    if not applycalib:
        config = Config(dict(NectarCAMEventSource=dict(
            NectarCAMR0Corrections=dict(
                calibration_path=None,
                apply_flatfield=False,
                select_gain=False,
            ))))

    print(GetRunURL(run,path))
    if applycalib:
        #print("applycalib")
        try:
            reader = BlockNectarCAMEventSource(input_url=GetRunURL(run,path),*args,**kwargs)
            #print("IF")
        except Exception:
            try:
                reader = NectarCAMEventSource(input_url=GetRunURL(run,path),*args,**kwargs)
            except Exception as err:
                print(f"Can't get a valid reader --> Is the run [{run}] ok ?")
                reader = None
    else:
        #print("do no applycalib")
        try:
            reader = BlockNectarCAMEventSource(input_url=GetRunURL(run,path),config=config,*args,**kwargs)
            #print("ELSE")
        except Exception as err:
            print(err)
            try:
                reader = NectarCAMEventSource(input_url=GetRunURL(run,path),config=config,*args,**kwargs)
            except Exception as err:
                print(f"Can't get a valid reader --> Is the run [{run}] ok ?")
                reader = None
        #reader = BlockNectarCAMEventSource(input_url=GetRunURL(run,path),config=config,*args,**kwargs)
    
    #print(f"reader: {reader}")
    #print(type(reader))
    return reader









if __name__ == "__main__":
    print("FileHandler is not meant to be run ==> You have likely done something wrong !")
