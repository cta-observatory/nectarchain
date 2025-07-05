try:
    import pickle

    import lz4.frame
except Exception as err:
    print(err)
    raise SystemExit


class PickleDataManager:
    def __init__(self, filename, method="rb"):
        # print("DataManager> __init__")
        pfile = None
        if filename is None or filename == "":
            raise ValueError(f"Given filename is empty or None --> Please check !")
        else:
            pfile = lz4.frame.open(filename, method)
        self.picklefile = pfile
        self.method = method

    ## Enter the context manager entries
    def __enter__(self):
        # print("DataManager> __enter__")
        # return self.picklefile
        return self

    def __exit__(self, type, value, traceback):
        # print(f"DataManager> __enter__: [{type = }] [{value = }] [{traceback = }]")
        self.picklefile.close()

    def __iter__(self):
        # print("DataManager> __iter__")
        return self

    def __next__(self):
        # print("DataManager> __next__")
        try:
            return pickle.load(self.picklefile)
        except EOFError:
            raise StopIteration
        except AttributeError:
            raise StopIteration

    def append(self, o):
        # print("DataManager> append")
        # Only for write mode
        if self.method != "wb":
            print("WARNING> Call append on a non writable class")
        if self.picklefile is not None:
            pickle.dump(o, self.picklefile)


class PickleDataReader(PickleDataManager):
    def __init__(self, filename):
        # print("DataReader> __init__")
        super().__init__(filename, "rb")

    # def load_all(self):
    #     while True:
    #         try:
    #             yield pickle.load( self.picklefile )
    #         except EOFError:
    #             break


class PickleDataWriter(PickleDataManager):
    def __init__(self, filename):
        # print("DataWriter> __init__")
        super().__init__(filename, "wb")
