from email.generator import Generator
import os
import glob
from DIRAC.Interfaces.API.Dirac import Dirac
from pathlib import Path
from typing import List,Tuple


import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

__all__ = ['DataManagment','ChainGenerator']

class DataManagment() :
    @staticmethod
    def findrun(run_number : int) -> Tuple[Path,List[Path]]: 
        """method to find in NECTARCAMDATA the list of *.fits.fz files associated to run_number

        Args:
            run_number (int): the run number

        Returns:
            (PosixPath,list): the path list of *fits.fz files
        """
        basepath=os.environ['NECTARCAMDATA']
        list = glob.glob(basepath+'**/*'+str(run_number)+'*.fits.fz',recursive=True)
        list_path = [Path(chemin) for chemin in list]
        name = list_path[0].name.split(".")
        name[2] = "*"
        name = Path(str(list_path[0].parent))/(f"{name[0]}.{name[1]}.{name[2]}.{name[3]}.{name[4]}")
        log.info(f"Found {len(list_path)} files matching {name}")
        return name,list_path

    @staticmethod
    def getRunFromDIRAC(lfns : list): 
        """method do get run files from GRID-EGI from input lfns

        Args:
            lfns (list): list of lfns path
        """

        dirac = Dirac()
        for lfn in lfns :
            if not(os.path.exists(f'{os.environ["NECTARCAMDATA"]}/{os.path.basename(lfn)}')):
                dirac.getFile(lfn=lfn,destDir=os.environ["NECTARCAMDATA"],printOutput=True)

class ChainGenerator():
    @staticmethod
    def chain(a : Generator ,b : Generator) :
        """generic metghod to chain 2 generators

        Args:
            a (Generator): generator to chain
            b (Generator): generator to chain

        Yields:
            Generator: a chain of a and b
        """
        yield from a
        yield from b


    @staticmethod
    def chainEventSource(list : list,max_events : int = None) : #useless with ctapipe_io_nectarcam.NectarCAMEventSource
        """recursive method to chain a list of ctapipe.io.EventSource (which may be associated to the list of *.fits.fz file of one run)

        Args:
            list (EventSource): a list of EventSource

        Returns:
            Generator: a generator which chains EventSource
        """
        if len(list) == 2 :
            return ChainGenerator.chain(list[0],list[1])
        else :
            return ChainGenerator.chain(list[0],ChainGenerator.chainEventSource(list[1:]))

