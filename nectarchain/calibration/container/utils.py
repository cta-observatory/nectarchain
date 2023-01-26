from email.generator import Generator
import os
import glob
import re
import mechanize
import requests
import browser_cookie3

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
    def findrun(run_number : int,search_on_GRID = True) -> Tuple[Path,List[Path]]: 
        """method to find in NECTARCAMDATA the list of *.fits.fz files associated to run_number

        Args:
            run_number (int): the run number

        Returns:
            (PosixPath,list): the path list of *fits.fz files
        """
        basepath=os.environ['NECTARCAMDATA']
        list = glob.glob(basepath+'**/*'+str(run_number)+'*.fits.fz',recursive=True)
        list_path = [Path(chemin) for chemin in list]
        if len(list_path) == 0 : 
            e = FileNotFoundError(f"run {run_number} is not present in {basepath}")
            if search_on_GRID : 
                log.warning(e,exc_info=True)
                log.info('will search files on GRID and fetch them')
                lfns = DataManagment.get_GRID_location(run_number)
                DataManagment.getRunFromDIRAC(lfns)
                list = glob.glob(basepath+'**/*'+str(run_number)+'*.fits.fz',recursive=True)
                list_path = [Path(chemin) for chemin in list]
            else : 
                log.error(e,exc_info=True)
                raise e

            

        name = list_path[0].name.split(".")
        name[2] = "*"
        name = Path(str(list_path[0].parent))/(f"{name[0]}.{name[1]}.{name[2]}.{name[3]}.{name[4]}")
        log.info(f"Found {len(list_path)} files matching {name}")

        #to sort list path
        _sorted = sorted([[file,int(file.suffixes[1][1:])] for file in list_path])
        list_path = [_sorted[i][0] for i in range(len(_sorted))]

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


    @staticmethod
    def get_GRID_location(run_number : int,output_lfns = True, username = None,password = None) : 
        """method to get run location on GRID from Elog (work in progress!)

        Args:
            run_number (int): run number
            output_lfns (bool, optional): if True, return lfns path of fits.gz files, else return parent directory of run location. Defaults to True.
            username (_type_, optional): username for Elog login. Defaults to None.
            password (_type_, optional): password for Elog login. Defaults to None.

        Returns:
            _type_: _description_
        """

        url = "http://nectarcam.in2p3.fr/elog/nectarcam-data-qm/?cmd=Find"

        url_run = f"http://nectarcam.in2p3.fr/elog/nectarcam-data-qm/?mode=full&reverse=0&reverse=1&npp=20&subtext=%23{run_number}"

        #try to acces data by getting cookies from firefox and Chrome
        log.debug('try to get data with cookies from Firefox abnd Chrome')
        cookies = browser_cookie3.load()
        req = requests.get(f'http://nectarcam.in2p3.fr/elog/nectarcam-data-qm/?jcmd=&mode=Raw&attach=1&printable=1&reverse=0&reverse=1&npp=20&ma=&da=&ya=&ha=&na=&ca=&last=&mb=&db=&yb=&hb=&nb=&cb=&Author=&Setup=&Category=&Keyword=&Subject=&ModuleCount=&subtext=%23{run_number}',cookies = cookies)
        
        if "<title>ELOG Login</title>" in req.text : 
            log.debug('log to Elog with cookies impossible, try again with password')
            #log to Elog
            br = mechanize.Browser()
            br.open(url)

            form = br.select_form('form1')

            for i in range(4) : 
                log.debug(br.form.find_control(nr=i).name)

            br.form['uname'] = username
            br.form['upassword'] = password
            br.method = "POST"

            req = br.submit()
            #html_page = req.get_data()
            cookies = br._ua_handlers['_cookies'].cookiejar

            #get data
            req = requests.get(f'http://nectarcam.in2p3.fr/elog/nectarcam-data-qm/?jcmd=&mode=Raw&attach=1&printable=1&reverse=0&reverse=1&npp=20&ma=&da=&ya=&ha=&na=&ca=&last=&mb=&db=&yb=&hb=&nb=&cb=&Author=&Setup=&Category=&Keyword=&Subject=&ModuleCount=&subtext=%23{run_number}',cookies = cookies)


        lines = req.text.split('\r\n')

        url_data = None
        for line in lines : 
            if '<p>' in line : 
                url_data = line.split("</p>")[0].split('FC:')[1]
                break

        if output_lfns : 
            try : 
                #Dirac
                dirac = Dirac()
                loc = f"/vo.cta.in2p3.fr/nectarcam/{url_data.split('/')[-2]}/{url_data.split('/')[-1]}"
                res = dirac.listCatalogDirectory(loc, printOutput=True)

                lfns = []
                for key in res['Value']['Successful'][loc]['Files'].keys():
                    if str(run_number) in key and "fits.fz" in key : 
                        lfns.append(key)
            except Exception as e : 
                log.error(e,exc_info = True)
            return lfns
        else : 
            return url_data


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

