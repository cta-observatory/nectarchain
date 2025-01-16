import glob
import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..utils import KeepLoggingUnchanged

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers

__all__ = ["DataManagement"]

# The DIRAC magic 2 lines !
try:
    with KeepLoggingUnchanged():
        import DIRAC

        DIRAC.initialize()
except ImportError:
    log.warning("DIRAC probably not installed")
except Exception as e:
    log.warning(f"DIRAC could not be properly initialized: {e}")


class DataManagement:
    @staticmethod
    def findrun(run_number: int, search_on_GRID=True) -> Tuple[Path, List[Path]]:
        """Method to find in NECTARCAMDATA the list of ``*.fits.fz`` files
        associated to run_number.

        Parameters
        ----------
        run_number: int
            the run number

        Returns
        -------
        (PosixPath,list):
            the path list of ``*.fits.fz`` files
        """
        basepath = f"{os.environ['NECTARCAMDATA']}/runs/"
        list = glob.glob(
            basepath + "**/*" + str(run_number) + "*.fits.fz", recursive=True
        )
        list_path = [Path(chemin) for chemin in list]
        if len(list_path) == 0:
            e = FileNotFoundError(f"run {run_number} is not present in {basepath}")
            if search_on_GRID:
                log.warning(e, exc_info=True)
                log.info("will search files on GRID and fetch them")
                lfns = DataManagement.get_GRID_location(run_number)
                DataManagement.getRunFromDIRAC(lfns)
                list = glob.glob(
                    basepath + "**/*" + str(run_number) + "*.fits.fz", recursive=True
                )
                list_path = [Path(chemin) for chemin in list]
            else:
                log.error(e, exc_info=True)
                raise e

        name = list_path[0].name.split(".")
        name[2] = "*"
        name = Path(str(list_path[0].parent)) / (
            f"{name[0]}.{name[1]}.{name[2]}.{name[3]}.{name[4]}"
        )
        log.info(f"Found {len(list_path)} files matching {name}")

        # to sort list path
        _sorted = sorted([[file, int(file.suffixes[1][1:])] for file in list_path])
        list_path = [_sorted[i][0] for i in range(len(_sorted))]

        return name, list_path

    @staticmethod
    def getRunFromDIRAC(lfns: list):
        """Method to get run files from the EGI grid from input lfns.

        Parameters
        ----------
        lfns: list
            list of lfns path
        """
        with KeepLoggingUnchanged():
            from DIRAC.Interfaces.API.Dirac import Dirac

            dirac = Dirac()
            for lfn in lfns:
                if not (
                    os.path.exists(
                        f'{os.environ["NECTARCAMDATA"]}/runs/{os.path.basename(lfn)}'
                    )
                ):
                    dirac.getFile(
                        lfn=lfn,
                        destDir=f"{os.environ['NECTARCAMDATA']}/runs/",
                        printOutput=True,
                    )
                    pass

    @staticmethod
    def get_GRID_location(
        run_number: int,
        output_lfns=True,
        basepath="/vo.cta.in2p3.fr/nectarcam/",
        fromElog=False,
        username=None,
        password=None,
    ):
        """Method to get run location on GRID from Elog (work in progress!)

        Parameters
        ----------
        run_number: int
            Run number
        output_lfns: bool, optional
            If True, return lfns path of fits.gz files, else return parent directory
            of run location. Defaults to True.
        basepath: str
            The path on GRID where nectarCAM data are stored. Default to
            ``/vo.cta.in2p3.fr/nectarcam/``.
        fromElog: bool, optional
            To force to use the method which read the Elog. Default to False. To use
            the method with DIRAC API.
        username: _type_, optional
            username for Elog login. Defaults to None.
        password: _type_, optional
            password for Elog login. Defaults to None.

        Returns
        -------
        __get_GRID_location_ELog or __get_GRID_location_DIRAC
        """
        if fromElog:
            return __class__.__get_GRID_location_ELog(
                run_number=run_number,
                output_lfns=output_lfns,
                username=username,
                password=password,
            )
        else:
            return __class__.__get_GRID_location_DIRAC(
                run_number=run_number, basepath=basepath
            )

    @staticmethod
    def __get_GRID_location_DIRAC(
        run_number: int, basepath="/vo.cta.in2p3.fr/nectarcam/"
    ):
        with KeepLoggingUnchanged():
            from contextlib import redirect_stdout

            from DIRAC.DataManagementSystem.Client.FileCatalogClientCLI import (
                FileCatalogClientCLI,
            )
            from DIRAC.Interfaces.Utilities.DCommands import DCatalog

            from nectarchain.utils import StdoutRecord

            catalog = DCatalog()
            with redirect_stdout(sys.stdout):
                fccli = FileCatalogClientCLI(catalog.catalog)
                sys.stdout = StdoutRecord(keyword=f"Run{run_number}")
                fccli.do_find("-q " + basepath)
                lfns = sys.stdout.output
                sys.stdout = sys.__stdout__
        return lfns

    @staticmethod
    def __get_GRID_location_ELog(
        run_number: int, output_lfns=True, username=None, password=None
    ):
        import browser_cookie3
        import mechanize
        import requests

        url = "http://nectarcam.in2p3.fr/elog/nectarcam-data-qm/?cmd=Find"

        if not (username is None or password is None):
            log.debug("log to Elog with username and password")
            # log to Elog
            br = mechanize.Browser()
            br.open(url)
            # form = br.select_form("form1")
            for i in range(4):
                log.debug(br.form.find_control(nr=i).name)
            br.form["uname"] = username
            br.form["upassword"] = password
            br.method = "POST"
            req = br.submit()
            # html_page = req.get_data()
            cookies = br._ua_handlers["_cookies"].cookiejar
            # get data
            req = requests.get(
                f"http://nectarcam.in2p3.fr/elog/nectarcam-data-qm/"
                f"?jcmd=&mode=Raw&attach=1&printable=1&reverse=0&reverse=1&npp=20&"
                f"ma=&da=&ya=&ha=&na=&ca=&last=&mb=&db=&yb=&hb=&nb=&cb=&Author=&"
                f"Setup=&Category=&Keyword=&Subject=%23{run_number}&"
                f"ModuleCount=&subtext=",
                cookies=cookies,
            )

        else:
            # try to acces data by getting cookies from firefox and Chrome
            log.debug("try to get data with cookies from Firefox abnd Chrome")
            cookies = browser_cookie3.load()
            req = requests.get(
                f"http://nectarcam.in2p3.fr/elog/nectarcam-data-qm/?"
                f"jcmd=&mode=Raw&attach=1&printable=1&reverse=0&reverse=1"
                f"&npp=20&ma=&da=&ya=&ha=&na=&ca=&last=&mb=&db=&yb=&hb="
                f"&nb=&cb=&Author=&Setup=&Category=&Keyword="
                f"&Subject=%23{run_number}&ModuleCount=&subtext=",
                cookies=cookies,
            )

        # if "<title>ELOG Login</title>" in req.text :

        lines = req.text.split("\r\n")

        url_data = None
        for i, line in enumerate(lines):
            if "<p>" in line:
                url_data = line.split("</p>")[0].split("FC:")[1]
                log.debug(f"url_data found {url_data}")
                break

        if i == len(lines) - 1:
            e = Exception("lfns not found on GRID")
            log.error(e, exc_info=True)
            log.debug(lines)
            raise e

        if output_lfns:
            lfns = []
            try:
                # Dirac
                from DIRAC.Interfaces.API.Dirac import Dirac

                dirac = Dirac()
                loc = (
                    f"/vo.cta.in2p3.fr/nectarcam/{url_data.split('/')[-2]}/"
                    f"{url_data.split('/')[-1]}"
                )
                log.debug(f"searching in Dirac filecatalog at {loc}")
                res = dirac.listCatalogDirectory(loc, printOutput=True)

                for key in res["Value"]["Successful"][loc]["Files"].keys():
                    if str(run_number) in key and "fits.fz" in key:
                        lfns.append(key)
            except Exception as e:
                log.error(e, exc_info=True)
            return lfns
        else:
            return url_data

    @staticmethod
    def find_waveforms(run_number, max_events=None):
        return __class__.__find_computed_data(
            run_number=run_number, max_events=max_events, data_type="waveforms"
        )

    @staticmethod
    def find_charges(
        run_number, method="FullWaveformSum", str_extractor_kwargs="", max_events=None
    ):
        return __class__.__find_computed_data(
            run_number=run_number,
            max_events=max_events,
            ext=f"_{method}_{str_extractor_kwargs}.h5",
            data_type="charges",
        )

    @staticmethod
    def find_photostat(
        FF_run_number,
        ped_run_number,
        FF_method="FullWaveformSum",
        ped_method="FullWaveformSum",
        str_extractor_kwargs="",
    ):
        full_file = glob.glob(
            pathlib.Path(
                f"{os.environ.get('NECTARCAMDATA','/tmp')}/PhotoStat/"
                f"PhotoStatisticNectarCAM_FFrun{FF_run_number}_{FF_method}"
                f"_{str_extractor_kwargs}_Pedrun{ped_run_number}_{ped_method}.h5"
            ).__str__()
        )
        log.debug("for now it does not check if there are files with max events")
        if len(full_file) != 1:
            raise Exception(f"the files is {full_file}")
        return full_file

    @staticmethod
    def find_SPE_combined(
        run_number, method="FullWaveformSum", str_extractor_kwargs=""
    ):
        return __class__.find_SPE_HHV(
            run_number=run_number,
            method=method,
            str_extractor_kwargs=str_extractor_kwargs,
            keyword="FlatFieldCombined",
        )

    @staticmethod
    def find_SPE_nominal(
        run_number, method="FullWaveformSum", str_extractor_kwargs="", free_pp_n=False
    ):
        return __class__.find_SPE_HHV(
            run_number=run_number,
            method=method,
            str_extractor_kwargs=str_extractor_kwargs,
            free_pp_n=free_pp_n,
            keyword="FlatFieldSPENominal",
        )

    @staticmethod
    def find_SPE_HHV(
        run_number,
        method="FullWaveformSum",
        str_extractor_kwargs="",
        free_pp_n=False,
        **kwargs,
    ):
        keyword = kwargs.get("keyword", "FlatFieldSPEHHV")
        std_key = "" if free_pp_n else "Std"
        full_file = glob.glob(
            pathlib.Path(
                f"{os.environ.get('NECTARCAMDATA','/tmp')}/SPEfit/"
                f"{keyword}{std_key}NectarCAM_run{run_number}*_{method}"
                f"_{str_extractor_kwargs}.h5"
            ).__str__()
        )
        # need to improve the files search !!
        #       -> unstable behavior with SPE results computed
        #           with maxevents not to None
        if len(full_file) != 1:
            all_files = glob.glob(
                pathlib.Path(
                    f"{os.environ.get('NECTARCAMDATA','/tmp')}/SPEfit/"
                    f"FlatFieldSPEHHVStdNectarCAM_run{run_number}_maxevents*_"
                    f"{method}_{str_extractor_kwargs}.h5"
                ).__str__()
            )
            max_events = 0
            for i, file in enumerate(all_files):
                data = file.split("/")[-1].split(".h5")[0].split("_")
                for _data in data:
                    if "maxevents" in _data:
                        _max_events = int(_data.split("maxevents")[-1])
                        break
                if _max_events >= max_events:
                    max_events = _max_events
                    index = i
            return [all_files[index]]
        else:
            return full_file

    @staticmethod
    def __find_computed_data(
        run_number, max_events=None, ext=".h5", data_type="waveforms"
    ):
        out = glob.glob(
            pathlib.Path(
                f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/"
                f"{data_type}/*_run{run_number}{ext}"
            ).__str__()
        )
        if not (max_events is None):
            all_files = glob.glob(
                pathlib.Path(
                    f"{os.environ.get('NECTARCAMDATA','/tmp')}/runs/"
                    f"{data_type}/*_run{run_number}_maxevents*{ext}"
                ).__str__()
            )
            best_max_events = np.inf
            best_index = None
            for i, file in enumerate(all_files):
                data = file.split("/")[-1].split(".h5")[0].split("_")
                for _data in data:
                    if "maxevents" in _data:
                        _max_events = int(_data.split("maxevents")[-1])
                        break
                if _max_events >= max_events:
                    if _max_events < best_max_events:
                        best_max_events = _max_events
                        best_index = i
            if not (best_index is None):
                out = [all_files[best_index]]
        return out
