try:
    import argparse
    import datetime
    import os
    import subprocess
    import sys
    from multiprocessing import Pool

    import numpy as np
    from IPython import embed


except ImportError as e:
    print(e)
    raise SystemExit


# Let's do a multi-inheritence for the fun, since argp-arse does not
# provide it, maybe because it was too easy....
# thanks : http://stackoverflow.com/questions/18462610
class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    pass


def CreateDir(dirpath):
    print(f"Creating Directory [{dirpath}]")
    res = subprocess.run(
        ["mkdir", "-p", dirpath], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")


def ListAllDIRACFiles(dirac_path):
    res = subprocess.run(["dfind", dirac_path], stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )
    l = res.split()
    return l


def GetListOfFile(path, recursive=False, removeHidden=True):
    files = list()
    for dirpath, dirnames, filenames in os.walk(path):
        files.extend(
            [
                os.path.abspath(os.path.join(dirpath, name))
                for name in filenames
                if not (name.startswith(".") and removeHidden)
            ]
        )
        if not recursive:
            break
    return files


def GetDBFiles(files, strip_path):
    db_files = [
        os.path.basename(f) if strip_path else f for f in files if f.endswith(".sqlite")
    ]
    return db_files


def GetDiracDBFiles(files, strip_path=False):
    return GetDBFiles(files, strip_path)


def GetDestinationDBFiles(path, strip_path=False):
    files = GetListOfFile(path, recursive=True, removeHidden=True)
    return GetDBFiles(files, strip_path)


def GetDataFromDirac(dirac_path, dest_dir):
    print(f"Transfering {dirac_path} to {dest_dir}")
    ## Enter sub-directory
    ## use the dirac command
    ## cd {destdir} && dirac-dms-get-file -d {f}
    counter = 0
    max_try = 5
    succeeded = False
    while not succeeded and counter < max_try:
        res = subprocess.run(
            ["dirac-dms-get-file", "-d", dirac_path], cwd=dest_dir, capture_output=True
        )
        # res.stdout.decode('utf-8')
        succeeded = res.returncode == 0
        if not succeeded:
            print(f"Failed Transfer {res.stdout.decode('utf-8')}")
        counter += 1
    return succeeded


def TransferData(arglist):
    p = argparse.ArgumentParser(
        description="Transfer a Run from Dirac",
        epilog="examples:\n" "\t python %(prog)s  --run 4871 --nas",
        formatter_class=CustomFormatter,
    )

    p.add_argument(
        "--run", dest="runs", type=int, nargs="+", help="Run to get from dirac"
    )
    # p.add_argument("--dest-path", dest='dest_path', type=str, nargs='?', default='/Users/vm273425/Programs/NectarCAM/data/', help='Destination base path')
    p.add_argument(
        "--dest-path",
        dest="dest_path",
        type=str,
        nargs="?",
        default=os.environ.get("NECTARCAMDATA"),
        help="Destination base path",
    )
    p.add_argument(
        "--dirac-path",
        dest="dirac_path",
        type=str,
        nargs="?",
        default="/vo.cta.in2p3.fr/nectarcam/",
        help="Dirac base path",
    )
    p.add_argument(
        "--nas",
        action="store_true",
        help="Use the address of the NAS for the destination (overwrite dest_path)",
    )
    p.add_argument(
        "--db",
        action="store_true",
        help="Sync the db files that correspond to the wanted run. If no run is given, will sync all the db files",
    )
    p.add_argument(
        "--njobs",
        dest="njobs",
        type=int,
        default=5,
        help="Number of transfer jobs in parallel",
    )

    args = p.parse_args(arglist)

    if args.runs is None and not args.db:
        # ERROR_OUT("You didn't provide a command to submit, I won't submit anything !")
        p.print_help()
        return

    if args.dest_path is None:
        print(
            "ERROR> No destination path has been given, please enter one (or set a NECTARCAMDATA environment variable)"
        )
        return

    if args.nas:
        args.dest_path = "/Users/vm273425/Programs/NectarCAM/data_nas/"

    CreateDir(args.dest_path)

    files_in_dirac = ListAllDIRACFiles(args.dirac_path)
    files_in_dirac.sort()

    files_to_get = list()

    ## Get the run files that the use wants
    run_files_to_get = list()
    if args.runs is not None:
        for run in args.runs:
            run_filter = f"Run{run}"
            for f in files_in_dirac:
                if run_filter in f:
                    # print(f)
                    run_files_to_get.append(f)
    files_to_get.extend(run_files_to_get)

    if args.db and args.runs is None:
        ## If there is no run, then download the complete DB
        dest_dbfiles = set(GetDestinationDBFiles(args.dest_path, strip_path=True))
        dirac_dbfiles = GetDiracDBFiles(files_in_dirac, strip_path=False)
        # dest_dbfiles.add( 'nectarcam_monitoring_db_2023-06-23.sqlite')
        db_files_to_get = [
            d for d in dirac_dbfiles if os.path.basename(d) not in dest_dbfiles
        ]
        print(f"There is {len(db_files_to_get)} DB files to retrieve")
        files_to_get.extend(db_files_to_get)
    elif args.db:
        ## If there is a run, then download the corresponding DB
        run_directories = set(
            [os.path.basename(os.path.dirname(f)) for f in run_files_to_get]
        )
        # Expand by one the day for the directory for the DB as we can have a run that go through all night
        run_directories.update(
            [
                (
                    datetime.datetime.strptime(r, "%Y%m%d")
                    + datetime.timedelta(seconds=86400)
                ).strftime("%Y%m%d")
                for r in run_directories
            ]
        )
        print(sorted(run_directories))
        selected_dirac_files = [
            f
            for f in files_in_dirac
            if any(substring in f for substring in run_directories)
        ]
        db_files_to_get = GetDiracDBFiles(selected_dirac_files, strip_path=False)
        print(f"There is {len(db_files_to_get)} DB files to retrieve")
        files_to_get.extend(db_files_to_get)

    files_to_get.sort()

    for f in files_to_get:
        print(f)

    ## Prepare the creation of the file list
    directories = set()
    destination_paths = list()
    dirac_paths = list()

    for f in files_to_get:
        relative_path = os.path.relpath(f, args.dirac_path)
        dest_full_path = os.path.join(args.dest_path, relative_path)
        dest_dir = os.path.dirname(dest_full_path)
        directories.add(dest_dir)
        destination_paths.append(dest_dir)
        dirac_paths.append(f)

    ## Create the destination directory
    for d in directories:
        CreateDir(d)

    destination_paths = np.array(destination_paths)
    dirac_paths = np.array(dirac_paths)
    bad_mask = np.ones(len(dirac_paths)).astype(bool)
    results = np.zeros(len(dirac_paths)).astype(bool)

    # print(destination_paths)
    # print(dirac_paths)

    counter = 0
    max_try = 5

    # embed()
    ## Launch the transfer command

    while np.count_nonzero(bad_mask == True) > 0 and counter < 5:
        print(f"Iteration: {counter}")
        with Pool(np.min([np.count_nonzero(bad_mask == True), args.njobs])) as p:
            results[bad_mask] = p.starmap(
                GetDataFromDirac,
                zip(dirac_paths[bad_mask], destination_paths[bad_mask]),
            )
        p.close()
        p.join()
        bad_mask = ~results
        counter += 1
    if np.count_nonzero(bad_mask == True) > 0:
        print("WARNING> Transfer file not complete !!!")
        print("Missing transfer:")
        for f in dirac_paths[bad_mask]:
            print(f)


if __name__ == "__main__":
    TransferData(sys.argv[1:])
