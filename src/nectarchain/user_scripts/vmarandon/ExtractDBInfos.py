try:
    import argparse
    import sys
    from datetime import datetime, timedelta

    from DataUtils import GetFirstLastEventTime
    from DBHandler2 import DBInfos, to_datetime
    from IPython import embed
    from Utils import (
        CustomFormatter,
        GetDefaultDataPath,
        GetDefaultDBPath,
        save_simple_data,
    )

except ImportError as e:
    print(e)
    raise SystemExit


def ExtractDBInfos(arglist):
    p = argparse.ArgumentParser(
        description="Dump db infos",
        epilog="examples:\n" "\t python %(prog)s --run 123456  \n",
        formatter_class=CustomFormatter,
    )

    p.add_argument(
        "--run", dest="run", type=int, help="Run number for which to dump db infos"
    )
    p.add_argument(
        "--data-path",
        dest="dataPath",
        type=str,
        default=GetDefaultDataPath(),
        help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata",
    )
    p.add_argument(
        "--db-path",
        dest="dbPath",
        type=str,
        default=GetDefaultDBPath(),
        help="Path to the directory with db files. The program will recursively search in all directory below",
    )

    args = p.parse_args(arglist)
    if args.run is None:
        p.print_help()
        return -1

    # dbinfos = DBInfos.init_from_run(run=args.run,path=args.dataPath,dbpath=args.dbPath)

    begin_time, end_time = GetFirstLastEventTime(args.run, args.dataPath)
    begin_time = to_datetime(begin_time)
    end_time = to_datetime(end_time)
    if (end_time - begin_time).total_seconds() < 60:
        print("WARNING> Extend end_time to be at least 1 minute")
        end_time = begin_time + timedelta(seconds=60.0)
    dbinfos = DBInfos.init_from_time(
        begin_time=begin_time, end_time=end_time, dbpath=args.dbPath, verbose=False
    )
    try:
        dbinfos.show_available_infos()
        dbinfos.connect("*")
        dbinfos.db = None  # Remove the sqlite3 connection as it can't be pickled
        save_simple_data(dbinfos, f"run_{args.run}_dbinfos.pickle.lz4")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    ExtractDBInfos(sys.argv[1:])
