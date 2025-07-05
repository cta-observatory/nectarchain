try:
    import argparse
    import sys

    from DataUtils import CountEventTriggers, DataReader
    from IPython import embed
    from Utils import CustomFormatter

except ImportError as e:
    print(e)
    raise SystemExit


def ShowDataContent2(arglist):
    p = argparse.ArgumentParser(
        description="Print the data content of a given run",
        epilog="examples:\n" "\t python %(prog)s --run 123456  \n",
        formatter_class=CustomFormatter,
    )

    p.add_argument("--run", dest="run", type=int, help="Run number to be converted")
    p.add_argument(
        "--data-path",
        dest="dataPath",
        type=str,
        default=None,
        help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata",
    )
    # p.add_argument("--nevents",dest="nEvents",type=int,default=-1,help="Number of event to be analysed")

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1

    event_types = CountEventTriggers(run=args.run, path=args.dataPath)
    print(f"run {args.run}:")
    # embed()
    for t, n in event_types.items():
        print(f"\t{t.name} --> {n} events")

    return 0


if __name__ == "__main__":
    retval = ShowDataContent2(sys.argv[1:])
    sys.exit(retval)
