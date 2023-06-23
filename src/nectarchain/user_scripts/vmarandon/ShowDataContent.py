try:
    import sys
    import argparse
    from DataUtils import CountEventTypes, DataReader
    from Utils import CustomFormatter

except ImportError as e:
    print(e)
    raise SystemExit


def ShowDataContent(arglist):
    p = argparse.ArgumentParser(description='Print the data content of a given run',
                                epilog='examples:\n'
                                '\t python %(prog)s --run 123456  \n',
                                formatter_class=CustomFormatter)

    p.add_argument("--run", dest='run', type=int, help="Run number to be converted")
    p.add_argument("--data-path",dest='dataPath',type=str,default=None,help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata")

    args = p.parse_args(arglist)

    if args.run is None:
        p.print_help()
        return -1


    event_types = CountEventTypes(run=args.run,path=args.dataPath)
    print(f"run {args.run}:")
    for t, n in event_types.items():
        print(f"\t{t} --> {n} events")

    return 0


if __name__ == "__main__":
    retval = ShowDataContent(sys.argv[1:])
    sys.exit(retval)

