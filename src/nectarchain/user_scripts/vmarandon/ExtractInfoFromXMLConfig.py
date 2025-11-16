try: 
    import sys
    import pandas as pd
    import argparse
    from Utils import CustomFormatter,GetDefaultDataPath,FindFile
    from IPython.display import display, HTML
    import pandas as pd
    from datetime import datetime, timedelta
    import xml.etree.ElementTree as ET 

except ImportError as e:
    print(e) 
    raise SystemExit



def guessed_configfile_name(run):
    return f'NectarCAM.Run{run}.NMC.xml'

def ExtractInfoFromXMLConfig(arglist):

    p = argparse.ArgumentParser(description='Extract target HV and current trip limit from config file',
                                epilog='examples:\n'
                                '\t python %(prog)s --config NectarCAM.Run5007.NMC.xml --run 5007  \n',
                                formatter_class=CustomFormatter)

    p.add_argument("--config",dest='config',type=str,default=None,help="Name of the config file")
    p.add_argument("--run", dest='run', type=int, default=None, help="Run number")
    p.add_argument("--data-path",dest='dataPath',type=str,default=GetDefaultDataPath(),help="Path to the rawdata directory. The program will recursively search in all directory for matching rawdata")

    args = p.parse_args(arglist)

    print(args)

    if args.config is None and args.run is None:
        p.print_help()
        return -1
    elif args.config is not None:
        ## priority to the configuration give
        config_file =  FindFile(args.config,args.dataPath)
        if config_file is None:
            print(f"Can't find the file [{args.config}] in [{args.dataPath}]")
            return -1
    elif args.run is not None:
        guessed_name = guessed_configfile_name(args.run)
        config_file = FindFile(guessed_name,args.dataPath)
        if config_file is None:
            print(f"Can't find the file [{guessed_name}] in [{args.dataPath}]")
            return -1
    else:
        ## Should i be here ?
        p.print_help()
        return -1
    
    pixels = list()
    currentTrips = list()
    voltages = list()

    tree = ET.parse(config_file)
    root = tree.getroot()
    
    for m in root:
        module_id = int(m.attrib['number'])
        HVPA_infos = m.find("HVPA")
        for p in range(7):
            pixel_id = module_id*7+p
            pixels.append(pixel_id)
            currentTrips.append( HVPA_infos.attrib[f'currentTrip_{p}'] )
            voltages.append( HVPA_infos.attrib[f'voltage{p}'] )
    
    if len(pixels) != len(set(pixels)):
        print("WARNING> It looks like there is multiple time the same pixel in the XML config.... YOU SHOULD CHECK !")
    
    
    d = {'pixel' : pixels, 'target_hv' : voltages, 'current_trips' : currentTrips}
    df = pd.DataFrame(data=d)

    display(df)
    
    if args.run is not None:
        outname = f'run_{args.run}_config_infos.csv'
    else:
        outname = "config_infos.csv"
    
    df.to_csv(outname,index=False)
                    
    

if __name__ == '__main__':
    ExtractInfoFromXMLConfig(sys.argv[1:])
