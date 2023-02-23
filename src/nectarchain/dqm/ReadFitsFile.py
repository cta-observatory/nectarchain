from astropy.io import fits

# Open the fits file
with fits.open('NectarCAM_Run2720_Results.fits') as hdulist:
    # Get the first table by name
    # Print the tables
    hdulist.info()

    for i in range(1,4):
        print(hdulist[i].columns.names)