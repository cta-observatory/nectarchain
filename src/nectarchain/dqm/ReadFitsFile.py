from astropy.io import fits

# Open the fits file
with fits.open('NectarCAM_Run3645_Results.fits') as hdulist:
    # Get the first table by name
    # Print the tables
    hdulist.info()

    for i in range(1,4):
        print(hdulist[i].columns.names)

    table_data1 = hdulist['Camera'].data
    print(table_data1["CHARGE-INTEGRATION-IMAGE-ALL-AVERAGE-HIGH-GAIN"])

    table_data2 = hdulist['MWF'].data
    print(table_data2["WF-PHY-AVERAGE-CHAN-HIGH-GAIN"])


    table_data3 = hdulist['Trigger'].data
    print(table_data3["TRIGGER-STATISTICS"])