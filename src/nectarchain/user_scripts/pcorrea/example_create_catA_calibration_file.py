"""
Example script to create a category A calibration file in h5 and fits formats from
NectarCAM calibration files in h5 format.
"""

from nectarchain.tools.create_catA_calibration_file import CalibrationWriterNectarCAM

# Directory for input/output
dir_calib_files = "./calibration_files/"

# Input files of calibration coefficients in h5 format
pedestal_file = dir_calib_files + "pedestal_6249.h5"
spefit_file = (
    dir_calib_files + "FlatFieldSPENominalStdNectarCAM_run3936_maxevents50000"
    "_LocalPeakWindowSum_window_shift_4_window_width_16.h5"
)
photostat_file = (
    dir_calib_files + "PhotoStatisticNectarCAM_FFrun4940_LocalPeakWindowSum_window"
    "_shift_4_window_width_16_Pedrun6249_FullWaveformSum_maxevents50000.h5"
)
flatfield_file = dir_calib_files + "2FF_4940.h5"

# Output file of Cat-A calibration file in either h5 or fits format
# NOTE: the HDF5 writer will skip Fields that are not filled, like e.g.
# `time_correction`, which is not relevant to NectarCAM. For a fits file
# `time_correction` is an array of np.nan values.
output_file = dir_calib_files + "CatACalibrationFile.fits"

# Provenance log for bookkeeping
provenance_log = dir_calib_files + "CalibrationWriterNectarCAM.provenance.log"

# Option to specify whether gain is computed with SPE fit or photostatistic method
spe = True
if spe:
    gain_file = spefit_file
else:
    gain_file = photostat_file


def main():
    exe = CalibrationWriterNectarCAM(
        pedestal_file=pedestal_file,
        gain_file=gain_file,
        flatfield_file=flatfield_file,
        output_file=output_file,
        provenance_log=provenance_log,
        overwrite=True,
        spe=spe,  # toggle to use gain estimated with SPE fit or phototstat methods
    )

    exe.run()


if __name__ == "__main__":
    main()
