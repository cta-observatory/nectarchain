#!/bin/env bash

ORIGPWD=$PWD

CONTAINER="oras://ghcr.io/cta-observatory/nectarchain:latest"
OUTDIR=SPEFitter_$DIRACJOBID
DIRAC_OUTDIR=/vo.cta.in2p3.fr/user/j/jlenain/nectarcam/$OUTDIR

export NECTARCAMDATA=$PWD/$OUTDIR
export NECTARCHAIN_LOG=$NECTARCAMDATA
export NECTARCHAIN_TEST=$NECTARCAMDATA
export NECTARCHAIN_FIGURES=$NECTARCAMDATA
[ ! -d $NECTARCAMDATA ] && mkdir -p $NECTARCAMDATA

echo "Environment variables are:"
env

mv *.fits.fz $NECTARCAMDATA/.
cd $NECTARCAMDATA

# Create a wrapper BASH script with cleaned environment, see https://redmine.cta-observatory.org/issues/51483
WRAPPER="apptainerWrapper.sh"
cat > $WRAPPER <<EOF
#!/bin/env bash
echo "Cleaning environment \$CLEANED_ENV"
[ -z "\$CLEANED_ENV" ] && exec /bin/env -i CLEANED_ENV="Done" HOME=\${HOME} SHELL=/bin/bash /bin/bash -l "\$0" "\$@"


# Some environment variables related to python, to be passed to container:
export APPTAINERENV_MPLCONFIGDIR=/tmp
export APPTAINERENV_NUMBA_CACHE_DIR=/tmp
export APPTAINERENV_NECTARCAMDATA=$NECTARCAMDATA
export APPTAINERENV_NECTARCHAIN_LOG=$NECTARCHAIN_LOG
export APPTAINERENV_NECTARCHAIN_TEST=$NECTARCHAIN_TEST
export APPTAINERENV_NECTARCHAIN_FIGURES=$NECTARCHAIN_FIGURES

PYTHONBIN=/opt/conda/envs/nectarchain/bin/python
SCRIPTDIR=/opt/cta/nectarchain/src/nectarchain/user_scripts/ggrolleron

echo
echo "Running" 
cmd="apptainer exec --home $PWD $CONTAINER \
               \$PYTHONBIN \$SCRIPTDIR/load_wfs_compute_charge.py -s 3936 \
               --extractorMethod LocalPeakWindowSum --extractor_kwargs '{\"window_width\":16,\"window_shift\":4}'"
echo \$cmd
eval \$cmd

cmd="apptainer exec --home $PWD $CONTAINER \
               \$PYTHONBIN \$SCRIPTDIR/load_wfs_compute_charge.py -f 3937 \
               --extractorMethod FullWaveformSum"
echo \$cmd
eval \$cmd

cmd="apptainer exec --home $PWD $CONTAINER \
               \$PYTHONBIN \$SCRIPTDIR/load_wfs_compute_charge.py -f 3937 \
               --extractorMethod LocalPeakWindowSum --extractor_kwargs '{\"window_width\":16,\"window_shift\":4}'"
echo \$cmd
eval \$cmd

cmd="apptainer exec --home $PWD $CONTAINER \
               \$PYTHONBIN \$SCRIPTDIR/load_wfs_compute_charge.py -p 3938 \
               --extractorMethod LocalPeakWindowSum --extractor_kwargs '{\"window_width\":16,\"window_shift\":4}'"
echo \$cmd
eval \$cmd

cmd="apptainer exec --home $PWD $CONTAINER \
               \$PYTHONBIN \$SCRIPTDIR/load_wfs_compute_charge.py -s 3942 \
               --extractorMethod LocalPeakWindowSum --extractor_kwargs '{\"window_width\":16,\"window_shift\":4}'"
echo \$cmd
eval \$cmd

cmd="apptainer exec --home $PWD $CONTAINER \
               \$PYTHONBIN \$SCRIPTDIR/gain_PhotoStat_computation.py -p 3938 -f 3937 \
               --chargeExtractorPath LocalPeakWindowSum_4-12 --correlation --overwrite \
               --FFchargeExtractorWindowLength 16 \
               --SPE_fit_results_tag 3936 --SPE_fit_results \"$NECTARCAMDATA/../SPEfit/data/MULTI-nominal-SPEStd-3936-LocalPeakWindowSum_4-12/output_table.ecsv\""
echo \$cmd
eval \$cmd

# cmd="apptainer exec --home $PWD $CONTAINER \
#                \$PYTHONBIN \$SCRIPTDIR/gain_SPEfit_combined_computation.py -r 3936 \
#                --SPE_fit_results_tag 3942 --chargeExtractorPath LocalPeakWindowSum_4-12 \
#                --overwrite \
#                # --multiproc --chunksize 50 \
#                --VVH_fitted_results \"$NECTARCAMDATA/../SPEfit/data/MULTI-1400V-SPEStd-3942-LocalPeakWindowSum_4-12/output_table.ecsv\""
# echo \$cmd
# eval \$cmd
EOF

chmod u+x $WRAPPER
./${WRAPPER}

cd $ORIGPWD

# Archive the output directory and push it on DIRAC before leaving the job:
tar zcf ${OUTDIR}.tar.gz ${OUTDIR}/
dirac-dms-add-file ${DIRAC_OUTDIR}/${OUTDIR}.tar.gz ${OUTDIR}.tar.gz LPNHE-USER

# Some cleanup before leaving:
[ -d $OUTDIR ] && rm -rf $OUTDIR
[ -f $WRAPPER ] && rm -f $WRAPPER
