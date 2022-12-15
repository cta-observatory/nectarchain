#!/bin/env bash
#
# Time-stamp: "2022-12-15 22:13:21 jlenain"


function usage ()
{
    echo "Usage: `basename $0` -r <run number>"
}

function help ()
{
    usage
    cat <<EOF

This script launches a data quality monitoring processing of a given NectarCAM run.

OPTIONS:
     -h                       This help message.
     -r <RUN NUMBER>          NectarCAM run number to be processed.
EOF
}

# Get options
while getopts ":hr:" option; do
   case $option in
      h) # display help
          help
	  exit;;
      r) # Enter a run number
         runnb=$OPTARG;;
     \?) # Invalid option
         usage
	 exit 1;;
   esac
done
shift $((OPTIND-1))

if [ -z $runnb ]; then
    usage
    exit 1
fi

CONTAINER=nectarchain_w_2022_50.sif
OUTDIR=NectarCAM_DQM_Run${runnb}
DIRAC_OUTDIR=/vo.cta.in2p3.fr/user/j/jlenain/nectarcam/dqm

# Halim's DQM code needs to use a specific output directory:
export NECTARDIR=$PWD/$OUTDIR
[ ! -d $NECTARDIR ] && mkdir -p $NECTARDIR
mv nectarcam*.sqlite NectarCAM.Run*.fits.fz $NECTARDIR/.

LISTRUNS=""
for run in $NECTARDIR/NectarCAM.Run${runnb}.*.fits.fz; do
    LISTRUNS="$LISTRUNS $run"
done

# Create a wrapper BASH script with cleaned environment, see https://redmine.cta-observatory.org/issues/51483
WRAPPER="sing.sh"
cat > $WRAPPER <<EOF
#!/bin/env bash
echo "Cleaning environment \$CLEANED_ENV" 
[ -z "\$CLEANED_ENV" ] && exec /bin/env -i CLEANED_ENV="Done" HOME=\${HOME} SHELL=/bin/bash /bin/bash -l "\$0" "\$@" 


# Some environment variables related to python, to be passed to container:
export SINGULARITYENV_MPLCONFIGDIR=/tmp
export SINGULARITYENV_NUMBA_CACHE_DIR=/tmp
export SINGULARITYENV_NECTARDIR=$NECTARDIR

echo
echo "Running" 
# Instantiate the nectarchain Singularity image, run our DQM example run within it:
cmd="singularity exec --home $PWD $CONTAINER /opt/conda/envs/nectarchain/bin/python /opt/cta/nectarchain/nectarchain/dqm/start_calib.py $LISTRUNS" 
echo \$cmd
eval \$cmd
EOF

chmod u+x $WRAPPER
./${WRAPPER}


# Archive the output directory and push it on DIRAC before leaving the job:
tar zcf ${OUTDIR}.tar.gz ${OUTDIR}output/
dirac-dms-add-file ${DIRAC_OUTDIR}/${OUTDIR}.tar.gz ${OUTDIR}.tar.gz LPNHE-USER

# Some cleanup before leaving:
[ -d $CONTAINER ] && rm -rf $CONTAINER
[ -f $CONTAINER ] && rm -f $CONTAINER
[ -d $OUTDIR ] && rm -rf $OUTDIR
[ -d ${OUTDIR}output ] && rm -rf ${OUTDIR}output
[ -f $WRAPPER ] && rm -f $WRAPPER
