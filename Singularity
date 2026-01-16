# nectarchain.sif - A singularity image for nectarchain
#
# Built from miniforge, with special conda environment containing nectarchain
#
# Typically, build this image with:
# `sudo apptainer build nectarchain.sif Singularity`
#
# Then, typically run an instance of this image with:
# `apptainer shell nectarchain.sif`

Bootstrap: docker
From: condaforge/miniforge3

%setup
    mkdir -p ${SINGULARITY_ROOTFS}/opt/cta/nectarchain

%files
    . /opt/cta/nectarchain

# From https://github.com/hpcng/singularity/issues/5075#issuecomment-594391772
%environment
    action="${0##*/}"
    if [ "$action" = "shell" ]; then
        if [ "${SINGULARITY_SHELL:-}" = "/bin/bash" ]; then
            set -- --noprofile --init-file /.singularity_bash
        elif test -z "${SINGULARITY_SHELL:-}"; then
            export SINGULARITY_SHELL=/bin/bash
            set -- --noprofile --init-file /.singularity_bash
        fi
    fi

%post
    # Install CA certificates
    apt -y update
    # Install dependencies for Qt
    apt -y install freeglut3-dev
    # cf. https://serverfault.com/a/992421
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install software-properties-common curl
    apt -y install xvfb libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 libxcb-shape0 libglib2.0-0 libgl1-mesa-dev
    curl -L https://repository.egi.eu/sw/production/cas/1/current/GPG-KEY-EUGridPMA-RPM-3 | apt-key add -
    # add-apt-repository -y 'deb https://repository.egi.eu/sw/production/cas/1/current egi-igtf core'
    # cf. https://askubuntu.com/a/952022
    echo "deb https://repository.egi.eu/sw/production/cas/1/current egi-igtf core" | tee /etc/apt/sources.list.d/egi.list
    apt -y -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update
    apt-get -y -o APT::Get::AllowUnauthenticated=true install ca-policy-egi-core || apt -y install -f

    . /opt/conda/etc/profile.d/conda.sh
    . /opt/conda/etc/profile.d/mamba.sh
    mamba update --quiet --name base conda mamba

    # Install nectarchain
    mamba env create --quiet -y --file /opt/cta/nectarchain/environment.yml --prefix /opt/conda/envs/nectarchain
    mamba activate nectarchain
    cd /opt/cta/nectarchain
    pip install -e .

    # Optionally install and configure DIRAC:
    mamba install --quiet -y -c conda-forge "dirac-grid=8"
    conda env config vars set X509_CERT_DIR=${CONDA_PREFIX}/etc/grid-security/certificates X509_VOMS_DIR=${CONDA_PREFIX}/etc/grid-security/vomsdir X509_VOMSES=${CONDA_PREFIX}/etc/grid-security/vomses
    mamba deactivate
    mamba activate nectarchain
    pip install "CTADIRAC<3"

    mamba clean --quiet -y --all

    # Since there is no proxy available at build time, manually configure the CTADIRAC client
    cat <<EOF > ${CONDA_PREFIX}/etc/dirac.cfg
DIRAC
{
  Setup = CTA
  Configuration
  {
    Servers = dips://dcta-servers03.pic.es:9135/Configuration/Server
    Servers += dips://dcta-agents03.pic.es:9135/Configuration/Server
    Servers += dips://ctadirac-01.cscs.cta-observatory.org:9135/Configuration/Server
    Servers += dips://ctadirac-02.cscs.cta-observatory.org:9135/Configuration/Server
    Servers += dips://ctadirac-03.cscs.cta-observatory.org:9135/Configuration/Server
    Servers += dips://ctadirac-04.cscs.cta-observatory.org:9135/Configuration/Server
    Servers += dips://ctadirac-05.cscs.cta-observatory.org:9135/Configuration/Server
    Servers += dips://cta-dirac.zeuthen.desy.de:9135/Configuration/Server
  }
  Security
  {
    UseServerCertificate = no
  }
}
LocalInstallation
{
  Setup = CTA
  ConfigurationServer = dips://dcta-servers03.pic.es:9135/Configuration/Server
  ConfigurationServer += dips://dcta-agents03.pic.es:9135/Configuration/Server
  ConfigurationServer += dips://ctadirac-01.cscs.cta-observatory.org:9135/Configuration/Server
  ConfigurationServer += dips://ctadirac-02.cscs.cta-observatory.org:9135/Configuration/Server
  ConfigurationServer += dips://ctadirac-03.cscs.cta-observatory.org:9135/Configuration/Server
  ConfigurationServer += dips://ctadirac-04.cscs.cta-observatory.org:9135/Configuration/Server
  ConfigurationServer += dips://ctadirac-05.cscs.cta-observatory.org:9135/Configuration/Server
  ConfigurationServer += dips://cta-dirac.zeuthen.desy.de:9135/Configuration/Server
  SkipCAChecks = True
}
EOF
    mkdir -p ${CONDA_PREFIX}/etc/grid-security/vomses
    cat <<EOF > ${CONDA_PREFIX}/etc/grid-security/vomses/ctao
"ctao" "voms-ctao.cloud.cnaf.infn.it" "15007" "/DC=org/DC=terena/DC=tcs/C=IT/L=Roma/O=Istituto Nazionale di Fisica Nucleare/CN=voms-ctao.cloud.cnaf.infn.it" "ctao" "24"
EOF
    mkdir -p ${CONDA_PREFIX}/etc/grid-security/vomsdir/ctao
    cat <<EOF > ${CONDA_PREFIX}/etc/grid-security/vomsdir/ctao/voms-ctao.cloud.cnaf.infn.it.lsc
/DC=org/DC=terena/DC=tcs/C=IT/L=Roma/O=Istituto Nazionale di Fisica Nucleare/CN=voms-ctao.cloud.cnaf.infn.it
/C=GR/O=Hellenic Academic and Research Institutions CA/CN=GEANT TLS RSA 1
EOF
    ln -s /etc/grid-security/certificates ${CONDA_PREFIX}/etc/grid-security/.

    echo "## Activate nectarchain environment" >> /.singularity_bash
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /.singularity_bash
    echo "source /opt/conda/etc/profile.d/mamba.sh" >> /.singularity_bash
    echo "conda activate nectarchain" >> /.singularity_bash

%runscript
    echo "This is a miniforge container with a nectarchain environment"
    exec /bin/bash --noprofile --init-file /.singularity_bash "$@"

%startscript
    echo "This is a miniforge container with a nectarchain environment"
    exec /bin/bash --noprofile --init-file /.singularity_bash "$@"
