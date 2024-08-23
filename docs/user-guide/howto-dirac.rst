.. _dirac:

How to use DIRAC
----------------

NectarCAM data are stored on the `EGI <https://www.egi.eu/>`_ grid using `CTA-DIRAC <https://redmine.cta-observatory.org/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide>`_.

Starting with DIRAC
===================

To start interacting with DIRAC, one needs to initialize a proxy, with the ``cta_nectarcam`` role enabled:

.. code-block:: console

   $ dirac-proxy-init -M -g cta_nectarcam

DIRAC commands are quite long and can be tedious to learn and handle. Two main families of DIRAC commands which are useful are:

* ``dirac-dms-<XXX>`` which interact with the Data Management System (i.e. data storage);
* ``dirac-wms-<XXX>`` which interact with the Workload Management System (i.e. to submit and interact with jobs on the grid).

Many more details can be found in the `CTA-DIRAC users guide <https://redmine.cta-observatory.org/projects/cta_dirac/wiki/CTA-DIRAC_Users_Guide>`_.

How to explore NectarCAM data on DIRAC
======================================

Several possibilities exist to explore NectarCAM data on the grid:

* Using the ``dirac-dms-filecatalog-cli`` command:

.. code-block:: console

   $ dirac-dms-filecatalog-cli
   Starting FileCatalog container console.
   Note that you will access several catalogs at the same time:
      DIRACFileCatalog - Master
      TSCatalog - Write
   If you want to work with a single catalog, specify it with the -f option
   FC:/> ls
   prod4_sst
   vo.cta.in2p3.fr
   FC:/> cd /vo.cta.in2p3.fr/nectarcam/2024/20240722
   FC:/vo.cta.in2p3.fr/nectarcam/2024/20240722>ls
   NectarCAM.Run5568.0000.fits.fz
   NectarCAM.Run5568.0001.fits.fz
   NectarCAM.Run5568.0002.fits.fz
   NectarCAM.Run5568.0003.fits.fz
   NectarCAM.Run5568.0004.fits.fz

* Using the `COMDIRAC <https://github.com/DIRACGrid/COMDIRAC/wiki>`_ convenient features, which provides simpler aliases to DIRAC commands, such as:

  * ``dls`` equivalent to ``ls`` on Linux;

  * ``dget``, an alias for ``dirac-dms-get-file``, to download data from DIRAC;

  * ``dsub``, an alias for ``dirac-wms-job-submit``, to submit jobs to DIRAC;

  * ``dstat`` to list your active jobs on DIRAC.

To use these commands, one should start a COMDIRAC session with:

.. code-block:: console

   $ dinit -p

NectarCAM data can then be explored using ``dls``:

.. code-block:: console

   $ dls /vo.cta.in2p3.fr/nectarcam/2024/20240722
   /vo.cta.in2p3.fr/nectarcam/2024/20240722:
   NectarCAM.Run5568.0000.fits.fz
   NectarCAM.Run5568.0001.fits.fz
   NectarCAM.Run5568.0002.fits.fz
   NectarCAM.Run5568.0003.fits.fz
   NectarCAM.Run5568.0004.fits.fz

Sometimes, it is useful to quickly find the DIRAC location of a given NectarCAM run. The command ``dfind`` comes in handy. Suppose you want to know where run `RUN_NUMBER` is, or how much slice files it contains:

.. code-block:: console

    $ dfind /vo.cta.in2p3.fr/nectarcam | grep 5326
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0000.fits.fz
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0001.fits.fz
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0002.fits.fz
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0003.fits.fz
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0004.fits.fz
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0005.fits.fz
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0006.fits.fz
    /vo.cta.in2p3.fr/nectarcam/2024/20240520/NectarCAM.Run5326.0007.fits.fz
    $ dfind /vo.cta.in2p3.fr/nectarcam | grep 5326 | grep -e "fits.fz" | wc -l
    8


The `nectarchain.data.management.DataManagement.findrun` method will
automatically localize NectarCAM data on DIRAC, given a run number, and fetch
the run files for you.

Tips
====

Web portal
^^^^^^^^^^

CTA-DIRAC has a web portal available at https://ctadirac-01.cscs.cta-observatory.org/DIRAC/ to interact with it. In particular, the `Job Monitor` application is very useful to interact with your jobs on DIRAC.

.. tip::

    If you submitted your jobs using the ``cta_nectarcam`` role, you would need to re-authenticate in the portal with this role: Click on `Settings` in the bottom left, and select `cta_nectarcam` under `Group`.

Proxy error
^^^^^^^^^^^

If from your laptop, when initializing your DIRAC proxy, you ever encounter an error such as:

.. code-block:: console

   $ dirac-proxy-init -M -g cta_nectarcam
   Your proxy is valid until Sat Aug  3 11:31:07 2024
   ; StdErr: ..........................................................
   [...]
   Certificate verification failed.
   outdated CRL found, revoking all certs till you get new CRL
   Function: certificate validation error: CRL has expired

this can be due to outdated certificates for DIRAC services stored on your computer.
One can re-synchronise them using the following command:

.. code-block:: console

   $ dirac-admin-get-CAs