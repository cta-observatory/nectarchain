# How to setup a cloud VM with a DB for DQM

These notes are meant to admins deploying a Bokeh service to display NectarCAM DQM results through a web app.

Two virtual machines (VM) should be created:
* one with read/write access to a ZODB database (the _primary_ host);
* another one with a replicated, read-only database using `zc.zrs` (the _secondary_ host).

## Primary VM, with read/write database

Create a cloud VM using the `cloud-init` configuration file `dqm-web-app-rw_cloud-init.yml` provided in this directory. It is assumed to be based on a Debian/Ubuntu image.

You will first need to adjust the password hash (`<YOUR_PASSWORD_HASH_HERE>`) and SSH public key (`<YOUR_SSH_PUBLIC_KEY_HERE>`) to be used to create the `nectarcam` user therein.

A Docker container with a ZODB/ZEO server with `zc.zrs` replication enabled on port 5000 will automatically be created and launched within the VM.

The database can then be directly fed using the DQM starting script `start_calib.py`, which writes on the local database deployed with ZEO on `localhost`.

## Secondary, read-only database VM

Create another cloud VM using the `cloud-init` configuration file `dqm-web-app-ro_cloud-init.yml` provided in this directory. It is also assumed to be based on a Debian/Ubuntu image. 

Here also, you will first need to adjust the password hash (`<YOUR_PASSWORD_HASH_HERE>`) and SSH public key (`<YOUR_SSH_PUBLIC_KEY_HERE>`) to be used to create the `nectarcam` user therein. You will also need to provide the internal IP address of the primary host we created just before (`<PRIMARY_HOST_INTERNAL_IP_ADDRESS>`), for the secondary to know which host the ZODB database has to be replicated from.

Any attempt to write in the database on the secondary host will rightfully result in a `ZODB.POSException.ReadOnlyError` error.

## That's it!

Congrats! You now have two VMs with ZODB/ZEO servers with ZRS replication enabled, with:
* a primary serving as read/write database;
* a secondary, replicated from the primary, and serving the database read-only, to be used for the Bokeh web application.

## Launching the web app

The Bokeh service is launched from the second VM (the one hosting the secondary, read-only database) with:
```shell
bokeh serve --num-procs $(grep -c ^processor /proc/cpuinfo) \
            --allow-websocket-origin=<SECONDARY_HOST_IP_ADDRESS>:5006 \
            --allow-websocket-origin=<DNS_ALIAS_FOR_SECONDARY_HOST>:5006 \
            --show bokeh_app
```
and served on `localhost` on port 5006.

Then, make sure with your local IT team that `<SECONDARY_HOST_IP_ADDRESS>:5006/bokeh_app` is exposed to the web.

## How to back the database up ?

Backups of the DQM ZODB database can be created, from the _primary_ VM, with (cf. e.g. https://docs.docker.com/storage/volumes/#back-up-a-volume):

```shell
docker run --rm --volumes-from plone-zeo-zeo-1 -v $(pwd):/backup ubuntu tar zcf /backup/zodb.tar.gz /data
```
Such a backup can then be fed into a new production VM with:
```shell
docker run --rm --volumes-from plone-zeo-zeo-1 -v $(pwd):/backup ubuntu bash -c "cd /data && tar xvf /backup/zodb.tar.gz --strip 1"
```

##  TODO

* Open the access of the web app worldwide, protected with NectarCAM user/password.
* ~~Open write access to the DB for worker nodes on DIRAC, presumably using temporary, volatile tokens.~~ For security reasons, the DB should rather not be open in write access to the world. Instead, the revised workflow is the following:
  * Launch DQM jobs on DIRAC when transferring the data from CEA to DIRAC.
  * Instead of *pushing* the DQM output from DIRAC workers to the DQM DB, the primary VM will *pull* them from DIRAC to feed the DB.
  * So, we need an automatic mechanism, such as a cronjob, pulling DQM data from DIRAC, parsing the DQM FITS files, and writing to the DB.
