# How to setup a cloud VM with a DB for DQM

These notes are meant to admins deploying a Bokeh service to display NectarCAM DQM results through a web app.

Two virtual machines should be created: one with read/write access to a ZODB data base (the _primary_ host), another one with a replicated, read-only data base using `zc.zrs` (the _secondary_ host).

## Primary, read/write data base VM

Create a cloud VM with the following `cloud-init` script:
```shell
#cloud-config

# Update apt database on first boot
package_update: true

# Upgrade the instance OS packages on first boot
package_upgrade: true

# Add packages
packages:
  - docker.io
  - docker-compose

# Add users
users:
  - name: jlenain
    gecos: Jean-Philippe Lenain
    primary_group: jlenain
    groups: docker
    sudo: ALL=(ALL) ALL
    shell: /bin/bash
    lock_passwd: false
    passwd: <YOUR PASSWORD HASH>
    ssh_authorized_keys:
      - ssh-rsa <YOUR SSH PUBLIC KEY HERE>


# Add Docker compose file for Plone Zeo
write_files:
  - content: |
      version: "3"
      services:
        db:
          image: plone-zeo:latest
          restart: always
          volumes:
            - data:/data
          ports:
            - "8100:8100"
            - "5000:5000"
      
      volumes:
        data: {}
    path: /home/jlenain/plone-zeo/docker-compose.yml
    owner: 'jlenain:docker'
    permissions: '0644'
    defer: true
```

Within this VM, create a configuration file for ZEO `~/zeo.conf`:
```shell
%import zc.zrs

%define INSTANCE /app
%define DATA_DIR /data
%define SECONDARY_PORT 5000

<zeo>
  address 0.0.0.0:$(ZEO_PORT)
  read-only $(ZEO_READ_ONLY)
  invalidation-queue-size $(ZEO_INVALIDATION_QUEUE_SIZE)
  pid-filename $INSTANCE/var/zeo.pid
</zeo>

<zrs>
 replicate-to $SECONDARY_PORT
 keep-alive-delay 60

 <filestorage 1>
   path $DATA_DIR/filestorage/Data.fs
   blob-dir $DATA_DIR/blobstorage
   pack-keep-old $(ZEO_PACK_KEEP_OLD)
 </filestorage>
</zrs>

<eventlog>
  level info
  <logfile>
      path $DATA_DIR/log/zeo.log
      format %(asctime)s %(message)s
    </logfile>
</eventlog>

<runner>
  program $INSTANCE/bin/runzeo
  socket-name $INSTANCE/var/zeo.zdsock
  daemon true
  forever false
  backoff-limit 10
  exit-codes 0, 2
  directory $INSTANCE
  default-to-interactive true

  # This logfile should match the one in the zeo.conf file.
  # It is used by zdctl's logtail command, zdrun/zdctl doesn't write it.
  logfile $DATA_DIR/log/zeo.log
</runner>
```

and create a Docker container for a ZEO server with `zc.zrs` repliacation enabled on port 5000, with the following Dockerfile:

```shell
FROM plone/plone-zeo:latest

COPY zeo.conf /app/etc/zeo.conf
RUN /app/bin/pip install zc.zrs --no-cache-dir
```

The container is created with:
```shell
docker build -t plone-zeo --network host .
```

and then launched with:
```shell
docker-compose up -d
```

The data base can directly be fed using the DQM starting script `start_calib.py`, which writes on the local data base deployed with ZEO on `localhost`.

## Secondary, read-only data base VM

Create another cloud VM with the following `cloud-init` script:

```shell
#cloud-config

# Update apt database on first boot
package_update: true

# Upgrade the instance OS packages on first boot
package_upgrade: true

# Add packages
packages:
  - docker.io
  - docker-compose

# Add users
users:
  - name: jlenain
    gecos: Jean-Philippe Lenain
    primary_group: jlenain
    groups: docker
    sudo: ALL=(ALL) ALL
    shell: /bin/bash
    lock_passwd: false
    passwd: <YOUR PASSWORD HASH>
    ssh_authorized_keys:
      - ssh-rsa <YOUR SSH PUBLIC KEY HERE>

# Add Docker compose file for Plone Zeo
write_files:
  - content: |
      version: "3"
      services:
        db:
          image: plone-zeo:latest
          environment:
            ZEO_READ_ONLY: "true"
          restart: always
          volumes:
            - data:/data
          ports:
            - "8100:8100"
            - "5000:5000"
      
      volumes:
        data: {}
    path: /home/jlenain/plone-zeo/docker-compose.yml
    owner: 'jlenain:docker'
    permissions: '0644'
    defer: true
```

Within this VM, create a configuration file for ZEO `~/zeo.conf`:

```shell
%import zc.zrs
%define INSTANCE /app
%define DATA_DIR /data
%define PRIMARY_HOST <PRIMARY_HOST_IP_ADDRESS>
%define PRIMARY_PORT 5000
%define SECONDARY_PORT 5000

<zeo>
  address 0.0.0.0:$(ZEO_PORT)
  read-only $(ZEO_READ_ONLY)
  invalidation-queue-size $(ZEO_INVALIDATION_QUEUE_SIZE)
  pid-filename $INSTANCE/var/zeo.pid
</zeo>

<zrs>
 replicate-from $PRIMARY_HOST:$PRIMARY_PORT
 replicate-to $SECONDARY_PORT
 keep-alive-delay 60

 <filestorage 1>
   path $DATA_DIR/filestorage/Data.fs
   blob-dir $DATA_DIR/blobstorage
 </filestorage>
</zrs>
```

Create a secondary, read-only Docker instance of a ZEO server with replication from the primary, with the following Dockerfile:

```shell
FROM plone/plone-zeo:latest

COPY mylocalzeoconfig /app/etc/zeo.conf
RUN /app/bin/pip install zc.zrs --no-cache-dir
# ZRS is not ccompatible witj ZODB 5.6 and above:
RUN /app/bin/pip install ZODB==5.5 --no-cache-dir
```
to be built with:
```shell
docker build -t plone-zeo --network host .
```
and instantiated with:
```shell
docker-compose up -d
```

Congrats! You now finally have two VM with ZEO servers, with:
* a primary serving as read/write data base,
* a secondary, replicated from the primary, and serving the data base read-only, to be used for the Bokeh application.

## Launching the web app

The Bokeh service is launched from the second VM (the one hosting the secondray, read-only data base) with:
```shell
bokeh serve --show bokeh_app.py
```
and served on `localhost` on port 5006.

##  TODO

* Open the access of the web app worldwide, protected with NectarCAM user/password.
* Open write access to the DB for worker nodes on DIRAC, presumably using temporary, volatile tokens.
