# How to setup a cloud VM with a DB for DQM

These notes are meant to admins deploying a Bokeh service to display NectarCAM DQM results through a web app.

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
          image: plone/plone-zeo:latest
          restart: always
          volumes:
            - data:/data
          ports:
            - "8100:8100"
      
      volumes:
        data: {}
    path: /home/jlenain/plone-zeo/docker-compose.yml
    owner: 'jlenain:docker'
    permissions: '0644'
    defer: true

# Launch Docker compose
runcmd:
  - docker-compose -f /home/jlenain/plone-zeo/docker-compose.yml up -d
```

Alternatively, the `docker-compose-yml` configuration file in this repository can be used to create a Docker container with a ZEO server and a ZODB database within this VM, or any bare-metal server, with:
```shell
docker-compose up -d
```

The Bokeh service is launched with:
```shell
bokeh serve --show bokeh_app.py
```
and served on `localhost` on port 5006.

The data base can directly be fed using the DQM starting script `start_calib.py`, which writes on the local data base deployed with ZEO on `loclahost`.

##  TODO

* Open the access of the web app worldwide, protected with NectarCAM user/password.
* Open write access to the DB for worker nodes on DIRAC, presumably using temporary, volatile tokens.
