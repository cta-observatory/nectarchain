# How to setup a cloud VM with a DB for DQM

* Create a cloud VM
* Use the `docker-compose-yml` file to create Docker containers with a ZEO server and a ZODB database within this VM, with:
```shell
docker-compose up -d
```

# TODO
* Add `cloud-init` configuration script for the VM creation.
