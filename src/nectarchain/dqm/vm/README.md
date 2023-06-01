# How to setup a cloud VM with a DB for DQM

* Create a cloud VM
* Use the `docker-compose-yml` file to create Docker containers with a ZEO server and a ZODB database within this VM, with:
```shell
docker-compose up -d
```
* Launch Bokeh server with:
```shell
bokeh serve --show bokeh_app.py
```
TO BE TESTED!

# TODO
* Add `cloud-init` configuration script for the VM creation.
