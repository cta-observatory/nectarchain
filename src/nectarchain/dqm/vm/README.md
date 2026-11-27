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
export DQM_AUTH_PASSWORD='<SHARED_PASSWORD>'
export DQM_AUTH_SECRET='<generate with: python -c "import secrets; print(secrets.token_urlsafe(32))">'
bokeh serve --num-procs $(grep -c ^processor /proc/cpuinfo) \
            --allow-websocket-origin=<SECONDARY_HOST_IP_ADDRESS>:5006 \
            --allow-websocket-origin=<DNS_ALIAS_FOR_SECONDARY_HOST>:5006 \
            --enable-xsrf-cookies \
            --auth-module=/opt/cta/nectarchain/src/nectarchain/dqm/bokeh_app/auth.py \
            --show bokeh_app
```
and served on `localhost` on port 5006.

Then, make sure with your local IT team that `<SECONDARY_HOST_IP_ADDRESS>:5006/bokeh_app` is exposed to the web.

### Restricting access to the web app

The web app is protected by a login page backed by a shared NectarCAM user/password, implemented with a Bokeh authentication module (`--auth-module`, cf. `bokeh serve --help`), i.e. `bokeh_app/auth.py` in this repository.

* Access is controlled by a single shared account defined through environment variables:
  * `DQM_AUTH_USER` (default: `foo`);
  * `DQM_AUTH_PASSWORD` (required — this is the shared password to distribute to NectarCAM members);
  * `DQM_AUTH_SECRET` (required) signs the login session cookies. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"` and keep it private; changing it invalidates all active sessions.
  * `DQM_AUTH_URL_PREFIX` (default: `/bokeh_app`) — the **public** URL path, i.e. what visitors type in their browser (and what the reverse proxy exposes), e.g. `https://host/nectarcam-dqm/...`. Every URL handed to the browser — the login/logout redirects, the login form action — uses this prefix.
  * `DQM_AUTH_BACKEND_PREFIX` (default: `/bokeh_app`) — the path under which the app is served **on the Bokeh server**, i.e. what the reverse proxy forwards to port 5006. The login/logout **routes** are registered under it (`/bokeh_app/login`, `/bokeh_app/logout` by default). The two prefixes need not match: `bokeh_app/auth.py` maps between them, so a proxy that *rewrites* the path (e.g. public `/nectarcam-dqm` → backend `/bokeh_app`) works without any proxy-side change (see "Reverse proxy layouts" below).
* Unauthenticated visitors are redirected to `${DQM_AUTH_URL_PREFIX}/login`; after a successful login, a signed, `HttpOnly` session cookie is set, which expires after 12 hours. A logout endpoint is available under the same public prefix (exposed to the app as `curdoc().session_context.logout_url`; on the backend it is served at `${DQM_AUTH_BACKEND_PREFIX}/logout`).
* The Bokeh server will refuse to start if `DQM_AUTH_PASSWORD` or `DQM_AUTH_SECRET` are not defined, so an unprotected server cannot be launched by accident.
* Bokeh >= 3.1 is required for `--auth-module` support.

If the app is served behind a reverse proxy, that proxy must forward `${DQM_AUTH_BACKEND_PREFIX}` — including subpaths such as `/bokeh_app/login`, `/bokeh_app/logout` and `/bokeh_app/ws` — to the Bokeh server (port 5006). No root-level proxy rules are required; browsers only ever talk to `${DQM_AUTH_URL_PREFIX}`, which the proxy rewrites to the backend path.

#### Reverse proxy layouts (path rewrites)

Bokeh itself registers the login/logout routes and redirects browsers with a single path value, so a proxy that *rewrites* the path (e.g. Apache `ProxyPass /nectarcam-dqm http://host:5006/bokeh_app/` forwards `/bokeh_app/login` to the backend while browsers are told to open `/nectarcam-dqm/login`) would 404 at login. This is solved in `bokeh_app/auth.py`, which splits the two roles: routes live under `${DQM_AUTH_BACKEND_PREFIX}`, while every URL handed to the browser (login redirect, form action, `?next=`) uses `${DQM_AUTH_URL_PREFIX}`. The prefixes therefore never need to match, and a path-rewriting proxy works as-is:

* **Rewriting proxy (e.g. IT's current setup)** — keep serving the app under its own path and let the proxy rewrite the public name:
  ```shell
  export DQM_AUTH_URL_PREFIX=/nectarcam-dqm      # public path visitors use
  # DQM_AUTH_BACKEND_PREFIX stays at its default (/bokeh_app)
  bokeh serve --auth-module=/opt/cta/nectarchain/src/nectarchain/dqm/bokeh_app/auth.py ... bokeh_app
  ```
  with IT's existing `ProxyPass /nectarcam-dqm <SERVER:PORT>/bokeh_app/` (plus the matching `ProxyPassReverse`). Visitors open `/nectarcam-dqm/...`; the proxy rewrites to `/bokeh_app/...` on the Bokeh server.
* **Same path on both sides** (no rewrite) — e.g. a simple `ProxyPass /bokeh_app http://host:5006/bokeh_app`; leave both prefixes at their defaults (`/bokeh_app`). When the prefixes are equal, the module behaves exactly as before.

For security,
* serve the web app over HTTPS (e.g. with `--ssl-certfile`/`--ssl-keyfile`, or have your local IT team terminate TLS in front of it), so the password and session cookies are not exchanged in clear;
* when served over HTTPS, keep `DQM_AUTH_COOKIE_SECURE` at its default of `1`.

## Running the web app as a systemd service

As an alternative to the manual `bokeh serve` command above, the repository provides a systemd unit and an environment file in this directory:

* `dqm-web-app.service` — launches the Bokeh server (as the `nectarcam` user, from the `nectarchain` mamba environment) at boot, and restarts it on failure;
* `dqm-web-app.env` — environment file holding the shared-account settings (`DQM_AUTH_USER`, `DQM_AUTH_PASSWORD`, `DQM_AUTH_SECRET`).

Setup:

```shell
sudo mkdir -p /etc/dqm
sudo install -m 600 -o nectarcam -g cta dqm-web-app.env /etc/dqm/dqm-web-app.env
# then edit /etc/dqm/dqm-web-app.env to set the shared password and the
# cookie-signing secret (DQM_AUTH_SECRET), cf. section above
sudo install -m 644 dqm-web-app.service /etc/systemd/system/dqm-web-app.service
# replace the two --allow-websocket-origin entries in the unit file with the
# IP address / DNS alias of this host
sudo systemctl daemon-reload
sudo systemctl enable --now dqm-web-app
systemctl status dqm-web-app
journalctl -u dqm-web-app -f   # application logs (Bokeh + auth module)
```

Notes:

* The service runs the same command as the manual launch, minus `--show` (there is no browser on the VM).
* The Bokeh application also writes its own rotating logs to `bokeh_app/logs/app.log` (or the directory given by `$NECTARCHAIN_LOG`).
* To change the password, update `/etc/dqm/dqm-web-app.env` and run `sudo systemctl restart dqm-web-app`.

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

* ~~Open the access of the web app worldwide, protected with NectarCAM user/password.~~ Done: the web app is protected with a shared NectarCAM user/password via a Bokeh authentication module (`bokeh_app/auth.py`), see "Restricting access to the web app" above.
* ~~Open write access to the DB for worker nodes on DIRAC, presumably using temporary, volatile tokens.~~ For security reasons, the DB should rather not be open in write access to the world. Instead, the revised workflow is the following:
  * Launch DQM jobs on DIRAC when transferring the data from CEA to DIRAC.
  * Instead of *pushing* the DQM output from DIRAC workers to the DQM DB, the primary VM will *pull* them from DIRAC to feed the DB.
  * So, we need an automatic mechanism, such as a cronjob, pulling DQM data from DIRAC, parsing the DQM FITS files, and writing to the DB.
