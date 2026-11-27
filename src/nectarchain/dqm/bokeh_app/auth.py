"""Authentication module for the NectarCAM DQM Bokeh web application.

This module implements Bokeh's authentication hooks (``bokeh serve
--auth-module``) so that the DQM web app requires a valid, shared NectarCAM
user/password before any data is served.

Enable at the Bokeh server level with::

    bokeh serve --enable-xsrf-cookies \
        --auth-module=/opt/cta/nectarchain/src/nectarchain/dqm/bokeh_app/auth.py \
        bokeh_app

Unauthenticated visitors are redirected to the login page; a successful login
sets a signed, HttpOnly session cookie and redirects back to the requested
page. The authenticated username is then made available in the application
through ``curdoc().session_context.user``, and a logout endpoint clears the
session cookie.

Paths and reverse proxies
~~~~~~~~~~~~~~~~~~~~~~~~~

Two paths are involved when the app is served behind a reverse proxy:

* ``DQM_AUTH_URL_PREFIX`` (default: ``/bokeh_app``) — the **public** path,
  i.e. what visitors type in their browser and what the reverse proxy exposes
  (e.g. ``https://host/nectarcam-dqm/...``).

* ``DQM_AUTH_BACKEND_PREFIX`` (default: ``/bokeh_app``) — the path under which
  the app is served **on the Bokeh server**, i.e. what the reverse proxy
  forwards to the Bokeh port (e.g. Apache
  ``ProxyPass /nectarcam-dqm http://server:5006/bokeh_app/`` delivers requests
  whose path starts with ``/bokeh_app``).

Bokeh registers the login/logout **routes** at ``login_url``/``logout_url``
and, in its stock configuration, also redirects unauthenticated users to the
*same* value. Behind a *path-rewriting* proxy the backend path
(``/bokeh_app/login``) and the public path (``/nectarcam-dqm/login``) differ,
so this module splits them: the routes live under ``DQM_AUTH_BACKEND_PREFIX``
(what the proxy forwards) while every URL handed to the browser — the login
redirect, the login form action and the logout landing page — uses
``DQM_AUTH_URL_PREFIX``. A module-level ``get_login_url()`` function returns
the public login URL (with the ``next`` parameter mapped back to the public
path), and the handlers render public URLs, so the two prefixes never need to
match and a path-rewriting proxy works without any proxy-side change.

Because Bokeh's default ``AuthProvider`` validation forbids supplying both a
``login_url`` and a ``get_login_url()`` function, this module relaxes *only*
those two exclusivity checks for the single provider it is loaded into (cf.
``_allow_dual_url_providers()``). When the two prefixes are equal (the usual
no-rewrite layout, e.g. a simple ``ProxyPass /bokeh_app .../bokeh_app``) the
module behaves exactly as before.

Environment variables (required):
    DQM_AUTH_PASSWORD       shared password for the service (compared against
                            in constant time)
    DQM_AUTH_SECRET         secret used to HMAC-sign session cookies. Generate
                            one with, e.g.:
                                python -c "import secrets;
                                print(secrets.token_urlsafe(32))"
Optional:
    DQM_AUTH_USER           shared username (default: "foo")
    DQM_AUTH_URL_PREFIX     public URL path under which the app is exposed
                            (default: "/bokeh_app"); used for every URL given
                            to the browser (see above)
    DQM_AUTH_BACKEND_PREFIX path under which the app is served on the Bokeh
                            server, i.e. what the reverse proxy forwards to
                            it (default: "/bokeh_app"); used for the
                            login/logout routes
    DQM_AUTH_COOKIE_SECURE  whether to set the ``Secure`` flag on the session
                            cookie (default: 1). Set to 0 only when the
                            service is served over plain http, e.g. for local
                            testing.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time

import tornado.escape
from tornado.web import RequestHandler

USERNAME = os.environ.get("DQM_AUTH_USER", "foo")
PASSWORD = os.environ.get("DQM_AUTH_PASSWORD", "")
SECRET = os.environ.get("DQM_AUTH_SECRET", "")
COOKIE_SECURE = os.environ.get("DQM_AUTH_COOKIE_SECURE", "1") == "1"

COOKIE_NAME = "dqm_user"
COOKIE_MAX_AGE_S = 12 * 60 * 60  # 12 h sessions

# Public path visitors use in their browser; every URL that is handed to the
# browser (login redirect, login form action, logout landing page) uses this
# prefix.
PUBLIC_PREFIX = os.environ.get("DQM_AUTH_URL_PREFIX", "/bokeh_app").rstrip("/")
# Path under which the app is served on the Bokeh server itself, i.e. what
# the reverse proxy forwards to it. The login/logout ROUTES are registered
# under this prefix.
BACKEND_PREFIX = os.environ.get("DQM_AUTH_BACKEND_PREFIX", "/bokeh_app").rstrip("/")

if not PASSWORD:
    raise RuntimeError("DQM_AUTH_PASSWORD must be set to enable authentication")
if not SECRET:
    raise RuntimeError("DQM_AUTH_SECRET must be set to enable authentication")


# ------------------------------------------------------------------ helpers
def check_credentials(username: str, password: str) -> bool:
    """Return True if the submitted credentials match the shared account."""
    return hmac.compare_digest(username, USERNAME) and hmac.compare_digest(
        password, PASSWORD
    )


def _sign(payload: str) -> str:
    return hmac.new(SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _encode(data: dict) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode(token: str) -> dict | None:
    try:
        raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _make_cookie(user: str) -> str:
    payload = _encode({"user": user, "exp": int(time.time()) + COOKIE_MAX_AGE_S})
    return f"{payload}.{_sign(payload)}"


def _get_cookie_user(cookie: str | None) -> str | None:
    """Validate a signed session cookie and return the username, or None."""
    if not cookie:
        return None
    try:
        payload, signature = cookie.rsplit(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(signature, _sign(payload)):
        return None
    data = _decode(payload)
    if not data or data.get("exp", 0) < time.time():
        return None
    return data.get("user")


def _safe_next(value: str) -> str:
    """Prevent open redirects: only allow same-server absolute paths."""
    if value.startswith("/") and not value.startswith("//"):
        return value
    return "/"


def _to_public(path: str) -> str:
    """Map a backend request path to the corresponding public URL path.

    The reverse proxy forwards ``DQM_AUTH_URL_PREFIX`` to the Bokeh server
    under ``DQM_AUTH_BACKEND_PREFIX``, so a backend request path such as
    ``/bokeh_app/run/42`` must be handed back to the browser as
    ``/nectarcam-dqm/run/42``. Paths not under the backend prefix are left
    unchanged (this covers the common case where both prefixes are equal).
    """
    if not path.startswith(BACKEND_PREFIX):
        return path
    tail = path[len(BACKEND_PREFIX) :]
    if not tail.startswith("/"):
        tail = "/" + tail
    return PUBLIC_PREFIX + tail


# ------------------------------------------------------------ bokeh hooks
def get_user(request_handler):
    """Bokeh hook: return the authenticated username, or None.

    When this returns None, Bokeh redirects the visitor to the URL returned
    by ``get_login_url`` below.
    """
    return _get_cookie_user(request_handler.get_cookie(COOKIE_NAME))


# Routes are registered under the backend prefix (what the proxy forwards).
login_url = f"{BACKEND_PREFIX}/login"


def get_login_url(request_handler):
    """Bokeh hook: the public URL to redirect unauthenticated users to.

    Returns the public login URL with a ``next`` parameter carrying the
    *public* form of the requested path. The query string is included on
    purpose: Tornado's ``@authenticated`` decorator would otherwise append a
    ``next`` built from the (backend) request path, which is not routable on
    the public side of the proxy.
    """
    next_url = _safe_next(_to_public(request_handler.request.uri))
    return f"{PUBLIC_PREFIX}/login?next={tornado.escape.url_escape(next_url)}"


class LoginHandler(RequestHandler):
    """Render the login page (GET) and validate credentials (POST)."""

    _PAGE = """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>@@TITLE@@</title>
<style>
  body { font-family: "Segoe UI", sans-serif; background: #fafafa;
         display: flex; justify-content: center; align-items: center;
         height: 100vh; margin: 0; }
  form { background: #fff; border: 1px solid #ddd; border-radius: 10px;
         padding: 2rem; width: 350px; }
  h1 { font-size: 1.2rem; text-align: center; }
  label { display: block; margin: 0.8rem 0 0.25rem; }
  input[type=text], input[type=password] { width: 100%; box-sizing: border-box;
         padding: 0.6rem; border: 1px solid #ddd; border-radius: 4px; }
  button { width: 100%; margin-top: 1.2rem; padding: 0.7rem; border: 0;
           border-radius: 4px; background: #1f77b4; color: #fff;
           text-transform: uppercase; cursor: pointer; }
  .error { color: #c0392b; margin-top: 0.8rem; }
  .hint { color: #888; font-size: 0.8rem; text-align: center; margin-top: 1rem; }
</style></head><body>
<form action="@@LOGIN_URL@@" method="post">
  @@XSRF@@
  <h1>NectarCAM Data Quality Monitoring</h1>
  <input type="hidden" name="next" value="@@NEXT@@">
  <label for="username">Username</label>
  <input name="username" id="username" type="text" value="@@USER@@"
         autocapitalize="off" autocorrect="off" autofocus required>
  <label for="password">Password</label>
  <input name="password" id="password" type="password" required>
  <button type="submit">Log in</button>
  <div class="error">@@ERROR@@</div>
  <div class="hint">This DQM service is protected by a shared NectarCAM account.</div>
</form></body></html>
"""

    def get(self):
        error = (
            "<p>Invalid username or password.</p>"
            if self.get_argument("error", "") == "1"
            else ""
        )
        next_url = tornado.escape.xhtml_escape(
            _safe_next(_to_public(self.get_argument("next", "/")))
        )
        xsrf = self.xsrf_form_html() if self.settings.get("xsrf_cookies") else ""
        page = (
            self._PAGE.replace("@@TITLE@@", "NectarCAM DQM - login")
            .replace(
                "@@LOGIN_URL@@",
                tornado.escape.xhtml_escape(f"{PUBLIC_PREFIX}/login"),
            )
            .replace("@@XSRF@@", xsrf)
            .replace("@@NEXT@@", next_url)
            .replace("@@USER@@", tornado.escape.xhtml_escape(USERNAME))
            .replace("@@ERROR@@", error)
        )
        self.write(page)

    def post(self):
        username = self.get_argument("username", "")
        password = self.get_argument("password", "")
        next_url = _safe_next(_to_public(self.get_argument("next", "/")))
        if check_credentials(username, password):
            self.set_cookie(
                COOKIE_NAME,
                _make_cookie(username),
                max_age=COOKIE_MAX_AGE_S,
                httponly=True,
                secure=COOKIE_SECURE,
                samesite="Lax",
            )
            self.redirect(next_url, permanent=False)
        else:
            next_arg = tornado.escape.url_escape(next_url)
            self.redirect(
                f"{PUBLIC_PREFIX}/login?error=1&next={next_arg}",
                permanent=False,
            )


logout_url = f"{BACKEND_PREFIX}/logout"


class LogoutHandler(RequestHandler):
    def get(self):
        self.clear_cookie(COOKIE_NAME)
        self.redirect(f"{PUBLIC_PREFIX}/", permanent=False)


# ------------------------------------------------- bokeh schema relaxation
def _allow_dual_url_providers() -> None:
    """Relax one Bokeh validation required by the path split above.

    Bokeh's ``AuthProvider._validate`` (and therefore the ``AuthModule`` used
    by ``bokeh serve --auth-module``) forbids combining ``login_url`` with a
    ``get_login_url()`` function, and forbids a ``LoginHandler`` together with
    ``get_login_url()``. A path-rewriting reverse proxy needs exactly that
    combination (routes under the backend prefix, redirects to the public
    prefix), so this module relaxes only those two exclusivity checks for the
    auth provider it is loaded into. It matches the Bokeh 3.x error messages;
    everything else in ``_validate`` is left untouched.

    Bokeh loads this file through ``load_auth_module`` *before* running the
    provider's ``_validate``, so the relaxation is in effect when this module
    is used as ``--auth-module``. When Bokeh is not importable (e.g. unit
    tests loading the module standalone) this is a no-op.
    """
    try:
        import bokeh.server.auth_provider as auth_provider
    except ImportError:
        return
    provider_cls = auth_provider.AuthProvider
    if getattr(provider_cls, "_nectarchain_dqm_allow_dual_url_providers", False):
        return
    original_validate = provider_cls._validate

    def relaxed_validate(self):
        try:
            original_validate(self)
        except ValueError as exc:
            message = str(exc)
            if "At most one of login_url or get_login_url" in message:
                return
            if "cannot be used with a get_login_url() function" in message:
                return
            raise

    provider_cls._validate = relaxed_validate
    setattr(provider_cls, "_nectarchain_dqm_allow_dual_url_providers", True)


_allow_dual_url_providers()
