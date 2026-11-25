"""Authentication module for the NectarCAM DQM Bokeh web application.

This module implements Bokeh's authentication hooks (``bokeh serve
--auth-module``) so that the DQM web app requires a valid, shared NectarCAM
user/password before any data is served.

Enable at the Bokeh server level with::

    bokeh serve --enable-xsrf-cookies \
        --auth-module=/opt/cta/nectarchain/src/nectarchain/dqm/bokeh_app/auth.py \
        --show bokeh_app

Unauthenticated visitors are redirected to ``/login``; a successful login
sets a signed, HttpOnly session cookie and redirects back to the requested
page. The authenticated username is then made available in the application
through ``curdoc().session_context.user``, and a ``/logout`` endpoint clears
the session cookie.

Environment variables (required):
    DQM_AUTH_PASSWORD       shared password for the service (compared against
                            in constant time)
    DQM_AUTH_SECRET         secret used to HMAC-sign session cookies. Generate
                            one with, e.g.:
                                python -c "import secrets;
                                print(secrets.token_urlsafe(32))"
Optional:
    DQM_AUTH_USER           shared username (default: "foo")
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


# ------------------------------------------------------------ bokeh hooks
def get_user(request_handler):
    """Bokeh hook: return the authenticated username, or None.

    When this returns None, Bokeh (Tornado ``@authenticated``) redirects the
    visitor to ``login_url``.
    """
    return _get_cookie_user(request_handler.get_cookie(COOKIE_NAME))


login_url = "/login"


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
<form action="/login" method="post">
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
            _safe_next(self.get_argument("next", "/"))
        )
        xsrf = self.xsrf_form_html() if self.settings.get("xsrf_cookies") else ""
        page = (
            self._PAGE.replace("@@TITLE@@", "NectarCAM DQM - login")
            .replace("@@XSRF@@", xsrf)
            .replace("@@NEXT@@", next_url)
            .replace("@@USER@@", tornado.escape.xhtml_escape(USERNAME))
            .replace("@@ERROR@@", error)
        )
        self.write(page)

    def post(self):
        username = self.get_argument("username", "")
        password = self.get_argument("password", "")
        next_url = _safe_next(self.get_argument("next", "/"))
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
            self.redirect(
                f"{login_url}?error=1&next={tornado.escape.url_escape(next_url)}",
                permanent=False,
            )


logout_url = "/logout"


class LogoutHandler(RequestHandler):
    def get(self):
        self.clear_cookie(COOKIE_NAME)
        self.redirect("/", permanent=False)
