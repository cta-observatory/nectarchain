"""Tests for the DQM Bokeh authentication module (``bokeh_app/auth.py``).

Bokeh loads the auth module as a standalone source file (see
``bokeh.server.auth_provider.load_auth_module``), so these tests load
``auth.py`` the same way, with a controlled environment, instead of importing
it through the package.

Run from the repository root with, e.g.::

    pytest src/nectarchain/dqm/bokeh_app/tests/test_auth.py
"""

from __future__ import annotations

import importlib.util
import os
import time
import uuid
from pathlib import Path
from types import ModuleType

import pytest

AUTH_PATH = Path(__file__).resolve().parents[1] / "auth.py"

ENV_USERNAME = "nectar"
ENV_PASSWORD = "s3cret-test-password"
ENV_SECRET = "test-secret"


def _default_env(**overrides) -> dict:
    env = {
        "DQM_AUTH_USER": ENV_USERNAME,
        "DQM_AUTH_PASSWORD": ENV_PASSWORD,
        "DQM_AUTH_SECRET": ENV_SECRET,
    }
    env.update(overrides)
    return env


def _load_auth(env: dict | None = None) -> ModuleType:
    """Load ``auth.py`` as a standalone module with the given environment.

    Entries set to None are dropped, i.e. treated as unset, which allows to
    test that the module refuses to load without the required variables.
    """
    if env is None:
        env = _default_env()
    env = {key: value for key, value in env.items() if value is not None}
    old_environ = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        module_name = f"dqm_auth_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, AUTH_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@pytest.fixture()
def auth_mod():
    """A freshly loaded auth module, as Bokeh would load it."""
    return _load_auth()


class _FakeRequestHandler:
    """Minimal stand-in for the Bokeh/Tornado RequestHandler passed to
    ``get_user``."""

    def __init__(self, cookie: str | None = None):
        self._cookie = cookie

    def get_cookie(self, name: str) -> str | None:
        return self._cookie


# ------------------------------------------------------------ credentials
def test_check_credentials_valid(auth_mod):
    assert auth_mod.check_credentials(ENV_USERNAME, ENV_PASSWORD) is True


@pytest.mark.parametrize(
    "username,password",
    [
        (ENV_USERNAME, "wrong"),
        ("other", ENV_PASSWORD),
        (ENV_USERNAME, ""),
        ("", ENV_PASSWORD),
        ("", ""),
    ],
)
def test_check_credentials_invalid(auth_mod, username, password):
    assert auth_mod.check_credentials(username, password) is False


# ---------------------------------------------------------------- cookies
def test_cookie_roundtrip(auth_mod):
    cookie = auth_mod._make_cookie(ENV_USERNAME)
    assert auth_mod._get_cookie_user(cookie) == ENV_USERNAME


def test_get_user_roundtrip(auth_mod):
    cookie = auth_mod._make_cookie(ENV_USERNAME)
    assert auth_mod.get_user(_FakeRequestHandler(cookie)) == ENV_USERNAME


@pytest.mark.parametrize("cookie", [None, "", "garbage", "a.b.c"])
def test_get_user_malformed(auth_mod, cookie):
    assert auth_mod.get_user(_FakeRequestHandler(cookie)) is None


def test_tampered_cookie_rejected(auth_mod):
    cookie = auth_mod._make_cookie(ENV_USERNAME)
    payload, signature = cookie.rsplit(".", 1)
    # Tamper with the signature.
    tampered_signature = ("0" if signature[0] != "0" else "1") + signature[1:]
    assert auth_mod._get_cookie_user(f"{payload}.{tampered_signature}") is None
    # Tamper with the payload itself.
    tampered_payload = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    assert auth_mod._get_cookie_user(f"{tampered_payload}.{signature}") is None


def test_expired_cookie_rejected(auth_mod):
    payload = auth_mod._encode({"user": ENV_USERNAME, "exp": int(time.time()) - 10})
    cookie = f"{payload}.{auth_mod._sign(payload)}"
    assert auth_mod._get_cookie_user(cookie) is None


# ----------------------------------------------------------- open-redirect
@pytest.mark.parametrize(
    "next_url,expected",
    [
        ("/bokeh_app", "/bokeh_app"),
        ("/", "/"),
        ("", "/"),
        ("//evil.example.org", "/"),
        ("http://evil.example.org", "/"),
        ("https://evil.example.org", "/"),
    ],
)
def test_safe_next(auth_mod, next_url, expected):
    assert auth_mod._safe_next(next_url) == expected


# --------------------------------------------------- environment handling
def test_raises_without_password():
    with pytest.raises(RuntimeError, match="DQM_AUTH_PASSWORD"):
        _load_auth(_default_env(DQM_AUTH_PASSWORD=None))


def test_raises_without_secret():
    with pytest.raises(RuntimeError, match="DQM_AUTH_SECRET"):
        _load_auth(_default_env(DQM_AUTH_SECRET=None))


def test_custom_username(auth_mod):
    module = _load_auth(_default_env(DQM_AUTH_USER="other_user"))
    assert module.check_credentials("other_user", ENV_PASSWORD) is True
    assert module.check_credentials(ENV_USERNAME, ENV_PASSWORD) is False
