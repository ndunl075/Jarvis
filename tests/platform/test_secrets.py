"""Tests for jarvis.platform.secrets: the DPAPI seam and the string helpers.

Runs on Linux: `dpapi_available` and the `_protect` / `_unprotect` seam are
patched with a reversible in-process stand-in, the same way the launcher
tests patch ShellExecuteW. The unpatched (real DPAPI) tests only assert the
NotImplementedError contract that `_require_windows` gives off-Windows.
"""

from __future__ import annotations

import base64
import logging
import sys

import pytest

from jarvis.platform import secrets as sec

FAKE_MARKER = b"FAKEDPAPI"


def _fake_protect(data: bytes) -> bytes:
    """Reversible stand-in for CryptProtectData. Not encryption — it only
    has to be a bijection with a recognisable header so `_fake_unprotect`
    can reject foreign blobs the way DPAPI rejects another user's."""
    return FAKE_MARKER + data[::-1]


def _fake_unprotect(blob: bytes) -> bytes:
    if not blob.startswith(FAKE_MARKER):
        raise sec.SecretError("CryptUnprotectData failed (error 13)")
    return blob[len(FAKE_MARKER) :][::-1]


@pytest.fixture
def dpapi(monkeypatch):
    """Pretend DPAPI is present and working."""
    monkeypatch.setattr(sec, "dpapi_available", lambda: True)
    monkeypatch.setattr(sec, "_protect", _fake_protect)
    monkeypatch.setattr(sec, "_unprotect", _fake_unprotect)
    sec._reset_warning_state()
    yield
    sec._reset_warning_state()


@pytest.fixture
def no_dpapi(monkeypatch):
    """Pretend DPAPI is absent (the non-Windows contributor machine)."""
    monkeypatch.setattr(sec, "dpapi_available", lambda: False)
    sec._reset_warning_state()
    yield
    sec._reset_warning_state()


# --- platform gate ---------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="off-Windows contract")
def test_primitives_raise_not_implemented_off_windows():
    """Same contract as jarvis.platform.windows._require_windows."""
    with pytest.raises(NotImplementedError):
        sec._protect(b"x")
    with pytest.raises(NotImplementedError):
        sec._unprotect(b"x")


def test_dpapi_available_tracks_platform():
    assert sec.dpapi_available() is (sys.platform == "win32")


# --- storage format --------------------------------------------------------


def test_encrypted_value_is_prefixed_base64(dpapi):
    out = sec.encrypt_secret("BSA-super-secret")
    assert isinstance(out, str)
    assert out.startswith("dpapi:")
    # Payload must be real base64 so the value stays copy/paste-safe.
    payload = out[len("dpapi:") :]
    assert base64.b64decode(payload, validate=True).startswith(FAKE_MARKER)


def test_is_encrypted(dpapi):
    assert sec.is_encrypted(sec.encrypt_secret("k")) is True
    assert sec.is_encrypted("plain") is False
    assert sec.is_encrypted("") is False
    assert sec.is_encrypted(None) is False
    assert sec.is_encrypted(42) is False


def test_ciphertext_does_not_contain_plaintext(dpapi):
    out = sec.encrypt_secret("gsk_abcdef0123456789")
    assert "gsk_abcdef0123456789" not in str(out)


# --- round-trip ------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "a",
        "BSAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "gsk_" + "z" * 52,
        "with spaces and = signs ==",
        "unicode: café — naïve 😀",
        "x" * 4096,
        "dpapi",  # prefix-like but not the prefix
    ],
)
def test_round_trip_is_exact(dpapi, value):
    assert sec.decrypt_secret(sec.encrypt_secret(value)) == value


def test_encrypt_is_idempotent(dpapi):
    once = sec.encrypt_secret("token")
    twice = sec.encrypt_secret(once)
    assert twice == once
    assert sec.decrypt_secret(twice) == "token"


# --- empty / None / non-string --------------------------------------------


def test_empty_string_round_trips_unencrypted(dpapi):
    """`brave_api_key` / `groq_api_key` default to "". Encrypting an empty
    value would turn "unset" into meaningless ciphertext."""
    assert sec.encrypt_secret("") == ""
    assert sec.decrypt_secret("") == ""


def test_none_round_trips_unencrypted(dpapi):
    """`auth_token` defaults to None; it must stay None, not become "" or
    a blob."""
    assert sec.encrypt_secret(None) is None
    assert sec.decrypt_secret(None) is None


def test_non_string_values_pass_through(dpapi):
    for value in (42, True, [], {}):
        assert sec.encrypt_secret(value) is value
        assert sec.decrypt_secret(value) is value


# --- legacy plaintext ------------------------------------------------------


def test_decrypt_passes_through_legacy_plaintext(dpapi):
    """A pre-encryption config still works; the value is re-encrypted on
    the next save."""
    assert sec.decrypt_secret("legacy-plaintext-key") == "legacy-plaintext-key"


# --- corrupt / foreign ciphertext -----------------------------------------


def test_undecryptable_blob_degrades_to_empty(dpapi, caplog):
    """Config copied from another machine or another Windows user account:
    DPAPI refuses, and we must not crash startup."""
    foreign = "dpapi:" + base64.b64encode(b"someone-elses-blob").decode()
    with caplog.at_level(logging.WARNING):
        out = sec.decrypt_secret(foreign, setting="research.brave_api_key")
    assert out == ""
    assert "research.brave_api_key" in caplog.text
    assert "re-enter it in Settings" in caplog.text


def test_non_base64_payload_degrades_to_empty(dpapi, caplog):
    with caplog.at_level(logging.WARNING):
        out = sec.decrypt_secret("dpapi:not!valid!base64", setting="x.y")
    assert out == ""
    assert "x.y" in caplog.text


def test_undecodable_utf8_degrades_to_empty(dpapi, monkeypatch, caplog):
    monkeypatch.setattr(sec, "_unprotect", lambda blob: b"\xff\xfe\xfd")
    blob = "dpapi:" + base64.b64encode(b"whatever").decode()
    with caplog.at_level(logging.WARNING):
        assert sec.decrypt_secret(blob, setting="mcp_servers[trayce].auth_token") == ""
    assert "mcp_servers[trayce].auth_token" in caplog.text


def test_encrypt_failure_keeps_plaintext(dpapi, monkeypatch, caplog):
    """Never lose the user's key. A DPAPI failure on the way out falls back
    to plaintext (logged) rather than writing an empty value."""

    def _boom(data: bytes) -> bytes:
        raise sec.SecretError("CryptProtectData failed (error 87)")

    monkeypatch.setattr(sec, "_protect", _boom)
    with caplog.at_level(logging.WARNING):
        out = sec.encrypt_secret("keep-me", setting="research.groq_api_key")
    assert out == "keep-me"
    assert "research.groq_api_key" in caplog.text


# --- non-Windows fallback --------------------------------------------------


def test_off_windows_encrypt_returns_plaintext(no_dpapi, caplog):
    """Documented trade-off: contributors on Linux/macOS keep plaintext
    rather than losing the ability to save at all."""
    with caplog.at_level(logging.WARNING):
        assert sec.encrypt_secret("dev-key") == "dev-key"
    assert "PLAINTEXT" in caplog.text


def test_off_windows_warning_is_logged_once(no_dpapi, caplog):
    with caplog.at_level(logging.WARNING):
        sec.encrypt_secret("a")
        sec.encrypt_secret("b")
        sec.encrypt_secret("c")
    hits = [r for r in caplog.records if "PLAINTEXT" in r.getMessage()]
    assert len(hits) == 1


def test_off_windows_decrypt_of_ciphertext_degrades_to_empty(no_dpapi, caplog):
    """A Windows-written config opened on Linux: nothing can read the blob,
    so hand back empty instead of leaking `dpapi:...` into an HTTP header."""
    with caplog.at_level(logging.WARNING):
        out = sec.decrypt_secret("dpapi:AAAA", setting="research.groq_api_key")
    assert out == ""
    assert "research.groq_api_key" in caplog.text


def test_off_windows_plaintext_values_still_work(no_dpapi):
    assert sec.decrypt_secret("plain") == "plain"
    assert sec.decrypt_secret("") == ""
    assert sec.decrypt_secret(None) is None
