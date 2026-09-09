"""Windows DPAPI primitives for encrypting config secrets at rest.

Same convention as `windows.py`: every Windows-specific call lives in
jarvis/platform/ so the rest of the code stays pure Python and testable.
`jarvis/core/config.py` never imports ctypes; it calls the string-level
helpers here and tests patch the `_protect` / `_unprotect` seam.

Why DPAPI and not a crypto library:
  - It is bound to the current *user account*, so another user on the same
    box cannot read the blob even with the file. That is exactly the threat
    model for `%APPDATA%\\Jarvis\\config.json`.
  - It needs no key management from us — no key file to protect, no
    passphrase to prompt for, nothing to lose. Jarvis has no crypto
    dependency today and should not gain one for this.
  - It is reachable from `ctypes`, so it costs zero install size.

Storage format (see `encrypt_secret`):

    dpapi:<base64 of CryptProtectData output>

Only the individual secret *values* are encrypted — never the whole file.
The rest of config.json stays human-readable and hand-editable, which is a
real usability property of this project. The `dpapi:` scheme prefix makes
an encrypted value obvious at a glance and is also how we tell "already
encrypted" from "plaintext, needs migration".

DELIBERATE TRADE-OFF — non-Windows hosts store secrets in plaintext.
DPAPI is a Win32 API with no portable equivalent that is worth a new
dependency here (Jarvis is a Windows app; Linux/macOS is a contributor
dev-machine story, not a shipping target). Rather than crash or refuse to
save, off-Windows we pass values through unchanged and log one warning per
process. Anything that would have been encrypted stays readable on disk —
the same situation as before this module existed, so nobody is worse off,
but it is a real gap and it is logged, not hidden.
"""

from __future__ import annotations

import base64
import binascii
import ctypes
import logging
import sys
from ctypes import wintypes

log = logging.getLogger(__name__)

# Scheme marker written before the base64 payload. Changing this string
# would strand every already-encrypted config, so: don't.
SCHEME_PREFIX = "dpapi:"

# Optional entropy mixed into the DPAPI blob. Not a secret (this file is
# open source) — it is a domain separator so a Jarvis blob cannot be
# decrypted by another app running as the same user that happens to call
# CryptUnprotectData on the bytes. Changing it makes existing ciphertext
# undecryptable, so it must stay fixed forever.
_ENTROPY = b"jarvis.config.secrets.v1"

# CryptProtectData / CryptUnprotectData flag: never show UI, fail instead.
# Jarvis loads config before there is a window to parent a prompt to.
_CRYPTPROTECT_UI_FORBIDDEN = 0x01

# One warning per process for the off-Windows plaintext fallback, so a
# config with several secrets does not spam the log on every save.
_warned_plaintext_fallback = False


class SecretError(Exception):
    """DPAPI call failed (bad blob, wrong user account, wrong machine)."""


# -- platform check -----------------------------------------------------


def _require_windows(feature: str) -> None:
    if sys.platform != "win32":
        raise NotImplementedError(
            f"{feature} is Windows-only; current platform: {sys.platform}"
        )


def dpapi_available() -> bool:
    """True when the DPAPI primitives below can actually run.

    Callers use this to choose between encrypting and the documented
    plaintext fallback. Tests patch this (plus `_protect`/`_unprotect`) to
    exercise the encrypted paths on Linux."""
    return sys.platform == "win32"


# -- raw DPAPI ----------------------------------------------------------


class _DataBlob(ctypes.Structure):
    """Win32 DATA_BLOB: a length + pointer pair."""

    _fields_ = [
        ("cbData", wintypes.DWORD),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _blob_in(data: bytes) -> tuple[_DataBlob, ctypes.Array]:
    """Build an input DATA_BLOB. The backing buffer is returned alongside it
    so the caller keeps it alive for the duration of the call — ctypes would
    otherwise free it while Windows still holds the pointer."""
    buf = ctypes.create_string_buffer(data, len(data))
    blob = _DataBlob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    return blob, buf


def _blob_bytes_and_free(blob: _DataBlob) -> bytes:
    """Copy an output DATA_BLOB into Python bytes and LocalFree the Win32
    allocation. Always call this on a successful Crypt*Data output blob."""
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32.LocalFree(blob.pbData)


def _protect(data: bytes) -> bytes:
    """CryptProtectData. Windows-only; raises SecretError on API failure."""
    _require_windows("DPAPI CryptProtectData")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)  # type: ignore[attr-defined]
    blob_in, _buf = _blob_in(data)
    ent_in, _ent_buf = _blob_in(_ENTROPY)
    blob_out = _DataBlob()
    ok = crypt32.CryptProtectData(
        ctypes.byref(blob_in),
        None,  # szDataDescr
        ctypes.byref(ent_in),
        None,  # pvReserved
        None,  # pPromptStruct
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not ok:
        err = ctypes.get_last_error()  # type: ignore[attr-defined]
        raise SecretError(f"CryptProtectData failed (error {err})")
    return _blob_bytes_and_free(blob_out)


def _unprotect(blob: bytes) -> bytes:
    """CryptUnprotectData. Windows-only; raises SecretError on API failure,
    which is the expected outcome for a blob produced by another user
    account or on another machine."""
    _require_windows("DPAPI CryptUnprotectData")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)  # type: ignore[attr-defined]
    blob_in, _buf = _blob_in(blob)
    ent_in, _ent_buf = _blob_in(_ENTROPY)
    blob_out = _DataBlob()
    ok = crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        None,  # ppszDataDescr
        ctypes.byref(ent_in),
        None,  # pvReserved
        None,  # pPromptStruct
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(blob_out),
    )
    if not ok:
        err = ctypes.get_last_error()  # type: ignore[attr-defined]
        raise SecretError(f"CryptUnprotectData failed (error {err})")
    return _blob_bytes_and_free(blob_out)


# -- string-level helpers (what config.py calls) ------------------------


def is_encrypted(value: object) -> bool:
    """True when `value` is a string in the `dpapi:` storage format."""
    return isinstance(value, str) and value.startswith(SCHEME_PREFIX)


def _warn_plaintext_once() -> None:
    global _warned_plaintext_fallback
    if _warned_plaintext_fallback:
        return
    _warned_plaintext_fallback = True
    log.warning(
        "Windows DPAPI unavailable on %s — config secrets (API keys, MCP "
        "auth tokens) are stored in PLAINTEXT in config.json. This is the "
        "documented non-Windows fallback; prefer environment variables "
        "(JARVIS_BRAVE_API_KEY, JARVIS_GROQ_API_KEY) on shared machines.",
        sys.platform,
    )


def _reset_warning_state() -> None:
    """Test hook: re-arm the one-time plaintext warning."""
    global _warned_plaintext_fallback
    _warned_plaintext_fallback = False


def encrypt_secret(value: object, *, setting: str = "secret") -> object:
    """Return `value` in on-disk form.

    - `None` and `""` round-trip as themselves. They carry no secret, and
      encrypting them would turn "unset" into meaningless ciphertext that
      no longer compares equal to the field default.
    - An already-`dpapi:`-prefixed string is returned unchanged, so calling
      this twice is a no-op. The idempotence matters: the v20->v21 migration
      and `save_config` both run over the same dict shape. The cost is that
      a *literal* secret beginning with `dpapi:` would be mistaken for
      ciphertext and stored readable; no provider Jarvis talks to issues
      keys in that shape (Brave `BSA…`, Groq `gsk_…`, Trayce hex), and the
      alternative — no way to recognise ciphertext — is far worse.
    - Non-string values (a hand-edited config with a number in there) pass
      through untouched; validation is pydantic's job, not ours.
    - If DPAPI is unavailable, or the call fails, the plaintext is returned
      unchanged. Silently losing a user's API key would be worse than
      storing it readable, so this never raises.
    """
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        return value
    if is_encrypted(value):
        return value
    if not dpapi_available():
        _warn_plaintext_once()
        return value
    try:
        blob = _protect(value.encode("utf-8"))
    except (SecretError, NotImplementedError, OSError) as e:
        log.warning("could not encrypt %s (%s); storing it in plaintext", setting, e)
        return value
    return SCHEME_PREFIX + base64.b64encode(blob).decode("ascii")


def decrypt_secret(value: object, *, setting: str = "secret") -> object:
    """Return the plaintext for an on-disk value.

    - `None` and `""` round-trip as themselves.
    - A value without the `dpapi:` prefix is a legacy plaintext secret from
      before this feature; it is returned as-is and gets encrypted on the
      next save. That is the whole migration story for existing users.
    - Undecryptable ciphertext returns `""`. This is the config-copied-to-
      another-machine (or another user account) case, and it is real: DPAPI
      blobs are user-bound. Crashing at startup over it would be terrible,
      so we log a warning naming the setting and hand back an empty value
      the user can re-enter in Settings.
    """
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        return value
    if not is_encrypted(value):
        return value  # legacy plaintext; re-encrypted on next save
    if not dpapi_available():
        _warn_plaintext_once()
        log.warning(
            "%s is DPAPI-encrypted but this host cannot decrypt it; "
            "treating it as empty",
            setting,
        )
        return ""
    payload = value[len(SCHEME_PREFIX) :]
    try:
        blob = base64.b64decode(payload, validate=True)
        plain = _unprotect(blob).decode("utf-8")
    except (
        SecretError,
        NotImplementedError,
        OSError,
        binascii.Error,
        UnicodeDecodeError,
        ValueError,
    ) as e:
        log.warning(
            "could not decrypt %s (%s) — it was most likely saved by a "
            "different Windows user account or on a different machine. "
            "Treating it as empty; re-enter it in Settings.",
            setting,
            e,
        )
        return ""
    return plain
