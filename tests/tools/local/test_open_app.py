"""Tests for jarvis.tools.local.open_app.OpenAppTool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.tools.local.open_app import OpenAppArgs, OpenAppTool


async def test_known_alias_maps_to_executable_token():
    """`edge` aliases to `msedge`; when the app isn't in the installed
    index, the bare canonical token reaches the platform seam. Guards the
    alias table from silent drift. app_resolver returns None so the test
    doesn't touch the real registry/Start Menu."""
    with patch("jarvis.tools.local.open_app.winplat.launch_app") as launch:
        result = await OpenAppTool(app_resolver=lambda q: None).execute(
            OpenAppArgs(name="edge")
        )
    assert result.success
    launch.assert_called_once_with("msedge")


async def test_unknown_name_passes_through_to_shell_resolution():
    """Names absent from the alias table go straight to start, so any
    PATH-resolvable executable or file association still works."""
    with patch("jarvis.tools.local.open_app.winplat.launch_app") as launch:
        result = await OpenAppTool(app_resolver=lambda q: None).execute(
            OpenAppArgs(name="some_custom_app")
        )
    assert result.success
    launch.assert_called_once_with("some_custom_app")


async def test_open_app_tries_fallback_command_on_first_failure():
    from jarvis.platform.windows_apps import InstalledApp

    app = InstalledApp(
        display_name="Foo App",
        launch_command=r"C:\Apps\Foo\missing.exe",
    )
    calls: list[str] = []

    def launch_path(path: str) -> None:
        calls.append(path)
        raise OSError("path launch failed")

    def launch_app(command: str) -> None:
        calls.append(command)

    with (
        patch("jarvis.tools.local.open_app.winplat.launch_path", side_effect=launch_path),
        patch("jarvis.tools.local.open_app.winplat.launch_app", side_effect=launch_app),
    ):
        result = await OpenAppTool(app_resolver=lambda q: app).execute(
            OpenAppArgs(name="foo app")
        )
    assert result.success
    assert calls == [r"C:\Apps\Foo\missing.exe", "foo app"]


async def test_fuzzy_installed_app_launches_via_path():
    from jarvis.platform.windows_apps import InstalledApp

    app = InstalledApp(
        display_name="Adobe Photoshop",
        launch_command=r"C:\Apps\Photoshop.lnk",
    )
    with (
        patch("jarvis.tools.local.open_app.winplat.launch_path") as launch_path,
        patch("jarvis.tools.local.open_app.winplat.launch_app") as launch_app,
    ):
        result = await OpenAppTool(app_resolver=lambda q: app).execute(
            OpenAppArgs(name="photoshop")
        )
    assert result.success
    launch_path.assert_called_once_with(r"C:\Apps\Photoshop.lnk")
    launch_app.assert_not_called()
    assert "Photoshop" in (result.output or "")


async def test_alias_lookup_case_insensitive():
    with patch("jarvis.tools.local.open_app.winplat.launch_app") as launch:
        await OpenAppTool(app_resolver=lambda q: None).execute(
            OpenAppArgs(name="Spotify")
        )
    launch.assert_called_once_with("spotify")


async def test_empty_name_rejected():
    with patch("jarvis.tools.local.open_app.winplat.launch_app") as launch:
        result = await OpenAppTool().execute(OpenAppArgs(name="   "))
    assert not result.success
    launch.assert_not_called()


async def test_non_windows_returns_error_not_raise():
    with patch(
        "jarvis.tools.local.open_app.winplat.launch_app",
        side_effect=NotImplementedError("Windows-only"),
    ):
        result = await OpenAppTool(app_resolver=lambda q: None).execute(
            OpenAppArgs(name="chrome")
        )
    assert not result.success
    assert "Windows" in (result.error or "")


async def test_aliased_app_prefers_installed_index_over_bare_command():
    """Regression: a per-user install (e.g. Spotify) must launch via its
    real Start-Menu .lnk, not the bare `start spotify` command that
    silently succeeds without opening anything. The installed index is
    consulted even for aliased names."""
    from jarvis.platform.windows_apps import InstalledApp

    app = InstalledApp(
        display_name="Spotify",
        launch_command=r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Spotify.lnk",
    )
    with (
        patch("jarvis.tools.local.open_app.winplat.launch_path") as launch_path,
        patch("jarvis.tools.local.open_app.winplat.launch_app") as launch_app,
    ):
        result = await OpenAppTool(app_resolver=lambda q: app).execute(
            OpenAppArgs(name="spotify")
        )
    assert result.success
    # Real shortcut path launched; bare `start spotify` never reached.
    launch_path.assert_called_once_with(app.launch_command)
    launch_app.assert_not_called()
    assert "Spotify" in (result.output or "")


async def test_alias_falls_back_to_bare_command_when_not_installed():
    """If the index has no match, an aliased name still launches via its
    canonical bare command (App-Paths / PATH resolution)."""
    with patch("jarvis.tools.local.open_app.winplat.launch_app") as launch:
        result = await OpenAppTool(app_resolver=lambda q: None).execute(
            OpenAppArgs(name="vscode")
        )
    assert result.success
    launch.assert_called_once_with("code")  # vscode -> code


async def test_resolver_exception_does_not_crash_falls_through():
    """A resolver that raises (registry error, non-Windows) is treated as
    'no installed match' and the raw token still reaches `start`."""
    def _boom(_q: str):
        raise RuntimeError("registry exploded")

    with patch("jarvis.tools.local.open_app.winplat.launch_app") as launch:
        result = await OpenAppTool(app_resolver=_boom).execute(
            OpenAppArgs(name="some_custom_app")
        )
    assert result.success
    launch.assert_called_once_with("some_custom_app")


def test_requires_confirmation_false():
    assert OpenAppTool().requires_confirmation is False


# -- shell-metacharacter injection --------------------------------------
#
# Candidate 4 hands the raw resolved token (the LLM tool argument or the
# voice transcription) to the platform launcher. These tests patch one
# level deeper than the others — at jarvis.platform.windows's ShellExecuteW
# seam — so the real launcher, including its validation, runs.


@pytest.fixture
def shell_execute():
    with (
        patch("jarvis.platform.windows._require_windows"),
        patch("jarvis.platform.windows._shell_execute") as exec_seam,
    ):
        yield exec_seam


@pytest.mark.parametrize(
    "name",
    ["zzqq&calc", "zzqq|calc", "zzqq>out.txt", "zzqq^&calc", 'zzqq"&calc'],
)
async def test_metacharacter_token_launches_nothing(name, shell_execute):
    """A token that matches no installed app reaches candidate 4 verbatim.
    Nothing may be launched from it: not the attacker's second command, and
    not the first one either."""
    result = await OpenAppTool(app_resolver=lambda q: None).execute(
        OpenAppArgs(name=name)
    )
    assert not result.success
    shell_execute.assert_not_called()


async def test_newline_in_name_is_collapsed_not_injected(shell_execute):
    """A CR/LF never gets as far as the launcher from this tool:
    normalize_open_query collapses whitespace, so the target stays a single
    string. (launch_app rejects control characters regardless — see
    tests/platform/test_windows_launchers.py, which covers callers that do
    not normalize.)"""
    await OpenAppTool(app_resolver=lambda q: None).execute(
        OpenAppArgs(name="zzqq\ncalc")
    )
    assert shell_execute.call_args.args == ("zzqq calc",)


async def test_legitimate_token_reaches_launcher_whole(shell_execute):
    """The same path an injection token takes still launches real apps,
    with the name passed as a single string."""
    result = await OpenAppTool(app_resolver=lambda q: None).execute(
        OpenAppArgs(name="chrome")
    )
    assert result.success
    shell_execute.assert_called_once_with("chrome")


@pytest.mark.parametrize(
    "launch_command",
    [
        r"C:\Program Files\App\app.exe",
        r"C:\Apps\Photoshop.lnk",
        r"C:\Games\Dungeons & Dragons Online.lnk",
    ],
)
async def test_installed_app_paths_still_launch(launch_command, shell_execute):
    """Paths with spaces, .lnk shortcuts and app names containing `&` are
    not rejected: they never touch a command line now."""
    from jarvis.platform.windows_apps import InstalledApp

    app = InstalledApp(display_name="App", launch_command=launch_command)
    result = await OpenAppTool(app_resolver=lambda q: app).execute(
        OpenAppArgs(name="app")
    )
    assert result.success
    shell_execute.assert_called_once_with(launch_command)


async def test_installed_match_still_wins_over_unsafe_raw_token(shell_execute):
    """An installed app whose name contains `&` launches by its real path;
    the raw token is never needed, so nothing is rejected."""
    from jarvis.platform.windows_apps import InstalledApp

    app = InstalledApp(
        display_name="Dungeons & Dragons Online",
        launch_command=r"C:\Games\DDO.lnk",
    )
    result = await OpenAppTool(app_resolver=lambda q: app).execute(
        OpenAppArgs(name="dungeons & dragons online")
    )
    assert result.success
    shell_execute.assert_called_once_with(r"C:\Games\DDO.lnk")
