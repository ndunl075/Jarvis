"""Tests for jarvis.tools.local.open_app.OpenAppTool."""

from __future__ import annotations

from unittest.mock import patch

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
        launch_command=(
            r"C:\Users\me\AppData\Roaming\Microsoft\Windows"
            r"\Start Menu\Programs\Spotify.lnk"
        ),
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
