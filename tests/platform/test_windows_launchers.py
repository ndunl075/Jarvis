"""Launcher regression tests for jarvis.platform.windows.

These launchers used to run `cmd /c start "" <target>` with shell=False.
That is not safe on Windows: subprocess.list2cmdline joins the argv list
back into one command line and quotes only arguments containing spaces or
tabs, so cmd.exe re-parsed the target and `notepad&calc` became two
commands. The launchers now call ShellExecuteW (os.startfile), which takes
the target as one opaque string.

Runs on Linux: _require_windows and the ShellExecuteW seam are patched.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.platform import windows as winplat

# Shell metacharacters that survived list2cmdline into cmd.exe, plus the
# control characters that can smuggle a second line into a command line.
INJECTION_TOKENS = [
    "notepad&calc",
    "notepad|calc",
    "notepad>out.txt",
    "notepad<in.txt",
    "notepad^&calc",
    'notepad"&calc',
    "notepad%PATH%",
    "notepad!PATH!",
    "notepad\ncalc",
    "notepad\r\ncalc",
    "notepad\x00calc",
]


@pytest.fixture
def shell_execute():
    """Patch both the platform gate and the single ShellExecuteW seam."""
    with (
        patch("jarvis.platform.windows._require_windows"),
        patch("jarvis.platform.windows._shell_execute") as exec_seam,
    ):
        yield exec_seam


@pytest.mark.parametrize("token", INJECTION_TOKENS)
def test_launch_app_rejects_shell_metacharacters(token, shell_execute):
    """A metacharacter token must never reach the launcher at all, so no
    command runs — let alone the attacker's second one."""
    with pytest.raises(ValueError):
        winplat.launch_app(token)
    shell_execute.assert_not_called()


@pytest.mark.parametrize("token", INJECTION_TOKENS)
def test_validate_launch_command_rejects_every_metacharacter(token):
    """The validator itself, independent of any platform gate."""
    with pytest.raises(ValueError):
        winplat.validate_launch_command(token)


def test_validate_launch_target_path_allows_command_metacharacters():
    """Paths are validated by a narrower rule on purpose: `&` and `!`
    appear in real app names and Store-app URIs, and a path never reaches
    a command line now."""
    winplat.validate_launch_target_path(r"C:\Games\Dungeons & Dragons.lnk")
    winplat.validate_launch_target_path("shell:appsFolder\\Foo_8wekyb3d8bbwe!App")


@pytest.mark.parametrize("name", ["chrome", "notepad", "msedge", "code", "wt"])
def test_launch_app_passes_bare_name_through_unchanged(name, shell_execute):
    """App Paths / PATH resolution still happens — ShellExecuteW does it —
    and the name arrives as one whole string."""
    winplat.launch_app(name)
    shell_execute.assert_called_once_with(name)


def test_launch_app_never_spawns_a_subprocess(shell_execute):
    """No cmd.exe (or any other child process) in the launch path."""
    with patch("jarvis.platform.windows.subprocess.Popen") as popen:
        winplat.launch_app("chrome")
    popen.assert_not_called()


def test_launch_app_rejects_empty_command(shell_execute):
    with pytest.raises(ValueError):
        winplat.launch_app("   ")
    shell_execute.assert_not_called()


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Program Files\App\app.exe",
        r"C:\Apps\Photoshop.lnk",
        r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Spotify.lnk",
        r"C:\Program Files\AT&T\app.exe",
        r"C:\Games\Dungeons & Dragons Online.lnk",
        r"C:\Program Files\Sid Meier's Civilization VI\Civ6.exe",
        "shell:appsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App",
    ],
)
def test_launch_path_passes_real_targets_through_unchanged(path, shell_execute):
    """Paths with spaces, .lnk shortcuts, Store-app URIs (which end in
    `!App`) and app names containing `&` all still launch: they were only
    dangerous while a command line re-parsed them."""
    winplat.launch_path(path)
    shell_execute.assert_called_once_with(path)


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Apps\app.exe|calc",
        r"C:\Apps\app.exe>out.txt",
        r"C:\Apps\app.exe<in.txt",
        'C:\\Apps\\app.exe"',
        "C:\\Apps\\app.exe\ncalc",
        "C:\\Apps\\app.exe\r\ncalc",
        "C:\\Apps\\app.exe\x00calc",
    ],
)
def test_launch_path_rejects_characters_illegal_in_a_windows_path(path, shell_execute):
    with pytest.raises(ValueError):
        winplat.launch_path(path)
    shell_execute.assert_not_called()


def test_launch_path_expands_environment_variables(shell_execute, monkeypatch):
    """`cmd /c start` expanded %VAR% before ShellExecuteW saw the target;
    App Paths values are sometimes REG_EXPAND_SZ, so keep that behaviour."""
    monkeypatch.setenv("JARVIS_TEST_PF", r"C:\Program Files")
    winplat.launch_path(r"%JARVIS_TEST_PF%\Foo\foo.exe")
    shell_execute.assert_called_once_with(r"C:\Program Files\Foo\foo.exe")


def test_launch_path_never_spawns_a_subprocess(shell_execute):
    with patch("jarvis.platform.windows.subprocess.Popen") as popen:
        winplat.launch_path(r"C:\Apps\Photoshop.lnk")
    popen.assert_not_called()


def test_launch_steam_game_uses_shell_execute(shell_execute):
    winplat.launch_steam_game(1245620)
    shell_execute.assert_called_once_with("steam://run/1245620")


def test_launch_shell_passes_store_uri_as_a_single_argument():
    """launch_shell runs explorer.exe, not cmd.exe: explorer takes the URI
    as one argument and never treats `&` as a command separator, so it was
    not part of the injection and still uses subprocess."""
    target = "shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"
    with (
        patch("jarvis.platform.windows._require_windows"),
        patch("jarvis.platform.windows.subprocess.Popen") as popen,
    ):
        winplat.launch_shell(target)
    args = popen.call_args.args[0]
    assert args == ["explorer.exe", target]


def test_launch_shell_rejects_control_characters():
    with (
        patch("jarvis.platform.windows._require_windows"),
        patch("jarvis.platform.windows.subprocess.Popen") as popen,
    ):
        with pytest.raises(ValueError):
            winplat.launch_shell("shell:AppsFolder\\Foo\nbar")
    popen.assert_not_called()
