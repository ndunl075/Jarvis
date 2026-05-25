"""Tests for jarvis.tools.local.volume.VolumeTool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.tools.local.volume import VolumeArgs, VolumeTool


@pytest.mark.parametrize("action,seam", [
    ("up", "volume_up"),
    ("down", "volume_down"),
])
async def test_up_and_down_call_correct_seam_with_amount(action, seam):
    with patch(f"jarvis.tools.local.volume.winplat.{seam}") as fn:
        result = await VolumeTool().execute(VolumeArgs(action=action, amount=7))
    assert result.success
    fn.assert_called_once_with(7)


@pytest.mark.parametrize("action", ["mute", "unmute"])
async def test_mute_and_unmute_both_toggle(action):
    """The Windows media-key surface only exposes a mute toggle, so both
    actions land on volume_mute_toggle. Documented behaviour."""
    with patch("jarvis.tools.local.volume.winplat.volume_mute_toggle") as toggle, \
         patch("jarvis.tools.local.volume.winplat.volume_up") as up, \
         patch("jarvis.tools.local.volume.winplat.volume_down") as down:
        result = await VolumeTool().execute(VolumeArgs(action=action))
    assert result.success
    toggle.assert_called_once_with()
    up.assert_not_called()
    down.assert_not_called()


async def test_non_windows_returns_error_not_raise():
    with patch(
        "jarvis.tools.local.volume.winplat.volume_up",
        side_effect=NotImplementedError("Windows-only"),
    ):
        result = await VolumeTool().execute(VolumeArgs(action="up"))
    assert not result.success


def test_amount_bounds_enforced_by_schema():
    with pytest.raises(Exception):  # noqa: B017
        VolumeArgs(action="up", amount=0)
    with pytest.raises(Exception):  # noqa: B017
        VolumeArgs(action="up", amount=99)


def test_requires_confirmation_false():
    assert VolumeTool().requires_confirmation is False
