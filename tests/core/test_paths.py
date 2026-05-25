"""Tests for jarvis.paths."""

from __future__ import annotations

from jarvis.paths import install_root, is_frozen


def test_install_root_is_directory():
    root = install_root()
    assert root.is_dir()


def test_is_frozen_false_in_dev():
    assert is_frozen() is False


def test_bundled_asset_report_keys():
    from jarvis.paths import bundled_asset_report

    report = bundled_asset_report()
    assert "voices_dir" in report
    assert "whisper_root" in report
