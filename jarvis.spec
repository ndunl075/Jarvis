# PyInstaller spec — one-folder Windows build (dist/Jarvis/Jarvis.exe).
# Run:  .\build.ps1

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH)
BUNDLE = ROOT / "packaging" / "bundle"

block_cipher = None

datas: list[tuple[str, str]] = []
binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = list(collect_submodules("jarvis"))

for pkg in ("PySide6", "openwakeword", "onnxruntime", "faster_whisper", "piper"):
    try:
        tmp = collect_all(pkg)
        datas += tmp[0]
        binaries += tmp[1]
        hiddenimports += tmp[2]
    except Exception as exc:
        print(f"collect_all({pkg}) skipped: {exc}")

hiddenimports += [
    "jarvis.paths",
    "scipy.signal",
    "sounddevice",
    "pynput",
    "pynput.keyboard",
    "httpx",
    "ctranslate2",
    "mcp",
    "mcp.client",
    "mcp.client.stdio",
]

# openWakeWord ONNX files must live inside the package tree (wake_word.py
# resolves via openwakeword.__file__). Piper + Whisper + Silero are copied
# beside Jarvis.exe by build.ps1 (see packaging/verify_bundle.py).
ow_src = BUNDLE / "models" / "openwakeword"
if ow_src.is_dir():
    for onnx in ow_src.glob("*.onnx"):
        datas.append((str(onnx), "openwakeword/resources/models"))

a = Analysis(
    [str(ROOT / "jarvis" / "__main__.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "runtime_hook.py")],
    excludes=["torch", "tensorflow", "matplotlib", "tkinter"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Jarvis",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Jarvis",
)
