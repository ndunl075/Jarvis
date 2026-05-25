# Build Jarvis for Windows (one-folder app + zip for download).
# Prerequisites: Python 3.11+ venv with pip install -e ".[dev]"
# Output:
#   dist\Jarvis\Jarvis.exe
#   dist\Jarvis-0.0.1-windows-x64.zip

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Create a venv first: python -m venv .venv; .venv\Scripts\pip install -e `".[dev]`""
}

$Version = & $Python -c "from importlib.metadata import version; print(version('jarvis'))"
$ZipName = "dist\Jarvis-$Version-windows-x64.zip"

Write-Host "==> Downloading/copying ML assets..."
& $Python packaging\download_assets.py

Write-Host "==> Running PyInstaller..."
& $Python -m PyInstaller --noconfirm --clean jarvis.spec

$DistDir = Join-Path $Root "dist\Jarvis"
if (-not (Test-Path (Join-Path $DistDir "Jarvis.exe"))) {
    Write-Error "Build failed: Jarvis.exe not found in dist\Jarvis"
}

Write-Host "==> Copying ML assets beside Jarvis.exe (portable layout)..."
$Bundle = Join-Path $Root "packaging\bundle"
Copy-Item -Recurse -Force (Join-Path $Bundle "voices") (Join-Path $DistDir "voices")
Copy-Item -Recurse -Force (Join-Path $Bundle "models") (Join-Path $DistDir "models")

Write-Host "==> Verifying bundle..."
& $Python packaging\verify_bundle.py

Write-Host "==> Creating zip for distribution..."
if (Test-Path $ZipName) { Remove-Item $ZipName -Force }
Compress-Archive -Path (Join-Path $DistDir "*") -DestinationPath $ZipName -CompressionLevel Optimal

Write-Host ""
Write-Host "Done."
Write-Host "  Run locally:  dist\Jarvis\Jarvis.exe"
Write-Host "  Upload:       $ZipName"
Write-Host ""
Write-Host "Bundled: wake-word, VAD, Whisper STT, Piper TTS voice."
Write-Host "Users still need Ollama installed separately (LLM is not bundled)."
Write-Host "Logs (if issues): %APPDATA%\Jarvis\logs\jarvis.log"
