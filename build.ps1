# Builds the two Windows releases into dist\:
#   CaptureSmooth-NVIDIA    PyTorch CUDA 12.8 (RTX 20 to RTX 50, GTX 16) + the DirectML engine as fallback
#                           -> built in .venv-cuda128 (pip install torch --index-url https://download.pytorch.org/whl/cu128)
#   CaptureSmooth-DirectML  ONNX Runtime + DirectML (any DX12 GPU) -> built in .venv-dml (no torch inside)
#
#   powershell -ExecutionPolicy Bypass -File build.ps1 [-Only NVIDIA|DirectML] [-Zip]
param([string]$Only = "", [switch]$Zip)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Venv($n) {
    foreach ($p in (Join-Path $PSScriptRoot "..\$n"), (Join-Path $PSScriptRoot $n)) { if (Test-Path "$p\Scripts\python.exe") { return $p } }
    throw "environnement $n introuvable"
}
$onnx = @("--add-data", "models\rife46_1920x1080_s100.onnx;models", "--add-data", "models\rife46_1920x1080_s50.onnx;models",
          "--add-data", "models\rife46_2560x1440_s50.onnx;models", "--add-data", "models\post_1920x1080.onnx;models",
          "--add-data", "models\post_2560x1440.onnx;models")
$common = @("--noconfirm", "--clean", "--noconsole", "--version-file", "version_info.txt", "--distpath", "dist", "--workpath", "build",
            "--collect-binaries", "glfw", "--collect-all", "windows_capture", "--exclude-module", "matplotlib", "--add-data", "assets\icon.ico;assets",
            "--icon", "assets\icon.ico",
            # Settings window: Qt Widgets only (PySide6 through QtPy). The tkinter one is not shipped, nor the Qt
            # modules the app does not use.
            "--exclude-module", "tkinter", "--exclude-module", "PyQt5", "--exclude-module", "PyQt6", "--exclude-module", "PySide2",
            "--exclude-module", "PySide6.QtNetwork", "--exclude-module", "PySide6.QtQml", "--exclude-module", "PySide6.QtQuick",
            "--exclude-module", "PySide6.QtOpenGL", "--exclude-module", "PySide6.QtSvg", "--exclude-module", "PySide6.QtPdf")

# The exe opens a settings window on double-click; only the rescue shortcut ships next to it.
$launchers = [ordered]@{
    "Reafficher la barre des taches.cmd" = "--show-taskbar"
}

function Finish($name) {
    $dir = "dist\$name"
    Remove-Item "$dir\_internal\cv2\opencv_videoio_ffmpeg*.dll" -ErrorAction SilentlyContinue  # ffmpeg backend unused
    # Qt extras the settings window does not need: software OpenGL (~20 MB, the app needs a real GPU anyway) and
    # Qt's own translations (the window is in French already).
    $qt = "$dir\_internal\PySide6"
    Remove-Item "$qt\opengl32sw.dll", "$qt\Qt6Network.dll", "$qt\Qt6Svg.dll", "$qt\plugins\platforms\qdirect2d.dll",
                "$qt\plugins\iconengines" -Recurse -ErrorAction SilentlyContinue
    Remove-Item "$qt\translations" -Recurse -ErrorAction SilentlyContinue
    Get-ChildItem "$qt\plugins\imageformats" -File -ErrorAction SilentlyContinue | Where-Object Name -ne "qico.dll" |
        Remove-Item  # only the .ico reader is used (window icon)
    foreach ($k in $launchers.Keys) {
        $body = "@echo off`r`nchcp 65001 >nul`r`ncd /d `"%~dp0`"`r`n$name.exe $($launchers[$k])`r`npause`r`n"
        [IO.File]::WriteAllText("$PSScriptRoot\$dir\$k", $body, [Text.UTF8Encoding]::new($false))
    }
    Copy-Item README.md, LICENSE, THIRD_PARTY.md $dir
    $mb = (Get-ChildItem $dir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("{0}: {1:N0} Mo" -f $name, $mb)
}

if ($Only -ne "NVIDIA") {
    & "$(Find-Venv .venv-dml)\Scripts\python.exe" -m PyInstaller @common @onnx --name CaptureSmooth-DirectML `
        --exclude-module torch --exclude-module av smooth.py  # raw capture only helps the CUDA engine
    if ($LASTEXITCODE) { throw "build DirectML" }
    Finish "CaptureSmooth-DirectML"
}

if ($Only -ne "DirectML") {
    & "$(Find-Venv .venv-cuda128)\Scripts\python.exe" -m PyInstaller @common @onnx --name CaptureSmooth-NVIDIA `
        --add-data "models\flownet_v4.6.pkl;models" --exclude-module onnx --exclude-module torchvision --exclude-module torchaudio `
        --exclude-module numba --exclude-module llvmlite --exclude-module bitsandbytes `
        --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module scipy --exclude-module pandas `
        --exclude-module transformers --exclude-module tensorboard --exclude-module IPython --exclude-module triton `
        --exclude-module sklearn --exclude-module PIL --exclude-module accelerate --exclude-module diffusers `
        --exclude-module nvvfx --hidden-import dataclasses --hidden-import numbers --hidden-import contextlib `
        smooth.py  # nvvfx (NVIDIA licence) is never shipped: downloaded by the user, it needs these stdlib modules
    if ($LASTEXITCODE) { throw "build NVIDIA" }
    Finish "CaptureSmooth-NVIDIA"
}

# Installers (Inno Setup 6): dist\installer\CaptureSmooth-<version>-<edition>-Setup.exe
if ($Zip) {
    $iscc = Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $iscc)) { $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" }
    foreach ($e in "DirectML", "NVIDIA") { & $iscc /Q "/DEdition=$e" installer\CaptureSmooth.iss }
}
