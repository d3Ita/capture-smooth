# Builds the two Windows releases into dist\:
#   CaptureSmooth-NVIDIA    PyTorch + CUDA (fastest, NVIDIA only)  -> built with the Python that has torch
#   CaptureSmooth-DirectML  ONNX Runtime + DirectML (any DX12 GPU) -> built in .venv-dml (no torch inside)
#
#   powershell -ExecutionPolicy Bypass -File build.ps1 [-Only NVIDIA|DirectML] [-Zip]
param([string]$Only = "", [switch]$Zip)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$venv = Join-Path $PSScriptRoot "..\.venv-dml"
if (-not (Test-Path "$venv\Scripts\python.exe")) { $venv = Join-Path $PSScriptRoot ".venv-dml" }
$common = @("--noconfirm", "--clean", "--console", "--distpath", "dist", "--workpath", "build",
            "--collect-binaries", "glfw", "--exclude-module", "matplotlib", "--add-data", "assets\icon.ico;assets",
            "--icon", "assets\icon.ico")

# The exe opens a settings window on double-click; only the rescue shortcut ships next to it.
$launchers = [ordered]@{
    "Reafficher la barre des taches.cmd" = "--show-taskbar"
}

function Finish($name) {
    $dir = "dist\$name"
    Remove-Item "$dir\_internal\cv2\opencv_videoio_ffmpeg*.dll" -ErrorAction SilentlyContinue  # ffmpeg backend unused
    foreach ($k in $launchers.Keys) {
        $body = "@echo off`r`nchcp 65001 >nul`r`ncd /d `"%~dp0`"`r`n$name.exe $($launchers[$k])`r`npause`r`n"
        [IO.File]::WriteAllText("$PSScriptRoot\$dir\$k", $body, [Text.UTF8Encoding]::new($false))
    }
    Copy-Item README.md, LICENSE, THIRD_PARTY.md $dir
    $mb = (Get-ChildItem $dir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
    Write-Host ("{0}: {1:N0} Mo" -f $name, $mb)
}

if ($Only -ne "NVIDIA") {
    & "$venv\Scripts\python.exe" -m PyInstaller @common --name CaptureSmooth-DirectML `
        --add-data "models\rife46_1920x1080_s100.onnx;models" --add-data "models\rife46_1920x1080_s50.onnx;models" `
        --add-data "models\rife46_2560x1440_s50.onnx;models" --exclude-module torch smooth.py
    if ($LASTEXITCODE) { throw "build DirectML" }
    Finish "CaptureSmooth-DirectML"
}

if ($Only -ne "DirectML") {
    python -m PyInstaller @common --name CaptureSmooth-NVIDIA --add-data "models\flownet_v4.6.pkl;models" `
        --exclude-module onnxruntime --exclude-module onnx --exclude-module torchvision --exclude-module torchaudio `
        --exclude-module numba --exclude-module llvmlite --exclude-module bitsandbytes --exclude-module PySide6 `
        --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module scipy --exclude-module pandas `
        --exclude-module transformers --exclude-module tensorboard --exclude-module IPython --exclude-module triton `
        --exclude-module sklearn --exclude-module PIL --exclude-module accelerate --exclude-module diffusers `
        smooth.py
    if ($LASTEXITCODE) { throw "build NVIDIA" }
    Finish "CaptureSmooth-NVIDIA"
}
