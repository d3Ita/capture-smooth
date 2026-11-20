@echo off
chcp 65001 >nul
cd /d "%~dp0"
python smooth.py --res 1440p --x 2
pause
