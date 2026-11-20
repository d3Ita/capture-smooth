@echo off
chcp 65001 >nul
cd /d "%~dp0"
python smooth.py --show-taskbar
pause
