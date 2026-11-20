@echo off
chcp 65001 >nul
cd /d "%~dp0"
python smooth.py --test --windowed --test-game-fps 30 --x 3
pause
