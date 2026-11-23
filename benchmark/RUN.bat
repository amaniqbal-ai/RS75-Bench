@echo off
setlocal
cd /d "%~dp0"
title Satellite benchmark - RUNNING  (use STOP.bat to stop)
echo ==================================================================
echo   Satellite benchmark
echo   Finished work is always kept. Double-click RUN.bat again to continue.
echo   To stop: double-click STOP.bat
echo ==================================================================
if not exist "%~dp0logs" mkdir "%~dp0logs"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo SETUP FAILED - read logs\setup.log, fix the problem, then double-click RUN.bat again.
  pause
  exit /b 1
)
if not exist "%~dp0runtime\python_path.txt" (
  echo SETUP FAILED - Python was not found. Delete the folder "runtime" and run again.
  pause
  exit /b 1
)
set /p PY=<"%~dp0runtime\python_path.txt"
"%PY%" "%~dp0main.py" run
echo.
echo ==================================================================
echo   The benchmark has finished or stopped. Details: logs\main.log
echo   Double-click RUN.bat to continue if it is not finished.
echo ==================================================================
pause
