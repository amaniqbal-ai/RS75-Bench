@echo off
setlocal
cd /d "%~dp0"
title Satellite benchmark - collecting results
echo ==================================================================
echo   Collecting all results into zip files (nothing is deleted)
echo ==================================================================
set PY=
if exist "%~dp0runtime\python_path.txt" set /p PY=<"%~dp0runtime\python_path.txt"
if not defined PY set PY=python
"%PY%" "%~dp0collect.py"
echo.
echo The zip files are in the folder COLLECT_OUTPUT (opening it now).
start "" "%~dp0COLLECT_OUTPUT"
pause
