@echo off
rem Starts humanmock. The first time, it sets itself up in a .venv folder here.
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\humanmock.exe" goto run

set "PYTHON="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "PYTHON=py -3"
if not defined PYTHON (
  where python >nul 2>nul && python -c "import sys; sys.exit(sys.version_info < (3, 11))" >nul 2>nul && set "PYTHON=python"
)
if not defined PYTHON (
  echo humanmock needs Python 3.11 or newer. Get it from https://www.python.org/downloads/
  pause
  exit /b 1
)

echo Setting up humanmock. This happens once and takes a few minutes.
%PYTHON% -m venv .venv || goto failed
".venv\Scripts\python.exe" -m pip install --quiet --disable-pip-version-check -e . || goto failed

:run
".venv\Scripts\humanmock.exe" start %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo humanmock could not start. The messages above say why.
pause
exit /b 1
