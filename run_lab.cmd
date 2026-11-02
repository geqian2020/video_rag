@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "pass" >nul 2>&1
if not errorlevel 1 (
  py -3.12 tools\lab.py %*
  goto done
)
py -3.11 -c "pass" >nul 2>&1
if not errorlevel 1 (
  py -3.11 tools\lab.py %*
  goto done
)
python tools\lab.py %*
:done
if errorlevel 1 echo Lab failed. See the message above and data\lab\logs.
pause
