@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "APP_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%APP_PYTHON%" (
    echo [ERROR] Existing Python environment not found:
    echo "%APP_PYTHON%"
    echo Put this launcher in the repository root, beside .venv and config.yaml.
    echo No software was installed or downloaded.
    pause
    exit /b 1
)

echo Starting the local video search page. Model loading may take a moment.
echo The browser will open when the service is ready.
"%APP_PYTHON%" -m serve.run --open-browser %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Service stopped with exit code %APP_EXIT_CODE%.
    pause
)
exit /b %APP_EXIT_CODE%
