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

for %%F in (config.yaml core\config.py serve\__init__.py serve\run.py serve\api.py serve\search.html) do (
    if not exist "%~dp0%%F" (
        echo [ERROR] Required project file is missing: %%F
        echo Repository: "%~dp0"
        echo Extract the complete update package here and preserve its folders.
        echo Put run.py and api.py inside the serve folder, not beside this BAT.
        pause
        exit /b 1
    )
)

echo Repository: "%~dp0"
echo Starting the local video search page. Model loading may take a moment.
echo The browser will open when the service is ready.
"%APP_PYTHON%" "%~dp0serve\run.py" --open-browser %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Service stopped with exit code %APP_EXIT_CODE%.
    pause
)
exit /b %APP_EXIT_CODE%
