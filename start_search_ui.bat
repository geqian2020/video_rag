@echo off
setlocal
set "PYTHONUTF8=1"
set "APP_HOME=%~dp0"
set "APP_REPO=%APP_HOME%"
set "APP_INDEX=%APP_HOME%..\video_rag_data\index"
if exist "%APP_REPO%.venv\Scripts\python.exe" if exist "%APP_REPO%serve\run.py" goto paths_ready
rem Also support this BAT living in the sibling video_rag_data directory.
set "APP_REPO=%APP_HOME%..\video_rag-experiment-post-rerank-context\"
set "APP_INDEX=%APP_HOME%index"
:paths_ready
for %%D in ("%APP_REPO%.") do set "APP_REPO=%%~fD"
for %%D in ("%APP_INDEX%") do set "APP_INDEX=%%~fD"
set "APP_PYTHON=%APP_REPO%\.venv\Scripts\python.exe"

if not exist "%APP_PYTHON%" (
    echo [ERROR] Existing Python environment not found:
    echo "%APP_PYTHON%"
    echo Expected code directory: "%APP_REPO%"
    echo This launcher supports the repository root or the sibling video_rag_data directory.
    echo No software was installed or downloaded.
    pause
    exit /b 1
)

for %%F in (config.yaml core\config.py serve\__init__.py serve\run.py serve\api.py serve\search.html) do (
    if not exist "%APP_REPO%\%%F" (
        echo [ERROR] Required project file is missing: %%F
        echo Repository: "%APP_REPO%"
        echo Copy the serve folder from the update package into that CODE directory.
        echo Keep code files out of the video_rag_data directory.
        pause
        exit /b 1
    )
)

cd /d "%APP_REPO%"
if errorlevel 1 (
    echo [ERROR] Could not enter code directory: "%APP_REPO%"
    pause
    exit /b 1
)
echo Repository: "%APP_REPO%"
echo Index: "%APP_INDEX%"
echo Starting the local video search page. Model loading may take a moment.
echo The browser will open when the service is ready.
"%APP_PYTHON%" "%APP_REPO%\serve\run.py" --index-root "%APP_INDEX%" --open-browser %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Service stopped with exit code %APP_EXIT_CODE%.
    pause
)
exit /b %APP_EXIT_CODE%
