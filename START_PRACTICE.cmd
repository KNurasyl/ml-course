@echo off
setlocal
set "courseRoot=%~dp0"
set "coursePython=%courseRoot%.venv\Scripts\python.exe"
if not exist "%coursePython%" (
    echo Python environment was not found in this course folder.
    echo Open README.md and follow the environment setup instructions.
    pause
    exit /b 1
)
if /I "%~1"=="--check" (
    "%coursePython%" -m jupyterlab --ServerApp.root_dir="%courseRoot%." --show-config-json
    exit /b
)
echo Starting the course practice in JupyterLab.
echo Keep this window open while working. Press Ctrl+C to stop.
"%coursePython%" -m jupyterlab --ServerApp.root_dir="%courseRoot%."
set "courseExit=%ERRORLEVEL%"
if not "%courseExit%"=="0" (
    echo JupyterLab stopped with an error. Keep the message above for troubleshooting.
    pause
)
exit /b %courseExit%
