@echo off
SETLOCAL

:: Set the virtual environment path
set VENV_ROOT_PATH=%USERPROFILE%\venv
set VENV_PATH=%VENV_ROOT_PATH%\slurm-pytorch-ddp-boilerplate

:: Ensure the folder exists or create it if not
if not exist "%VENV_ROOT_PATH%\" (
    echo Creating directory: %VENV_ROOT_PATH%
    mkdir "%VENV_ROOT_PATH%"
)

:setup
echo Creating venv environment...
python --version
python -m venv %VENV_PATH%
call %VENV_PATH%\Scripts\activate.bat
pip install -r requirements.txt
goto :eof

:gpu_setup
call :setup
echo Setting up GPU dependencies for CUDA 11.7...
pip install torch torchvision torchaudio
goto :eof

:cpu_setup
call :setup
echo Setting up CPU dependencies...
call %VENV_PATH%\Scripts\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
goto :eof

:clean
IF EXIST %VENV_PATH%\Scripts\deactivate.bat (
    echo Deactivating venv environment...
    call %VENV_PATH%\Scripts\deactivate.bat
)
echo Removing venv environment...
rmdir /s /q %VENV_PATH%
echo Venv environment removed successfully.
goto :eof

:main
IF "%~1"=="gpu" (
    call :gpu_setup
) ELSE IF "%~1"=="cpu" (
    call :cpu_setup
) ELSE IF "%~1"=="clean" (
    call :clean
) ELSE (
    echo Usage: %~nx0 {gpu^|cpu^|clean}
    exit /b 1
)
goto :eof

call :main
