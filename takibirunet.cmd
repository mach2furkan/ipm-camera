@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python ortami bulunamadi: .venv\Scripts\python.exe
    echo Once projenin Python ortamini kurun.
    pause
    exit /b 1
)
echo Kamera baglanti ekrani aciliyor...
".venv\Scripts\python.exe" -u -m tools.camera_app --traffic %*
set "camera_exit_code=%errorlevel%"
if "%camera_exit_code%"=="64" exit /b 0
if not "%camera_exit_code%"=="0" (
    echo Uygulama hata ile kapandi. Hata kodu: %camera_exit_code%
    pause
)
exit /b %camera_exit_code%
