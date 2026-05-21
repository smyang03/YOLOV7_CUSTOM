@echo off
call conda activate yolov7
if %errorlevel% neq 0 (
    echo [ERROR] conda activate yolov7 failed
    pause
    exit /b 1
)

python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    python -m pip install pyinstaller --quiet
)

python -m PyInstaller ^
    --onefile ^
    --windowed ^
    --name yolov7_export ^
    --hidden-import tkinter ^
    --hidden-import tkinter.filedialog ^
    --hidden-import tkinter.messagebox ^
    --collect-all torch ^
    --collect-all torchvision ^
    --collect-all onnx ^
    --collect-all onnxsim ^
    --collect-all onnxruntime ^
    --collect-all numpy ^
    --collect-all cv2 ^
    --collect-all yaml ^
    --collect-all models ^
    --collect-all utils ^
    export_gui.py

if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo.
echo Done! dist\yolov7_export.exe created.
echo Copy the EXE to YOLOv7 repo root and run.
pause
