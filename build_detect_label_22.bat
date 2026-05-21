@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

echo ============================================================
echo   Detect Label 22  —  Build Script  v1.0.0
echo ============================================================
echo.

:: ── YOLOV7_ROOT 확인 ───────────────────────────────────────────
if not defined YOLOV7_ROOT (
    echo [ERROR] YOLOV7_ROOT 환경 변수가 설정되지 않았습니다.
    echo.
    echo   예시:
    echo     set YOLOV7_ROOT=D:\codes\YOLOV7
    echo     build_detect_label_22.bat
    echo.
    pause
    exit /b 1
)

if not exist "%YOLOV7_ROOT%" (
    echo [ERROR] YOLOV7_ROOT 경로가 존재하지 않습니다: %YOLOV7_ROOT%
    pause
    exit /b 1
)

echo [INFO] YOLOV7_ROOT = %YOLOV7_ROOT%
echo.

:: ── conda 환경 활성화 (yolov7 환경이 없으면 base 사용) ─────────
call conda activate yolov7 2>nul
if %errorlevel% neq 0 (
    echo [WARN] conda 환경 yolov7 활성화 실패. 현재 Python 환경을 사용합니다.
)

:: ── PyInstaller 설치 확인 ──────────────────────────────────────
python -m pip show pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] PyInstaller 설치 중...
    python -m pip install pyinstaller --quiet
    if %errorlevel% neq 0 (
        echo [ERROR] PyInstaller 설치 실패
        pause
        exit /b 1
    )
)

:: ── 이전 빌드 정리 ─────────────────────────────────────────────
if exist "dist\DetectLabel22" (
    echo [INFO] 이전 빌드 정리 중...
    rmdir /s /q "dist\DetectLabel22"
)
if exist "build\DetectLabel22" (
    rmdir /s /q "build\DetectLabel22"
)

:: ── PyInstaller 빌드 ────────────────────────────────────────────
echo [INFO] PyInstaller 빌드 시작...
echo.

python -m PyInstaller ^
    --name DetectLabel22 ^
    --windowed ^
    --onedir ^
    --paths "%YOLOV7_ROOT%" ^
    --hidden-import tkinter ^
    --hidden-import tkinter.filedialog ^
    --hidden-import tkinter.messagebox ^
    --hidden-import tkinter.ttk ^
    --hidden-import torch ^
    --hidden-import torch.nn ^
    --hidden-import torch.nn.functional ^
    --hidden-import torchvision ^
    --hidden-import torchvision.ops ^
    --hidden-import torchvision.ops.nms ^
    --hidden-import numpy ^
    --hidden-import cv2 ^
    --hidden-import tqdm ^
    --hidden-import PIL ^
    --hidden-import PIL.Image ^
    --hidden-import models.experimental ^
    --hidden-import utils.datasets ^
    --hidden-import utils.general ^
    --hidden-import utils.torch_utils ^
    --collect-all torch ^
    --collect-all torchvision ^
    --collect-all numpy ^
    --collect-all cv2 ^
    --collect-all models ^
    --collect-all utils ^
    --add-data "%YOLOV7_ROOT%\models;models" ^
    --add-data "%YOLOV7_ROOT%\utils;utils" ^
    --noconfirm ^
    detect_label_22_gui.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 빌드 실패
    pause
    exit /b 1
)

:: ── detect_label_22.py 복사 (GUI가 import하므로 필요) ──────────
echo.
echo [INFO] 핵심 모듈 복사 중...
copy /Y detect_label_22.py "dist\DetectLabel22\" >nul

:: ── 실행 메뉴얼 생성 ────────────────────────────────────────────
echo [INFO] 사용 설명서 생성 중...
(
echo Detect Label 22  v1.0.0  —  사용 설명서
echo ==========================================
echo.
echo [실행 방법]
echo   DetectLabel22.exe 를 더블클릭하거나 터미널에서 실행하세요.
echo.
echo [입력 항목]
echo   source   : 이미지 폴더 또는 이미지 경로가 적힌 txt 파일
echo   weights  : YOLOv7 학습된 .pt 가중치 파일
echo   savedirs : 결과 이미지를 저장할 기본 경로
echo   project  : 로그/요약 저장 경로  (기본: runs/detect^)
echo   name     : 실행 이름  (기본: exp^)
echo.
echo [수치 설정]
echo   img-size  : 추론 이미지 크기  (기본: 1280^)
echo   conf-thres: 신뢰도 임계값  (기본: 0.25^)
echo   iou-thres : NMS IoU 임계값  (기본: 0.45^)
echo   min-recall: good_detect 기준 최소 recall  (기본: 0.8^)
echo   classes   : 평가할 클래스 번호  (예: 0,1,3,4  /  비우면 전체^)
echo.
echo [결과 폴더 구조]
echo   savedirs/
echo   ├── GOOD/  JPEGImages/  labels/  debug_images/
echo   ├── MISS/  JPEGImages/  labels/  debug_images/
echo   └── FAIL/  JPEGImages/  labels/  debug_images/
echo.
echo [결과 카테고리]
echo   GOOD  = good_detect (정상 감지^) + background (배경, 정상 무감지^)
echo   MISS  = miss_detect (미감지^)
echo   FAIL  = false_detect (오감지^) + low_conf (저신뢰도^)
echo.
echo [버튼]
echo   실행     : Detection 시작
echo   중지     : 현재 이미지 완료 후 중단
echo   결과 폴더: 저장된 결과 폴더를 탐색기로 열기
echo   리포트   : HTML 결과 리포트를 브라우저로 열기
) > "dist\DetectLabel22\사용설명서.txt"

:: ── 완료 ────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   빌드 완료!
echo   실행 파일: dist\DetectLabel22\DetectLabel22.exe
echo ============================================================
echo.
pause
