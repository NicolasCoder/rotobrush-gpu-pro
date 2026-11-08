@echo off
chcp 65001 >nul
if "%~1"=="" (
    echo Uso: Arraste um arquivo de video .mp4 para cima deste arquivo run_demo.bat
    echo ou execute: run_demo.bat caminho_do_video.mp4
    pause
    exit /b 1
)
echo ============================================================================
echo  🎬⚡ ROTOBRUSH 3.0 GPU PRO V2 — PROCESSANDO VIDEO EM GPU NVIDIA
echo ============================================================================
python "%~dp0rotobrush_gpu_pro.py" --input "%~1" --mode all --title "ROTOBRUSH 3.0" --subtitle "GPU PRO V2 • VITMATTE + LIGHT WRAP"
pause
