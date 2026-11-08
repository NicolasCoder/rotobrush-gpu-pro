@echo off
chcp 65001 >nul
echo ============================================================================
echo  🎬⚡ ROTOBRUSH 3.0 GPU PRO V2 — INSTALADOR AUTOMÁTICO (NVIDIA CUDA)
echo ============================================================================
echo.
echo [1/3] Instalando PyTorch com suporte NVIDIA CUDA 12.1...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
echo [2/3] Instalando dependencias de Visao Computacional (VitMatte, BiRefNet, OpenCV)...
pip install -r requirements.txt
echo.
echo [3/3] Verificando placa de video NVIDIA CUDA...
python -c "import torch; print('CUDA Disponivel:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
echo.
echo ✅ Instalacao concluida! Use:
echo    python rotobrush_gpu_pro.py --input seu_video.mp4 --mode all --title "SEU TEXTO 2.5D"
pause
