# 🎬⚡ RotoBrush 3.0 GPU Pro V2 (`VitMatte FP16 + Auto-Adaptive Despill + Optical Light Wrap`)

> **Cinema-Grade AI Rotoscoping, Sub-Pixel Hair Matting, 2.5D Depth Layering & Studio Background Replacement in Pure Python + PyTorch CUDA.**  
> Built to replace and surpass Adobe After Effects **Roto Brush 3.0 + Refine Edge** in automated video pipelines — with **zero manual brushstrokes**, **zero "Freeze" wait**, **zero white/green halos**, and **auto-adaptive background luminance calibration**.

---

## 🖼️ 400% Zoom Real-World Benchmark (8 Distinct People & Lighting Conditions)

![8-People 400% Zoom Forensic Benchmark](assets/AUDITORIA_8_PESSOAS_400X_ZOOM.png)

### 🧪 V1 vs. V2 Forensic Comparison (400% Zoom on Curly Hair, Glasses & White Shirt)

![V1 vs V2 400% Zoom Comparison](assets/AUDITORIA_FORENSE_V1_VS_V2_400X.png)

---

## 🧠 Why Standard AI Background Removers Look Like "Cheap Green Screen" (And How V2 Solves It)

Most open-source video matting models (`RVM`, `MODNet`, `MediaPipe`, or raw `BiRefNet`) fail in professional editing because of **6 classic optical defects**:
1. **Temporal Ghosting / White Streaks** when hands or shoulders move rapidly between keyframe anchors.
2. **Opaque Blobs Between Hair Curls** (bright wall pixels trapped inside dark or blonde curls).
3. **White/Gray Halo Around Shoulders & Shirt Collars** when composited over dark backgrounds.
4. **Wall Color Spill (Halation)** bleeding 6–12 pixels into semi-transparent hair strands.
5. **Dark Ring Artifacts on Dark Podcast Studios** when static despill clamps highlight luminance.
6. **"Cardboard Cutout / Sticker" Look** due to missing optical light wrap from the new background or 2.5D text.

### ⚡ The 6-Stage RotoBrush 3.0 GPU Pro V2 Architecture

| Stage | Module | What It Does Mathematically on GPU |
| :--- | :--- | :--- |
| **1. Temporal Core + Anti-Ghosting Gate** | `RVM ResNet50 ConvGRU (r1..r4)` + `BiRefNet_lite FP16` | Recurrent frame-to-frame memory with automatic scene-cut reset and a **Photometric Static Gate (`bi_static_gate`)** that blocks stale keyframe masks whenever a pixel moves (`diff > 0.025`), eliminating 100% of temporal white streaks. |
| **2. Sub-Pixel Hair Matting** | `VitMatte ViT FP16` (`hustvl/vitmatte-small-composition-1k`) + `Multi-Scale RGB Covariance Guided Filter` | Dynamically crops the head/hair ROI, builds an **Auto-Adaptive Trimap**, runs `VitMatte` in FP16 with fixed `(320, 480)` ViT tensor interpolation (zero VRAM fragmentation), and punches out trapped background pockets on the outer 3px curl fringe (`valley_key` + topological closure). |
| **3. Semantic Hair vs. Body Split** | `hair_zone_smooth` vs. `Body Choke (-2.5px)` | Automatically separates the upper hair/head region from the neck/shoulders/torso. Hair keeps full sub-pixel translucency, while the torso/shirt receives a surgical **-2.5px inward choke** with a crisp `1.1px` anti-aliased edge — zero wall aura around white shirts! |
| **4. Motion-Gated Temporal Damping** | `Reduce Chatter` (`--motion-damping 0.72`) | Locks alpha stability on static background/shoulder edges while opening 100% instant responsiveness on fast-moving hands and fingers. |
| **5. Deep Interior Pyramid Despill + Auto-Adaptive Luminance** | `decontaminate_edge_colors_gpu` (`bg_luma_auto_adapt`) | Erodes **7px into the body** and **14px into the hair core**, propagating pure interior RGB outward via a 3-scale pyramid (`96px` reach). Automatically compares external background luminance (`luma_bg`) against subject luminance (`luma_inp`) so it works on **bright white walls**, **colorful industrial rooms**, and **dark podcast studios** without manual tuning. |
| **6. Optical Light Wrap & Rim Bounce Roll-Off** | `compose_with_light_wrap_bgr` | Attenuates old room bounce light on the outer 4px rim of bright clothes (`rim_darken`) and wraps 2.5px of light from the new background or 2.5D typography around the speaker (`light_wrap_amount`). |

---

## 🚀 Quick Start (1-Click Windows CUDA Install)

### 1. Clone & Install
```bash
git clone https://github.com/NicolasCoder/rotobrush-gpu-pro.git
cd rotobrush-gpu-pro

# On Windows with NVIDIA GPU (RTX 3060 / 3070 / 4070 / 4090):
install_windows_cuda.bat
```
*(Or manually via pip)*:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
> **Zero Manual Model Setup**: On the first run, `rotobrush_gpu_pro.py` automatically downloads `rvm_resnet50_fp32.torchscript`, `ZhengPeng7/BiRefNet_lite`, and `hustvl/vitmatte-small-composition-1k` into `./models` and `./models_cache`.

---

## 💻 CLI Usage Examples

### A) Export Everything at Once (`Alpha Matte` + `Decontaminated FG` + `2.5D Text-Behind-Person` + `Studio Pro BG Replacement`)
```bash
python rotobrush_gpu_pro.py \
  --input speaker.mp4 \
  --output-dir ./outputs \
  --mode all \
  --title "ROTOBRUSH 3.0" \
  --subtitle "GPU PRO V2 • VITMATTE + LIGHT WRAP"
```

### B) Mode 1: Real 2.5D Depth (Keep Original Room with f/1.4 Bokeh + Put Giant Text Behind Speaker)
```bash
python rotobrush_gpu_pro.py \
  --input speaker.mp4 \
  --output-dir ./outputs \
  --mode 25d \
  --title "AUTORIDADE" \
  --subtitle "MODO 2.5D REAL • CENÁRIO ORIGINAL"
```

### C) Mode 2: Full Studio Pro Background Replacement (Cinema Dark Studio + Rim Glow + Light Wrap)
```bash
python rotobrush_gpu_pro.py \
  --input speaker.mp4 \
  --output-dir ./outputs \
  --mode studio \
  --title "ESTÚDIO PRO" \
  --subtitle "TROCA TOTAL DE CENÁRIO + LIGHT WRAP"
```

### D) Mode 3: Pure Alpha Matte & Decontaminated Foreground (For After Effects / Premiere / DaVinci / HyperFrames)
```bash
python rotobrush_gpu_pro.py \
  --input speaker.mp4 \
  --output-dir ./outputs \
  --mode alpha
```

### E) Mode 4: Ultra-Sharp 4K Static Image Cutout (PNG RGBA with Hair Strands)
```bash
python rotobrush_gpu_pro.py \
  --image portrait_4k.jpg \
  --output-png portrait_cutout_rgba.png
```

---

## 🐍 Python SDK Integration

```python
import cv2
from rotobrush_gpu_pro import RotoBrushGPUPro, RotoBrushConfig

# Initialize engine on NVIDIA GPU (FP16 VitMatte + BiRefNet + RVM ResNet50)
cfg = RotoBrushConfig(
    backbone="resnet50",
    use_birefnet_fusion=True,
    use_vitmatte_hair=True,
    shift_edge=-1.5,       # Hair edge shift (px)
    body_choke_px=-2.5,    # Surgical torso/shirt choke (px)
    decontaminate=0.98,    # Deep 14px interior color despill
    light_wrap_amount=0.24 # Optical light wrap
)
engine = RotoBrushGPUPro(cfg)

# Process frame-by-frame with recurrent temporal memory
frame_bgr = cv2.imread("frame.png")
fg_clean_bgr, alpha_u8 = engine.process_video_frame_bgr(frame_bgr)

# Composite over any new background with Optical Light Wrap
comp_bgr = engine.compose_with_light_wrap_bgr(
    fg_clean_bgr,
    alpha_u8,
    new_background_bgr,
    light_wrap_amount=0.24,
    rim_darken=0.16,
)
```

---

## ⚙️ Full CLI Parameters Reference (After Effects Roto Brush 3.0 Equivalent)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | — | Input video file (`.mp4`, `.mov`, `.mkv`) |
| `--output-dir`, `-o` | `<stem>_rotobrush_pro` | Output folder for rendered `.mp4` files and `rotobrush_report.json` |
| `--mode` | `all` | `all`, `25d`, `studio`, `alpha`, or `fg` |
| `--title` | `"ROTOBRUSH 3.0"` | Giant 2.5D headline rendered behind the speaker |
| `--subtitle` | `"GPU PRO V2..."` | Pill badge subtitle above the 2.5D headline |
| `--backbone` | `resnet50` | `resnet50` (maximum quality) or `mobilenetv3` (fastest) |
| `--refine-radius` | `8` | Uncertainty Trimap band radius in pixels (`Refine Edge Radius`) |
| `--feather` | `0.6` | Sub-pixel Gaussian feather on hair transitions |
| `--contrast` | `35.0` | S-curve contrast on alpha ramp (`0..100%`) |
| `--shift-edge` | `-1.5` | Hair edge choke/spread in pixels |
| `--body-choke` | `-2.5` | Surgical inward choke on torso/shoulders/shirt collar in pixels |
| `--motion-damping` | `0.72` | Temporal chatter reduction on static edges (`Reduce Chatter`) |
| `--decontaminate` | `0.98` | Deep interior color decontamination strength (`0.0..1.0`) |
| `--light-wrap` | `0.24` | Optical Light Wrap intensity around the composite border (`0.0..1.0`) |

---

## 📜 License
Released under the [MIT License](LICENSE). Created by **Nicolas (@NicolasCoder)**.
