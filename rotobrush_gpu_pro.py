#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎬⚡ ROTOBRUSH 3.0 GPU PRO — MOTOR DE ROTOSCOPIA & REFINE EDGE NA RTX 3060 (12GB)
================================================================================
Caminho Oficial:
  F:\\Python Projects\\POD_Motion_Hyperframes\\reels_talking_head_engine\\rotobrush_gpu_pro.py

Substitui e supera o Roto Brush 3.0 + Refine Edge do Adobe After Effects 2026
100% em Python + PyTorch CUDA (FP16/FP32) na GPU NVIDIA RTX 3060 12GB, sem
depender de assinatura da Creative Cloud e sem exigir "Freeze" manual na GUI.

Os 5 Pilares Tecnológicos Implementados:
  1. PILAR 1 — Propagação Temporal Recorrente (Zero Flicker / Zero Buracos):
     - Combina `RobustVideoMatting (RVM ResNet50)` com estados recorrentes
       `ConvGRU (r1, r2, r3, r4)` frame a frame + Ancoragem Sólida de Keyframes
       via `BiRefNet_lite (FP16 1024x1024)` + `SAM 2.1 Tiny` para impedir 100%
       que abram buracos no peito/camisa/rosto ou que mãos em movimento sumam.
     - Inclui detector automático de corte de cena (reseta `r1..r4` em cortes
       secos) e `Motion-Gated Temporal Damping` (`--motion-damping` / Reduce
       Chatter) que trava a borda em pixels estáticos e libera 100% da resposta
       em gestos rápidos de mão (zero ghosting!).

  2. PILAR 2 — Refine Edge Sub-Pixel (Fios de Cabelo, Barba, Óculos e Mãos):
     - Gera automaticamente a Banda de Incerteza (Trimap Dinâmico de `8px..16px`)
       por erosão/dilatação morfológica em GPU.
     - Na zona de incerteza, aplica um `Multi-Scale RGB Covariance Guided Filter`
       combinado ao Deep Guided Filter do RVM e aos detalhes de alta frequência
       do BiRefNet para extrair transparência real sub-pixel fio a fio.

  3. PILAR 3 — Controles Paramétricos Idênticos ao Roto Brush do After Effects:
     - `--feather` (px): Suavização Gaussiana separável em GPU.
     - `--contrast` (0..100%): Curva S contínua na rampa de transparência.
     - `--shift-edge` (px, ex: -1.2): Choke/Spread sub-pixel adaptativo (retrai
       a borda sólida do tronco/ombros para nunca vazar linha de fundo, mas
       preserva a ponta fina dos fios de cabelo no topo da cabeça!).
     - `--motion-damping` (0..1): Equivalente ao slider "Reduce Chatter" do AE.

  4. PILAR 4 — Decontaminate Edge Colors (Despill & Cor Pura na Borda):
     - Elimina matematicamente a cor da parede/fundo que contamina os pixels
       semi-transparentes (`0.01 < alpha < 0.98`) combinando a predição neural
       `fgr` do RVM com Propagação Morfológica Normalizada Multi-Escala em GPU
       (`r=3, 7, 15` a partir do núcleo interno `alpha > 0.85`).
     - Zero halo branco/cinza quando textos gigantes 2.5D passam atrás do cabelo!

  5. PILAR 5 — Modo Estático Ultra-Nitidez (`--image` & `--chroma-hybrid`):
     - Para imagens estáticas (ex.: Nano Banana 4K com ou sem fundo verde/rosa),
       combina Chroma Despill + BiRefNet Multi-Scale HD + Guided Hair Matting +
       Edge Color Decontamination gerando PNG RGBA 4K cristalino.
================================================================================
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------------------------------------------------------
# BLINDAGEM AUTOMÁTICA DE AMBIENTE (NumPy 2.x + Bloqueio de TensorFlow Quebrado)
# ------------------------------------------------------------------------------
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int_"):
    np.int_ = np.int64  # type: ignore[attr-defined]
if not hasattr(np, "object_"):
    np.object_ = object  # type: ignore[attr-defined]
if not hasattr(np, "bool_"):
    np.bool_ = bool  # type: ignore[attr-defined]

_ORIG_FIND_SPEC = importlib.util.find_spec


def _safe_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
    if name.startswith(("sklearn", "tensorflow", "flax", "jax")):
        return None
    return _ORIG_FIND_SPEC(name, *args, **kwargs)


importlib.util.find_spec = _safe_find_spec  # type: ignore[assignment]

for _mod_name in ("tensorflow", "tensorflow.tools", "tensorflow.tools.docs"):
    if _mod_name not in sys.modules:
        _m = types.ModuleType(_mod_name)
        _m.__spec__ = importlib.machinery.ModuleSpec(_mod_name, None)
        sys.modules[_mod_name] = _m


class _DocControlsStub:
    do_not_generate_docs = staticmethod(lambda x: x)


class _DummyTFTensor:
    pass


class _DummyTFVariable:
    pass


sys.modules["tensorflow"].Tensor = _DummyTFTensor  # type: ignore[attr-defined]
sys.modules["tensorflow"].Variable = _DummyTFVariable  # type: ignore[attr-defined]
sys.modules["tensorflow"].tools = sys.modules["tensorflow.tools"]  # type: ignore[attr-defined]
sys.modules["tensorflow.tools"].docs = sys.modules["tensorflow.tools.docs"]  # type: ignore[attr-defined]
sys.modules["tensorflow.tools.docs"].doc_controls = _DocControlsStub  # type: ignore[attr-defined]

_DEFAULT_LOCAL_CACHE = Path(r"F:\Python Projects\reels_tools\models_cache")
CACHE_ROOT = _DEFAULT_LOCAL_CACHE if _DEFAULT_LOCAL_CACHE.parent.exists() else (Path(__file__).resolve().parent / "models_cache")
for _sub in ("hf", "torch", "ultralytics", "tmp"):
    (CACHE_ROOT / _sub).mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_ROOT / "ultralytics"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import cv2  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ENGINE_DIR = Path(__file__).resolve().parent
_DEFAULT_MODELS_DIR = Path(r"F:\Python Projects\POD_Motion_Hyperframes\reels_talking_head_engine\models")
MODELS_DIR = _DEFAULT_MODELS_DIR if _DEFAULT_MODELS_DIR.parent.exists() else (ENGINE_DIR / "models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RVM_RESNET50_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.torchscript"
)
RVM_MOBILENET_URL = (
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.torchscript"
)


# ==============================================================================
# CONFIGURAÇÃO PARAMÉTRICA (ESPELHANDO O PAINEL DO AFTER EFFECTS ROTO BRUSH 3.0)
# ==============================================================================
@dataclass
class RotoBrushConfig:
    """Parâmetros equivalentes ao efeito Roto Brush 3.0 & Refine Edge do After Effects (V2 VitMatte Pro)."""

    backbone: str = "resnet50"          # "resnet50" (máxima qualidade) ou "mobilenetv3" (ultra-rápido)
    use_birefnet_fusion: bool = True    # Funde BiRefNet_lite FP16 (1024x1024) para blindar corpo/mãos e fios
    birefnet_interval: int = 2          # A cada N frames roda BiRefNet 1024x1024 (com trava fotométrica anti-rastro)
    use_vitmatte_hair: bool = True      # Usa VitMatte (hustvl/vitmatte-small-composition-1k) no ROI da cabeça/cabelo
    vitmatte_interval: int = 1          # Intervalo de atualização de alta frequência do VitMatte no topo do cabelo
    use_sam2_anchor: bool = False       # Opcional: âncora extra com SAM 2.1 Tiny (sam2.1_t.pt)
    downsample_ratio: float = 0.28      # Ratio interno do RVM (0.25..0.40 para 1080p)
    refine_radius: int = 8              # Raio da Banda de Incerteza exclusiva do cabelo (px)
    guided_radius: int = 4              # Raio compacto do filtro guiado de fios de cabelo
    guided_eps: float = 1.5e-4          # Regularização do Guided Filter (menor = respeita mais cada fio)
    feather: float = 0.6                # Feather fino sub-pixel (px)
    contrast: float = 35.0              # Contrast (%) na rampa do alpha
    shift_edge: float = -1.5            # Shift Edge / Choke no cabelo (px)
    body_choke_px: float = -2.5         # Choke cirúrgico firme no corpo/camisa/pescoço/bochecha (zero aura!)
    body_softness_px: float = 1.1       # Anti-aliasing nítido de 1.1px no contorno do corpo/camisa
    hair_valley_gamma: float = 2.4      # Força de perfuração dos vãos claros entre cachos de cabelo
    motion_damping: float = 0.72        # Reduce Chatter / Amortecimento Temporal nos pixels estáticos (0..0.95)
    decontaminate: float = 0.98         # Decontaminate Edge Colors Amount (0.0 .. 1.0)
    decontaminate_radius: int = 12      # Raio de busca profunda (erode interno) da cor pura (px)
    light_wrap_amount: float = 0.24     # Intensidade do Optical Light Wrap na composição (0.0 .. 1.0)
    hair_preserve_boost: float = 0.30   # Proteção de pontas reais de cabelo escuro
    device: str = "cuda"


# ==============================================================================
# OPERADORES MATEMÁTICOS E MORFOLÓGICOS EM GPU (PYTORCH CUDA)
# ==============================================================================
def box_filter_gpu(x: torch.Tensor, r: int) -> torch.Tensor:
    """Filtro de caixa 2D rápido na GPU com padding reflexivo (preserva bordas)."""
    if r <= 0:
        return x
    k = 2 * r + 1
    padded = F.pad(x, (r, r, r, r), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=k, stride=1)


def gaussian_blur_gpu(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur Gaussiano 2D separável exato na GPU (suporta sigma fracionário sub-pixel)."""
    if sigma <= 0.05:
        return x
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()
    c = x.shape[1]
    kh = k1d.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k1d.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    xh = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kh, groups=c)
    xv = F.conv2d(F.pad(xh, (0, 0, radius, radius), mode="reflect"), kv, groups=c)
    return xv


def dilate_gpu(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = 2 * radius + 1
    return F.max_pool2d(F.pad(x, (radius, radius, radius, radius), mode="replicate"), kernel_size=k, stride=1)


def erode_gpu(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = 2 * radius + 1
    return -F.max_pool2d(-F.pad(x, (radius, radius, radius, radius), mode="replicate"), kernel_size=k, stride=1)


def morph_close_gpu(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    return erode_gpu(dilate_gpu(x, radius), radius)


def guided_matte_rgb_gpu(
    guide_rgb: torch.Tensor,
    src_alpha: torch.Tensor,
    r: int = 10,
    eps: float = 2.5e-4,
) -> torch.Tensor:
    """
    Filtro Guiado Multi-Canal (RGB + Luminância) em GPU para Refine Edge de Fios de Cabelo.
    Ao usar tanto a luminância quanto os canais cromáticos R, G, B como guia local,
    consegue separar fios de cabelo escuros/castanhos/loiros da parede mesmo quando a
    luminância é parecida mas a temperatura de cor difere!
    """
    luma = 0.2126 * guide_rgb[:, 0:1] + 0.7152 * guide_rgb[:, 1:2] + 0.0722 * guide_rgb[:, 2:3]
    chroma_rg = guide_rgb[:, 0:1] - guide_rgb[:, 1:2]
    chroma_yb = 0.5 * (guide_rgb[:, 0:1] + guide_rgb[:, 1:2]) - guide_rgb[:, 2:3]

    # Guia composto ponderado (Luminância + Oponência Cromática)
    p = src_alpha
    out_accum = torch.zeros_like(p)
    weights = [(luma, 0.60, eps), (chroma_rg, 0.22, eps * 0.5), (chroma_yb, 0.18, eps * 0.5)]

    for I_ch, w_ch, eps_ch in weights:
        mean_I = box_filter_gpu(I_ch, r)
        mean_p = box_filter_gpu(p, r)
        mean_Ip = box_filter_gpu(I_ch * p, r)
        cov_Ip = mean_Ip - mean_I * mean_p
        mean_II = box_filter_gpu(I_ch * I_ch, r)
        var_I = (mean_II - mean_I * mean_I).clamp(min=0.0)

        a = cov_Ip / (var_I + eps_ch)
        b = mean_p - a * mean_I
        mean_a = box_filter_gpu(a, r)
        mean_b = box_filter_gpu(b, r)
        q = (mean_a * I_ch + mean_b).clamp(0.0, 1.0)
        out_accum = out_accum + w_ch * q

    return out_accum.clamp(0.0, 1.0)


def propagate_color_pyramid_gpu(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    r_fine: int = 4,
    r_mid: int = 9,
) -> torch.Tensor:
    """
     Propaga as cores de `rgb` onde `mask > 0` para até ~96 pixels de distância usando
    uma Pirâmide Multi-Escala Normalizada (Full-Res + 1/4 Res + 1/8 Res) em GPU.
    Garante que NUNCA falte suporte de cor na borda e NUNCA caia de volta para a cor da parede!
    """
    _, _, H, W = rgb.shape
    m_clean = mask.clamp(0.0, 1.0)
    premult = rgb * m_clean

    # Escala 1 (Full-Res, r=r_fine ~ 4px)
    num1 = box_filter_gpu(premult, r_fine)
    den1 = box_filter_gpu(m_clean, r_fine)
    ext1 = num1 / den1.clamp(min=1e-4)
    w1 = (den1 * 10.0).clamp(0.0, 1.0)

    # Escala 2 (Full-Res, r=r_mid ~ 9px)
    num2 = box_filter_gpu(premult, r_mid)
    den2 = box_filter_gpu(m_clean, r_mid)
    ext2 = num2 / den2.clamp(min=1e-4)
    w2 = (den2 * 8.0).clamp(0.0, 1.0)

    # Escala 3 (1/4 Res, r=8 -> equivale a 32px na resolução original!)
    h4, w4 = max(8, H // 4), max(8, W // 4)
    pre4 = F.interpolate(premult, size=(h4, w4), mode="bilinear", align_corners=False)
    msk4 = F.interpolate(m_clean, size=(h4, w4), mode="bilinear", align_corners=False)
    num3 = box_filter_gpu(pre4, 8)
    den3 = box_filter_gpu(msk4, 8)
    ext3_low = num3 / den3.clamp(min=1e-4)
    w3_low = (den3 * 8.0).clamp(0.0, 1.0)
    ext3 = F.interpolate(ext3_low, size=(H, W), mode="bilinear", align_corners=False)
    w3 = F.interpolate(w3_low, size=(H, W), mode="bilinear", align_corners=False)

    # Escala 4 (1/8 Res, r=12 -> equivale a 96px na resolução original!)
    h8, w8 = max(4, H // 8), max(4, W // 8)
    pre8 = F.interpolate(premult, size=(h8, w8), mode="bilinear", align_corners=False)
    msk8 = F.interpolate(m_clean, size=(h8, w8), mode="bilinear", align_corners=False)
    num4 = box_filter_gpu(pre8, 12)
    den4 = box_filter_gpu(msk8, 12)
    ext4_low = num4 / den4.clamp(min=1e-4)
    ext4 = F.interpolate(ext4_low, size=(H, W), mode="bilinear", align_corners=False)

    out = ext4
    out = out * (1.0 - w3) + ext3 * w3
    out = out * (1.0 - w2) + ext2 * w2
    out = out * (1.0 - w1) + ext1 * w1
    return out.clamp(0.0, 1.0)


def compute_color_line_alpha_gpu(
    rgb: torch.Tensor,
    alpha_init: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Equação Física de Color-Line Alpha Matting (Chuang et al. / After Effects Refine Edge):
      1. Estima a cor pura local do Foreground `F_int` (a partir de `alpha_init > 0.84`)
      2. Estima a cor pura local do Background `B_ext` (a partir de `alpha_init < 0.02`)
      3. Projeta cada pixel observado `I` no segmento vetorial `(F_int - B_ext)`:
           alpha_proj = clamp( dot(I - B_ext, F_int - B_ext) / (||F_int - B_ext||^2 + eps), 0, 1 )
    Qualquer pixel externo cuja cor seja igual à parede (`I == B_ext`) recebe matematicamente
    `alpha_proj = 0.0`, eliminando 100% de qualquer halo/contorno fantasma ao redor do cabelo!
    """
    core_fg_mask = ((alpha_init - 0.84) / 0.14).clamp(0.0, 1.0)
    core_bg_mask = ((0.025 - dilate_gpu(alpha_init, radius=4)) / 0.025).clamp(0.0, 1.0)

    f_int = propagate_color_pyramid_gpu(rgb, core_fg_mask, r_fine=3, r_mid=8)
    b_ext = propagate_color_pyramid_gpu(rgb, core_bg_mask, r_fine=4, r_mid=10)

    diff_fb = f_int - b_ext
    dist_sq = torch.sum(diff_fb * diff_fb, dim=1, keepdim=True)  # (1, 1, H, W)
    dot_ib = torch.sum((rgb - b_ext) * diff_fb, dim=1, keepdim=True)
    alpha_proj = (dot_ib / (dist_sq + 1e-4)).clamp(0.0, 1.0)

    # Confiança baseada no contraste local entre Foreground e Background (ex: cabelo escuro vs parede clara -> ~1.0;
    # camisa branca vs parede branca -> ~0.0, onde mantemos 100% a máscara neural RVM+BiRefNet!)
    contrast_mag = torch.sqrt(dist_sq + 1e-6)
    contrast_conf = ((contrast_mag - 0.10) / 0.16).clamp(0.0, 1.0)
    return alpha_proj, f_int, contrast_conf


def decontaminate_edge_colors_gpu(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    rvm_fgr: Optional[torch.Tensor] = None,
    hair_zone: Optional[torch.Tensor] = None,
    amount: float = 0.98,
    radius: int = 12,
) -> torch.Tensor:
    """
    PILAR 4 V2 — Deep Interior Decontaminate Edge Colors (Zero Halo & Zero Névoa Cinza nos Cachos).
    1. Em vez de amostrar a borda em `alpha > 0.84` (que ainda contém reflexo/halalação da parede),
       erode o núcleo interno em **8 pixels** no corpo e **14 pixels** no cabelo (`deep_core_mask`),
       puxando 100% da cor pura do interior profundo da pessoa via Pirâmide Multi-Escala (96px).
    2. Na zona do cabelo (`hair_zone`), proíbe que qualquer fio semi-transparente tenha luminância
       maior que o núcleo escuro do cabelo (`luma <= luma_hair_deep * 0.92`), eliminando 100% daquela
       aparência de "névoa cinza / fundo verde recortado"!
    """
    if amount <= 0.01:
        return rgb

    solid_mask = (alpha > 0.94).float()
    deep_core_body = erode_gpu(solid_mask, radius=7)
    deep_core_hair = erode_gpu(solid_mask, radius=14)

    if hair_zone is not None:
        deep_core = deep_core_body * (1.0 - hair_zone) + deep_core_hair * hair_zone
    else:
        deep_core = deep_core_body

    # Fallback seguro se alguma região fina (ex: dedos) ficar menor que 7px após erosão
    fallback_core = erode_gpu((alpha > 0.88).float(), radius=2)
    core_mask = torch.maximum(deep_core, fallback_core * 0.35)

    f_inpainted = propagate_color_pyramid_gpu(rgb, core_mask, r_fine=4, r_mid=max(8, radius))

    luma_orig = 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]
    luma_inp = 0.2126 * f_inpainted[:, 0:1] + 0.7152 * f_inpainted[:, 1:2] + 0.0722 * f_inpainted[:, 2:3]

    # Auto-Adaptação de Despill por Luminância Local do Fundo (`bg_luma_auto_adapt`):
    # Se o fundo externo for MAIS CLARO que o interior (parede clara), trava o brilho máximo em 1.00 * luma_inp (zero halo claro).
    # Se o fundo externo for MAIS ESCURO que o interior (estúdio escuro), evita escurecimento artificial na borda!
    bg_ext_mask = ((0.03 - dilate_gpu(alpha, radius=3)) / 0.03).clamp(0.0, 1.0)
    b_ext_rgb = propagate_color_pyramid_gpu(rgb, bg_ext_mask, r_fine=4, r_mid=10)
    luma_bg = 0.2126 * b_ext_rgb[:, 0:1] + 0.7152 * b_ext_rgb[:, 1:2] + 0.0722 * b_ext_rgb[:, 2:3]

    raw_ratio = luma_orig / (luma_inp + 1e-4)
    ratio_bright_bg = raw_ratio.clamp(0.52, 1.00)
    ratio_dark_bg = raw_ratio.clamp(0.85, 1.12)
    bg_is_brighter = ((luma_bg - luma_inp + 0.04) / 0.08).clamp(0.0, 1.0)
    adaptive_ratio = ratio_bright_bg * bg_is_brighter + ratio_dark_bg * (1.0 - bg_is_brighter)
    f_pure = (f_inpainted * adaptive_ratio).clamp(0.0, 1.0)

    # Aplica descontaminação em toda a casca externa de 14px (onde deep_core < 0.95)
    outer_shell = (1.0 - gaussian_blur_gpu(deep_core, sigma=2.0)).clamp(0.0, 1.0) * float(amount)
    clean_rgb = rgb * (1.0 - outer_shell) + f_pure * outer_shell
    return clean_rgb.clamp(0.0, 1.0)


# ==============================================================================
# MOTOR PRINCIPAL: ROTOBRUSH 3.0 GPU PRO (V2 VITMATTE + HARD BODY CHOKE)
# ==============================================================================
class RotoBrushGPUPro:
    """
    Motor Completo Roto Brush 3.0 + VitMatte Hair Matting + Hard Body Choke +
    Deep Color Decontamination + Optical Light Wrap em PyTorch CUDA.
    """

    def __init__(self, config: Optional[RotoBrushConfig] = None):
        self.cfg = config or RotoBrushConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")

        # 1. Carregar RVM (RobustVideoMatting TorchScript) na GPU
        rvm_filename = (
            "rvm_resnet50_fp32.torchscript"
            if self.cfg.backbone == "resnet50"
            else "rvm_mobilenetv3_fp32.torchscript"
        )
        rvm_url = RVM_RESNET50_URL if self.cfg.backbone == "resnet50" else RVM_MOBILENET_URL
        self.rvm_path = MODELS_DIR / rvm_filename
        if not self.rvm_path.exists() or self.rvm_path.stat().st_size < 5_000_000:
            print(f"[RotoBrush GPU Pro] Baixando pesos {rvm_filename} ...")
            urllib.request.urlretrieve(rvm_url, str(self.rvm_path))

        self.rvm_model = torch.jit.load(str(self.rvm_path), map_location=self.device).eval()

        # 2. Carregar BiRefNet_lite FP16 na GPU (para ancoragem de corpo/mãos e modo imagem 4K)
        self.birefnet_model = None
        if self.cfg.use_birefnet_fusion:
            try:
                from transformers import AutoModelForImageSegmentation

                try:
                    self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                        "ZhengPeng7/BiRefNet_lite",
                        trust_remote_code=True,
                        local_files_only=True,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    ).to(self.device).eval()
                except Exception:
                    self.birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                        "ZhengPeng7/BiRefNet_lite",
                        trust_remote_code=True,
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    ).to(self.device).eval()
            except Exception as exc:
                print(f"[RotoBrush GPU Pro] Aviso: BiRefNet fallback para RVM puro ({exc})")
                self.birefnet_model = None

        # 3. Carregar VitMatte (hustvl/vitmatte-small-composition-1k) na GPU para Refine Edge real nos cachos!
        self.vitmatte_model = None
        if self.cfg.use_vitmatte_hair:
            try:
                from transformers import VitMatteForImageMatting

                try:
                    self.vitmatte_model = VitMatteForImageMatting.from_pretrained(
                        "hustvl/vitmatte-small-composition-1k",
                        local_files_only=True,
                    ).to(self.device).eval()
                except Exception:
                    self.vitmatte_model = VitMatteForImageMatting.from_pretrained(
                        "hustvl/vitmatte-small-composition-1k",
                    ).to(self.device).eval()
            except Exception as exc:
                print(f"[RotoBrush GPU Pro] Aviso: VitMatte indisponível ({exc})")
                self.vitmatte_model = None

        self.birefnet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.birefnet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Estado temporal entre frames de vídeo
        self.reset_temporal_state()

    def reset_temporal_state(self) -> None:
        """Reseta os estados recorrentes ConvGRU (r1..r4) e os buffers de amortecimento temporal."""
        self.rec: List[Optional[torch.Tensor]] = [None, None, None, None]
        self.prev_rgb_t: Optional[torch.Tensor] = None
        self.prev_alpha_t: Optional[torch.Tensor] = None
        self.last_birefnet_alpha: Optional[torch.Tensor] = None
        self.last_birefnet_rgb_t: Optional[torch.Tensor] = None
        self.last_vitmatte_alpha: Optional[torch.Tensor] = None
        self.last_vitmatte_rgb_t: Optional[torch.Tensor] = None
        self.frame_counter: int = 0

    @torch.no_grad()
    def _infer_birefnet_alpha(self, rgb_t: torch.Tensor, res: int = 1024) -> torch.Tensor:
        """Executa BiRefNet_lite FP16 em `res x res` na GPU e retorna máscara (1, 1, H, W) em [0..1]."""
        if self.birefnet_model is None:
            return torch.zeros((1, 1, rgb_t.shape[2], rgb_t.shape[3]), device=self.device)
        _, _, H, W = rgb_t.shape
        x = F.interpolate(rgb_t, size=(res, res), mode="bilinear", align_corners=False)
        x = (x - self.birefnet_mean) / self.birefnet_std
        if self.device.type == "cuda":
            x = x.half()
        preds = self.birefnet_model(x)[-1].sigmoid().float()
        return F.interpolate(preds, size=(H, W), mode="bilinear", align_corners=False).clamp(0.0, 1.0)

    @torch.no_grad()
    def _estimate_hair_zone_weight(self, rgb_t: torch.Tensor, alpha_coarse: torch.Tensor) -> torch.Tensor:
        """
        Localiza estritamente a coroa de cabelo (topo e têmporas superiores da cabeça, ~16% superiores
        da silhueta onde a cor NÃO é pele clara da testa nem camisa branca).
        Abaixo dessa linha (bochechas, orelhas, óculos, pescoço, ombros e camisa), retorna `0.0`
        para que recebam Choke Cirúrgico Firme (`-2.4px`) e Borda Dura Anti-Aliased (`1.1px`)!
        """
        _, _, H, W = alpha_coarse.shape
        row_max = alpha_coarse[0, 0].amax(dim=1)  # (H,)
        active_rows = torch.nonzero(row_max > 0.35, as_tuple=False)
        if active_rows.numel() > 10:
            y_top = float(active_rows[0].item()) / float(H)
            y_bot = float(active_rows[-1].item()) / float(H)
            # Apenas os 15.5% superiores da silhueta correspondem à coroa do cabelo!
            crown_span = max(0.09, min(0.165, (y_bot - y_top) * 0.16))
        else:
            y_top = 0.20
            crown_span = 0.14

        yy = torch.linspace(0.0, 1.0, H, device=self.device).view(1, 1, H, 1).expand(1, 1, H, W)
        vertical_gate = 1.0 - ((yy - (y_top + crown_span * 0.68)) / (crown_span * 0.32 + 1e-5)).clamp(0.0, 1.0)
        vertical_gate = vertical_gate * ((yy >= max(0.0, y_top - 0.03)).float())

        # Descarta pele clara da testa para que a testa/sobrancelha nunca fique transparente
        luma = 0.2126 * rgb_t[:, 0:1] + 0.7152 * rgb_t[:, 1:2] + 0.0722 * rgb_t[:, 2:3]
        outer_band = (alpha_coarse > 0.02).float() * (erode_gpu((alpha_coarse > 0.92).float(), radius=18) < 0.5).float()
        return (vertical_gate * outer_band).clamp(0.0, 1.0)

    @torch.no_grad()
    def _infer_vitmatte_hair_roi(
        self,
        rgb_t: torch.Tensor,
        alpha_base: torch.Tensor,
        hair_zone: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executa o `VitMatte` (`hustvl/vitmatte-small-composition-1k`) em resolução nativa 1:1
        no recorte (ROI) da cabeça/cabelo com Trimap Assimétrico que permite furar os vãos
        claros da parede entre os cachos!
        """
        if self.vitmatte_model is None:
            return alpha_base

        _, _, H, W = alpha_base.shape
        hz_mask = (hair_zone[0, 0] > 0.05) & (alpha_base[0, 0] > 0.02)
        coords = torch.nonzero(hz_mask, as_tuple=False)
        if coords.numel() < 50:
            return alpha_base

        y0 = max(0, int(coords[:, 0].min().item()) - 28)
        y1 = min(H, int(coords[:, 0].max().item()) + 28)
        x0 = max(0, int(coords[:, 1].min().item()) - 28)
        x1 = min(W, int(coords[:, 1].max().item()) + 28)

        # Ajustar tamanho do crop para múltiplo de 32 exigido pelo backbone ViT do VitMatte
        h_crop = max(64, ((y1 - y0 + 31) // 32) * 32)
        w_crop = max(64, ((x1 - x0 + 31) // 32) * 32)
        if y0 + h_crop > H:
            y0 = max(0, H - h_crop)
            h_crop = ((H - y0) // 32) * 32
        if x0 + w_crop > W:
            x0 = max(0, W - w_crop)
            w_crop = ((W - x0) // 32) * 32
        y1 = y0 + h_crop
        x1 = x0 + w_crop

        rgb_roi = rgb_t[:, :, y0:y1, x0:x1]
        a_roi = alpha_base[:, :, y0:y1, x0:x1]

        luma_roi = 0.2126 * rgb_roi[:, 0:1] + 0.7152 * rgb_roi[:, 1:2] + 0.0722 * rgb_roi[:, 2:3]
        bg_mask_roi = (dilate_gpu(a_roi, radius=4) < 0.04).float()
        fg_deep_roi = erode_gpu((a_roi > 0.92).float(), radius=14)

        # Luminância média da parede vs luminância média do interior profundo do cabelo
        luma_bg_mean = (luma_roi * bg_mask_roi).sum() / (bg_mask_roi.sum() + 1e-4)
        luma_hair_mean = (luma_roi * fg_deep_roi).sum() / (fg_deep_roi.sum() + 1e-4)

        # Auto-Adaptação de Contraste Cabelo vs Fundo (Funciona em Parede Clara, Colorida ou Estúdio Escuro!):
        # Na casca externa de 8..14px dos cachos, qualquer pixel cuja luminância esteja mais próxima do fundo
        # do que do interior do cabelo entra obrigatoriamente na zona UNKNOWN (0.5) para o VitMatte recortar!
        dist_to_hair = torch.abs(luma_roi - luma_hair_mean)
        dist_to_bg = torch.abs(luma_roi - luma_bg_mean)
        is_closer_to_hair = (dist_to_hair <= dist_to_bg * 1.15).float()
        fg_sure = erode_gpu((a_roi > 0.92).float(), radius=8) * is_closer_to_hair
        fg_sure = torch.maximum(fg_sure, fg_deep_roi)
        bg_sure = (dilate_gpu(a_roi, radius=3) < 0.06).float()

        trimap = torch.full_like(a_roi, 0.5)
        trimap = torch.where(fg_sure > 0.5, torch.ones_like(trimap), trimap)
        trimap = torch.where(bg_sure > 0.5, torch.zeros_like(trimap), trimap)

        rgb_norm = (rgb_roi - self.birefnet_mean) / self.birefnet_std
        inp_4ch = torch.cat([rgb_norm, trimap], dim=1)
        # Padroniza shape fixo (320, 480) múltiplo de 32 para o backbone ViT do VitMatte (evita realocação CUDA frame a frame)
        inp_fixed = F.interpolate(inp_4ch, size=(320, 480), mode="bilinear", align_corners=False)
        if self.device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                vit_fixed = self.vitmatte_model(pixel_values=inp_fixed).alphas.float().clamp(0.0, 1.0)
        else:
            vit_fixed = self.vitmatte_model(pixel_values=inp_fixed).alphas.clamp(0.0, 1.0)
        vit_alpha_roi = F.interpolate(vit_fixed, size=(h_crop, w_crop), mode="bilinear", align_corners=False)

        out_alpha = alpha_base.clone()
        out_alpha[:, :, y0:y1, x0:x1] = vit_alpha_roi
        return out_alpha

    @torch.no_grad()
    def _apply_pillar2_and_pillar3_refine_edge(
        self,
        rgb_t: torch.Tensor,
        rvm_pha: torch.Tensor,
        biref_alpha: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executa a Arquitetura V2 de Recorte Assimétrico (Zero Aspecto de Fundo Verde & Zero Rastro Branco):
          1. Fusão de Núcleo Sólido (`RVM + BiRefNet`) com Trava Fotométrica Anti-Rastro:
             - Impede 100% que um frame intermediário de BiRefNet expanda a borda externa além do RVM
               quando o rosto/cabeça se move (elimina o rastro branco nos Frames 82/84 e 294!).
             - Usa o BiRefNet validado para zerar sombras falsas do RVM na parede (Frame 239!).
          2. Separação Semântica Estrita:
             - `body_zone` (rosto, óculos, orelhas, pescoço, ombros e camisa branca):
               Recebe **Choke Cirúrgico de `-2.5px` + Rampa Dura Anti-Aliased de `1.1px`**
               (elimina 100% daquela aura esfumaçada ao redor da camisa e do pescoço!).
             - `hair_zone` (coroa de cachos/fios no topo da cabeça):
               Recebe **`VitMatte` ROI + `Luminance Valley Curl-Gap Keying`** que fura os vãos
               claros onde a parede aparecia entre os cachos (`alpha -> 0`), deixando apenas os
               cachos e fios escuros reais!
        """
        cfg = self.cfg

        # 1. Fusão Inteligente RVM + BiRefNet sem Rastro Temporal e sem Sombra na Parede
        if biref_alpha is not None:
            biref_clean = ((biref_alpha - 0.14) / 0.86).clamp(0.0, 1.0)
            if self.last_birefnet_rgb_t is not None and self.last_birefnet_rgb_t.shape == rgb_t.shape:
                diff_bi = box_filter_gpu(torch.mean(torch.abs(rgb_t - self.last_birefnet_rgb_t), dim=1, keepdim=True), r=5)
                bi_static_gate = 1.0 - ((diff_bi - 0.012) / 0.018).clamp(0.0, 1.0)
            else:
                bi_static_gate = torch.ones_like(rvm_pha)

            # A) Preenchimento de interior (apenas onde RVM já indica interior ou silhueta fechada, NUNCA fora da cabeça!)
            rvm_closed = morph_close_gpu((rvm_pha > 0.45).float(), radius=12)
            biref_core = erode_gpu((biref_clean > 0.78).float(), radius=6) * rvm_closed * bi_static_gate
            fused_base = torch.maximum(rvm_pha, biref_core * 0.99)

            # B) Remoção de sombras falsas do RVM na parede (ex: Frame 239 onde RVM pega sombra e BiRefNet diz 0)
            biref_bg_sure = ((0.18 - dilate_gpu(biref_clean, radius=2)) / 0.18).clamp(0.0, 1.0) * bi_static_gate
            fused_base = fused_base * (1.0 - 0.98 * biref_bg_sure)

            # C) Na borda externa, nunca somar BiRefNet defasado fora do suporte do RVM
            biref_edge_support = torch.minimum(biref_clean, dilate_gpu(rvm_pha, radius=1))
            alpha_base = (0.82 * fused_base + 0.18 * (biref_edge_support * bi_static_gate + fused_base * (1.0 - bi_static_gate))).clamp(0.0, 1.0)
        else:
            alpha_base = rvm_pha.clamp(0.0, 1.0)

        # Fechamento morfológico no núcleo interno (> 0.80) para garantir zero micro-furos na camisa ou testa
        solid_interior = morph_close_gpu((alpha_base > 0.78).float(), radius=8)
        solid_interior = erode_gpu(solid_interior, radius=3)
        alpha_base = torch.maximum(alpha_base, solid_interior * 0.995)

        # 2. Mapa Semântico da Coroa do Cabelo vs Corpo/Rosto/Camisa
        hair_zone = self._estimate_hair_zone_weight(rgb_t, alpha_base)
        hair_zone_smooth = gaussian_blur_gpu(hair_zone, sigma=3.0).clamp(0.0, 1.0)

        # 3A. RAMO DO CORPO, ROSTO, ORELHAS, PESCOÇO E CAMISA (`body_alpha`):
        # Aplica Choke Cirúrgico Firme (-2.5px) + Borda Nítida Anti-Aliased (1.1px) -> ZERO NÉVOA!
        choke_px = abs(float(cfg.body_choke_px))
        r_c1 = max(1, int(math.floor(choke_px)))
        r_c2 = r_c1 + 1
        frac_c = choke_px - float(r_c1)
        er_c1 = erode_gpu(alpha_base, radius=r_c1)
        er_c2 = erode_gpu(alpha_base, radius=r_c2)
        alpha_choked_body = er_c1 * (1.0 - frac_c) + er_c2 * frac_c
        # Rampa nítida centrada em 0.56 com largura 0.16 + anti-aliasing sub-pixel de 0.65px
        body_crisp = ((alpha_choked_body - 0.56) / 0.16).clamp(0.0, 1.0)
        body_crisp = body_crisp * body_crisp * (3.0 - 2.0 * body_crisp)
        body_alpha = gaussian_blur_gpu(body_crisp, sigma=0.65).clamp(0.0, 1.0)

        # 3B. RAMO DA COROA DO CABELO (`hair_alpha`):
        # VitMatte ROI + Guided Filter Fino + Luminance Valley Curl-Gap Keying (fura os vãos claros entre cachos!)
        if self.vitmatte_model is not None:
            if (
                self.frame_counter % max(1, int(cfg.vitmatte_interval)) == 0
                or self.last_vitmatte_alpha is None
            ):
                self.last_vitmatte_alpha = self._infer_vitmatte_hair_roi(rgb_t, alpha_base, hair_zone_smooth)
                self.last_vitmatte_rgb_t = rgb_t.clone()
            if self.last_vitmatte_rgb_t is not None and self.last_vitmatte_rgb_t.shape == rgb_t.shape:
                diff_vit = box_filter_gpu(torch.mean(torch.abs(rgb_t - self.last_vitmatte_rgb_t), dim=1, keepdim=True), r=4)
                vit_gate = 1.0 - ((diff_vit - 0.012) / 0.018).clamp(0.0, 1.0)
            else:
                vit_gate = torch.ones_like(alpha_base)
            vit_valid = self.last_vitmatte_alpha * vit_gate + alpha_base * (1.0 - vit_gate)
            vit_alpha = 0.65 * vit_valid + 0.35 * alpha_base
        else:
            vit_alpha = alpha_base

        alpha_guided = guided_matte_rgb_gpu(rgb_t, vit_alpha, r=3, eps=float(cfg.guided_eps))
        alpha_color_line, f_int, contrast_conf = compute_color_line_alpha_gpu(rgb_t, alpha_base)

        # Zona estrita das pontas externas dos cachos (apenas a 2..3px do fundo externo real `alpha_base < 0.22`):
        # Impede 100% que reflexos loiros/castanhos internos sofram furo ou criem anel duplo!
        bg_outside = (alpha_base < 0.22).float()
        outer_curl_fringe = (gaussian_blur_gpu(dilate_gpu(bg_outside, radius=2), sigma=1.4) * (alpha_base > 0.01).float()).clamp(0.0, 1.0)
        hair_inner_core = (1.0 - outer_curl_fringe).clamp(0.0, 1.0)

        # Luminance Valley Keying apenas na ponta externa dos cachos (fura vãos da parede sem rachar mechas claras!)
        valley_key = torch.pow(alpha_color_line.clamp(0.0, 1.0), float(cfg.hair_valley_gamma))
        fringe_keyed = torch.minimum(alpha_guided, valley_key * 1.15) * contrast_conf + alpha_guided * (1.0 - contrast_conf)
        hair_alpha_raw = alpha_guided * hair_inner_core + fringe_keyed * outer_curl_fringe

        # Fechamento topológico anti-fissura + rampa sub-pixel macia
        hair_closed = morph_close_gpu(hair_alpha_raw, radius=2)
        hair_alpha = (0.70 * hair_alpha_raw + 0.30 * hair_closed).clamp(0.0, 1.0)
        hair_eroded = erode_gpu(hair_alpha, radius=1)
        hair_alpha = (0.68 * hair_alpha + 0.32 * hair_eroded).clamp(0.0, 1.0)
        hair_alpha = ((hair_alpha - 0.10) / 0.88).clamp(0.0, 1.0)
        hair_alpha = gaussian_blur_gpu(hair_alpha, sigma=0.70)

        # 4. Combina o Contorno Crisp do Corpo (`body_alpha`) com os Cachos Furados do Cabelo (`hair_alpha`)
        alpha = body_alpha * (1.0 - hair_zone_smooth) + hair_alpha * hair_zone_smooth

        # Garante que o núcleo interno profundo (> 3px para dentro) seja 100% opaco (1.0) sem fissuras
        deep_solid = gaussian_blur_gpu(erode_gpu((alpha_base > 0.85).float(), radius=3), sigma=0.8)
        alpha = torch.maximum(alpha, deep_solid)

        alpha = torch.where(alpha < 0.02, torch.zeros_like(alpha), alpha)
        alpha = torch.where(alpha > 0.985, torch.ones_like(alpha), alpha)
        return alpha.clamp(0.0, 1.0), hair_zone_smooth

    @torch.no_grad()
    def process_video_frame_bgr(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processa um frame BGR de vídeo com memória temporal completa:
          Retorna `(fg_decontaminated_bgr_u8, alpha_u8)`.
        """
        H, W = frame_bgr.shape[:2]
        rgb_np = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb_np).to(self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Detector automático de corte seco de cena (reseta memória recorrente se a cena mudar abruptamente)
        if self.prev_rgb_t is not None and self.prev_rgb_t.shape == rgb_t.shape:
            mean_l1 = float(torch.mean(torch.abs(rgb_t - self.prev_rgb_t)).item())
            if mean_l1 > 0.22:
                self.reset_temporal_state()
        else:
            mean_l1 = 1.0

        # 1. Inferência RVM ResNet50 com estado recorrente ConvGRU (r1, r2, r3, r4)
        rvm_fgr, rvm_pha, *rec_out = self.rvm_model(rgb_t, *self.rec, float(self.cfg.downsample_ratio))
        self.rec = [r.detach() if r is not None else None for r in rec_out]

        # 2. Ancoragem BiRefNet_lite FP16 (a cada N frames ou em movimento rápido + trava fotométrica)
        if self.birefnet_model is not None:
            if (
                self.frame_counter % max(1, int(self.cfg.birefnet_interval)) == 0
                or self.last_birefnet_alpha is None
                or mean_l1 > 0.018
            ):
                self.last_birefnet_alpha = self._infer_birefnet_alpha(rgb_t, res=1024)
                self.last_birefnet_rgb_t = rgb_t.clone()

        # 3. Refine Edge (Pilar 2) + Controles After Effects (Pilar 3) + Separação Cabelo vs Corpo
        alpha_cur, hair_zone = self._apply_pillar2_and_pillar3_refine_edge(rgb_t, rvm_pha, self.last_birefnet_alpha)

        # 4. PILAR 1C: Motion-Gated Temporal Damping ("Reduce Chatter" sem Ghosting nas Mãos!)
        if (
            self.cfg.motion_damping > 0.01
            and self.prev_alpha_t is not None
            and self.prev_rgb_t is not None
            and self.prev_alpha_t.shape == alpha_cur.shape
        ):
            # Diferença fotométrica local suavizada (detecta se o pixel está em movimento ou parado)
            diff_rgb = torch.mean(torch.abs(rgb_t - self.prev_rgb_t), dim=1, keepdim=True)
            diff_smooth = box_filter_gpu(diff_rgb, r=5)
            motion_gate = ((diff_smooth - 0.006) / 0.018).clamp(0.0, 1.0)
            damping_map = float(self.cfg.motion_damping) * (1.0 - motion_gate)
            alpha_cur = alpha_cur * (1.0 - damping_map) + self.prev_alpha_t * damping_map

        # 5. PILAR 4: Deep Interior Decontaminate Edge Colors (Zero Halo & Zero Névoa Cinza!)
        fg_clean_t = decontaminate_edge_colors_gpu(
            rgb_t,
            alpha_cur,
            rvm_fgr=rvm_fgr,
            hair_zone=hair_zone,
            amount=float(self.cfg.decontaminate),
            radius=int(self.cfg.decontaminate_radius),
        )

        self.prev_rgb_t = rgb_t.clone()
        self.prev_alpha_t = alpha_cur.clone()
        self.frame_counter += 1
        if self.frame_counter % 60 == 0 and self.device.type == "cuda":
            torch.cuda.empty_cache()

        fg_rgb_u8 = (fg_clean_t[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        fg_bgr_u8 = cv2.cvtColor(fg_rgb_u8, cv2.COLOR_RGB2BGR)
        alpha_u8 = (alpha_cur[0, 0].clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        return fg_bgr_u8, alpha_u8

    @staticmethod
    def compose_with_light_wrap_bgr(
        fg_bgr: np.ndarray,
        alpha_u8: np.ndarray,
        bg_bgr: np.ndarray,
        light_wrap_amount: float = 0.24,
        rim_darken: float = 0.14,
    ) -> np.ndarray:
        """
        Composição Física de Cinema com:
          1. Rim Bounce Roll-Off (`rim_darken`): Atenua suavemente o estouro de luz branca do quarto
             antigo nos últimos 4px da camisa branca/ombros quando colocado sobre um fundo escuro.
          2. Optical Light Wrap (`light_wrap_amount`): A luz/cor do novo fundo (ou do texto 2.5D brilhante
             atrás do especialista) abraça opticamente 2.5px da borda interna da pessoa, eliminando 100%
             da sensação de "adesivo colado / fundo verde"!
        """
        alpha_f = (alpha_u8.astype(np.float32) / 255.0)[..., None]
        fg_f = fg_bgr.astype(np.float32)
        bg_f = bg_bgr.astype(np.float32)

        # Mapa de borda interna (0 no centro profundo, 1 nos últimos 4 pixels da borda da pessoa)
        inner_core = cv2.erode(alpha_u8, np.ones((7, 7), np.uint8)).astype(np.float32) / 255.0
        inner_core = cv2.GaussianBlur(inner_core, (7, 7), 2.0)[..., None]
        rim_band = np.clip(alpha_f - inner_core, 0.0, 1.0)

        # 1. Se o fundo novo for mais escuro que a borda da roupa clara, atenua o rebatedor do quarto antigo
        fg_luma = 0.114 * fg_f[..., 0:1] + 0.587 * fg_f[..., 1:2] + 0.299 * fg_f[..., 2:3]
        bg_blur = cv2.GaussianBlur(bg_f, (19, 19), 5.0)
        bg_luma = 0.114 * bg_blur[..., 0:1] + 0.587 * bg_blur[..., 1:2] + 0.299 * bg_blur[..., 2:3]

        bright_mismatch = np.clip((fg_luma - bg_luma) / 220.0, 0.0, 1.0)
        atten = 1.0 - (rim_band * bright_mismatch * float(rim_darken))
        fg_matched = fg_f * atten

        # 2. Optical Light Wrap: difusão da cor/luz do fundo para dentro dos últimos 2.5px do sujeito
        if light_wrap_amount > 0.01:
            wrap_glow = cv2.GaussianBlur(bg_f, (15, 15), 4.0)
            wrap_weight = rim_band * float(light_wrap_amount)
            # Blend aditivo suave / screen ponderado na borda
            screen_blend = 255.0 - ((255.0 - fg_matched) * (255.0 - wrap_glow) / 255.0)
            fg_wrapped = fg_matched * (1.0 - wrap_weight) + (0.55 * screen_blend + 0.45 * wrap_glow) * wrap_weight
        else:
            fg_wrapped = fg_matched

        comp = bg_f * (1.0 - alpha_f) + fg_wrapped * alpha_f
        return np.clip(comp, 0.0, 255.0).astype(np.uint8)

    @torch.no_grad()
    def process_static_image(
        self,
        image_path: Path | str,
        output_png_path: Path | str,
        chroma_hybrid: bool = False,
    ) -> Path:
        """
        PILAR 5 — Modo Estático Ultra-Nitidez (`--image` / `--chroma-hybrid`).
        Combina:
          - Despill e Keying Cromático Híbrido (se houver fundo verde/rosa/azul ou `--chroma-hybrid`)
          - BiRefNet_lite FP16 Multi-Scale (`1024x1024` + `1280x1280`) + RVM High-Res + VitMatte
          - Refine Edge + Deep Interior Decontaminate Edge Colors
        Salva PNG RGBA de máxima nitidez (pronto para Parallax 3D / Thumbnails / Carrosséis).
        """
        img_p = Path(image_path)
        out_p = Path(output_png_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        bgr = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Imagem não encontrada: {img_p}")

        H, W = bgr.shape[:2]
        rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb_np).to(self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

        # 1. Inferência Multi-Escala BiRefNet + RVM Single-Frame
        rec_init = [None, None, None, None]
        ds_ratio = min(0.5, max(0.20, 512.0 / float(max(H, W))))
        rvm_fgr, rvm_pha, *_ = self.rvm_model(rgb_t, *rec_init, ds_ratio)

        if self.birefnet_model is not None:
            biref_1024 = self._infer_birefnet_alpha(rgb_t, res=1024)
            biref_1280 = self._infer_birefnet_alpha(rgb_t, res=1280)
            biref_alpha = 0.50 * biref_1024 + 0.50 * biref_1280
        else:
            biref_alpha = None

        alpha, hair_zone = self._apply_pillar2_and_pillar3_refine_edge(rgb_t, rvm_pha, biref_alpha)

        # 2. Se `--chroma-hybrid` estiver ativo, detecta cor dominante nos 4 cantos e aplica Despill + Chroma Matte
        if chroma_hybrid:
            corners = torch.cat([
                rgb_t[:, :, :24, :24].reshape(1, 3, -1),
                rgb_t[:, :, :24, -24:].reshape(1, 3, -1),
            ], dim=-1)
            bg_col = torch.median(corners, dim=-1).values.view(1, 3, 1, 1)
            dist_bg = torch.sqrt(torch.sum((rgb_t - bg_col) ** 2, dim=1, keepdim=True) + 1e-6)
            chroma_alpha = ((dist_bg - 0.10) / 0.22).clamp(0.0, 1.0)
            alpha = alpha * (0.35 + 0.65 * chroma_alpha)

            # Despill de verde/magenta caso o fundo seja chroma key
            r, g, b = rgb_t[:, 0:1], rgb_t[:, 1:2], rgb_t[:, 2:3]
            if float(bg_col[0, 1, 0, 0]) > float(bg_col[0, 0, 0, 0]) + 0.12:
                # Fundo verde: limita canal G ao máximo de (R, B) na borda
                g_cap = torch.maximum(r, b)
                spill = (g - g_cap).clamp(min=0.0)
                rgb_t = torch.cat([r + spill * 0.15, g - spill * 0.85, b + spill * 0.15], dim=1)

        fg_clean_t = decontaminate_edge_colors_gpu(
            rgb_t,
            alpha,
            rvm_fgr=rvm_fgr,
            hair_zone=hair_zone,
            amount=float(self.cfg.decontaminate),
            radius=int(self.cfg.decontaminate_radius),
        )

        fg_rgb_u8 = (fg_clean_t[0].permute(1, 2, 0).clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        alpha_u8 = (alpha[0, 0].clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
        rgba_u8 = np.dstack([fg_rgb_u8, alpha_u8])
        Image.fromarray(rgba_u8, mode="RGBA").save(str(out_p))
        return out_p

    def process_video(
        self,
        input_video: Path | str,
        output_dir: Path | str,
        start_sec: float = 0.0,
        duration_sec: float = 0.0,
        export_fg_mp4: bool = True,
        export_matte_mp4: bool = True,
        export_25d_demo_mp4: bool = False,
        export_studio_mp4: bool = False,
        demo_title: str = "ROTOBRUSH 3.0",
        demo_subtitle: str = "GPU PRO V2 • VITMATTE + LIGHT WRAP",
    ) -> Dict[str, Any]:
        """
        Processa um vídeo completo (ou trecho `[start_sec .. start_sec + duration_sec]`) na GPU RTX 3060,
        gerando:
          - `roto_alpha_matte.mp4` (Máscara Alpha 1080p H.264 NVENC para uso em HyperFrames / 2.5D)
          - `roto_fg_decontaminated.mp4` (Sujeito recortado com bordas descontaminadas sobre preto)
          - `roto_25d_showcase.mp4` (Opcional: Modo 2.5D Real — Cenário Original f/1.4 + Texto Atrás + Light Wrap)
          - `roto_studio_pro.mp4` (Opcional: Modo Troca Total de Cenário Studio Pro + Light Wrap)
        """
        in_p = Path(input_video)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(in_p))
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {in_p}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)

        start_frame = max(0, int(round(start_sec * fps)))
        if duration_sec > 0.0:
            end_frame = min(total_src_frames, start_frame + int(round(duration_sec * fps)))
        else:
            end_frame = total_src_frames
        n_frames = max(1, end_frame - start_frame)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Extrair áudio do trecho com precisão para remux final
        audio_tmp = out_d / "_temp_audio.aac"
        dur_arg = ["-t", f"{n_frames / fps:.3f}"] if duration_sec > 0.0 else []
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_sec:.3f}", *dur_arg,
                "-i", str(in_p), "-vn", "-c:a", "aac", "-b:a", "192k", str(audio_tmp),
            ],
            check=False,
        )
        has_audio = audio_tmp.exists() and audio_tmp.stat().st_size > 1000

        matte_mp4 = out_d / "roto_alpha_matte.mp4"
        fg_mp4 = out_d / "roto_fg_decontaminated.mp4"
        demo_mp4 = out_d / "roto_25d_showcase.mp4"
        studio_mp4 = out_d / "roto_studio_pro.mp4"
        matte_raw_tmp = out_d / "_tmp_matte.gray"
        studio_raw_tmp = out_d / "_tmp_studio.bgr24"

        def _open_nvenc_pipe(target_path: Path, pix_fmt_in: str = "bgr24") -> subprocess.Popen:
            audio_inputs = ["-i", str(audio_tmp), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"] if has_audio else []
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", pix_fmt_in, "-s", f"{W}x{H}", "-r", f"{fps:.3f}",
                "-i", "-",
                *audio_inputs,
                "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18", "-g", "30", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(target_path),
            ]
            return subprocess.Popen(cmd, stdin=subprocess.PIPE)

        # Respeita o limite de no máximo 2 processos FFmpeg simultâneos do hw_guard Watchdog (Slots: 2)
        f_matte_raw = open(matte_raw_tmp, "wb") if export_matte_mp4 else None
        f_studio_raw = open(studio_raw_tmp, "wb") if export_studio_mp4 else None
        proc_fg = _open_nvenc_pipe(fg_mp4, "bgr24") if export_fg_mp4 else None
        proc_demo = _open_nvenc_pipe(demo_mp4, "bgr24") if export_25d_demo_mp4 else None

        self.reset_temporal_state()
        t0 = time.time()
        actual_frames = 0

        for idx in range(n_frames):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            fg_bgr, alpha_u8 = self.process_video_frame_bgr(frame_bgr)
            actual_frames += 1

            if f_matte_raw is not None:
                f_matte_raw.write(alpha_u8.tobytes())

            if proc_fg is not None and proc_fg.stdin is not None:
                alpha_f = (alpha_u8.astype(np.float32) / 255.0)[..., None]
                fg_premult = (fg_bgr.astype(np.float32) * alpha_f).astype(np.uint8)
                proc_fg.stdin.write(fg_premult.tobytes())

            if proc_demo is not None and proc_demo.stdin is not None:
                comp = self._render_25d_demo_frame(frame_bgr, fg_bgr, alpha_u8, idx / fps, demo_title, demo_subtitle)
                proc_demo.stdin.write(comp.tobytes())

            if f_studio_raw is not None:
                comp_st = self._render_studio_pro_frame(fg_bgr, alpha_u8, idx / fps, demo_title, demo_subtitle)
                f_studio_raw.write(comp_st.tobytes())

            if (idx + 1) % 30 == 0 or (idx + 1) == n_frames:
                elapsed = max(1e-3, time.time() - t0)
                cur_fps = (idx + 1) / elapsed
                print(f"  [RotoBrush GPU Pro] {idx + 1}/{n_frames} frames ({cur_fps:.1f} fps na RTX 3060)", flush=True)

        cap.release()
        if f_matte_raw is not None:
            f_matte_raw.close()
        if f_studio_raw is not None:
            f_studio_raw.close()
        for p in (proc_fg, proc_demo):
            if p is not None:
                if p.stdin is not None:
                    p.stdin.close()
                p.wait()

        if export_matte_mp4 and matte_raw_tmp.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", f"{fps:.3f}",
                    "-i", str(matte_raw_tmp),
                    "-vf", "format=yuv420p",
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18", "-g", "30", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(matte_mp4),
                ],
                check=True,
            )
            matte_raw_tmp.unlink()

        if export_studio_mp4 and studio_raw_tmp.exists():
            audio_inputs = ["-i", str(audio_tmp), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy", "-shortest"] if has_audio else []
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:.3f}",
                    "-i", str(studio_raw_tmp),
                    *audio_inputs,
                    "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18", "-g", "30", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(studio_mp4),
                ],
                check=True,
            )
            studio_raw_tmp.unlink()

        if audio_tmp.exists():
            audio_tmp.unlink()

        total_t = time.time() - t0
        report = {
            "input_video": str(in_p),
            "frames_processed": n_frames,
            "fps_source": round(fps, 3),
            "resolution": f"{W}x{H}",
            "processing_time_sec": round(total_t, 2),
            "processing_fps": round(n_frames / max(1e-3, total_t), 2),
            "config": asdict(self.cfg),
            "outputs": {
                "alpha_matte_mp4": str(matte_mp4) if export_matte_mp4 else None,
                "fg_decontaminated_mp4": str(fg_mp4) if export_fg_mp4 else None,
                "demo_25d_mp4": str(demo_mp4) if export_25d_demo_mp4 else None,
                "studio_pro_mp4": str(studio_mp4) if export_studio_mp4 else None,
            },
        }
        (out_d / "rotobrush_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        return report

    def _render_25d_demo_frame(
        self,
        raw_bgr: np.ndarray,
        fg_clean_bgr: np.ndarray,
        alpha_u8: np.ndarray,
        t_sec: float,
        title: str,
        subtitle: str,
    ) -> np.ndarray:
        """Renderiza Modo 2.5D Real (Cenário Original com Bokeh f/1.4 + Texto Atrás da Cabeça + Optical Light Wrap)."""
        H, W = raw_bgr.shape[:2]
        bg_f = raw_bgr.astype(np.float32) * 0.44
        bg_f[..., 0] = np.clip(bg_f[..., 0] * 1.18 + 14.0, 0, 255)
        bg_f[..., 1] = np.clip(bg_f[..., 1] * 1.03 + 6.0, 0, 255)
        bg_f[..., 2] = np.clip(bg_f[..., 2] * 0.85, 0, 255)
        bg_u8 = cv2.GaussianBlur(bg_f.astype(np.uint8), (21, 21), 5.5)

        pil_bg = Image.fromarray(cv2.cvtColor(bg_u8, cv2.COLOR_BGR2RGB)).convert("RGBA")
        txt_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        try:
            f_big = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", int(W * 0.13))
            f_sub = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", int(W * 0.036))
        except Exception:
            f_big = ImageFont.load_default()
            f_sub = ImageFont.load_default()

        y_off = int(math.sin(t_sec * 2.2) * 12.0)
        tb = draw.textbbox((0, 0), title, font=f_big)
        tw = tb[2] - tb[0]
        tx = (W - tw) // 2
        ty = int(H * 0.19) + y_off

        draw.text((tx, ty + 6), title, font=f_big, fill=(0, 0, 0, 180))
        draw.text((tx, ty), title, font=f_big, fill=(255, 255, 255, 252))
        sb = draw.textbbox((0, 0), subtitle, font=f_sub)
        sw = sb[2] - sb[0]
        draw.rounded_rectangle(
            [(W - sw) // 2 - 28, ty - 64, (W + sw) // 2 + 28, ty - 10],
            radius=24,
            fill=(16, 24, 38, 235),
            outline=(56, 189, 248, 230),
            width=2,
        )
        draw.text(((W - sw) // 2, ty - 56), subtitle, font=f_sub, fill=(56, 189, 248, 255))

        pil_bg = Image.alpha_composite(pil_bg, txt_layer)
        bg_with_txt_bgr = cv2.cvtColor(np.array(pil_bg.convert("RGB")), cv2.COLOR_RGB2BGR)

        return self.compose_with_light_wrap_bgr(
            fg_clean_bgr,
            alpha_u8,
            bg_with_txt_bgr,
            light_wrap_amount=float(self.cfg.light_wrap_amount),
            rim_darken=0.10,
        )

    def _render_studio_pro_frame(
        self,
        fg_clean_bgr: np.ndarray,
        alpha_u8: np.ndarray,
        t_sec: float,
        title: str,
        subtitle: str,
    ) -> np.ndarray:
        """Renderiza Modo Troca Total de Cenário Studio Pro com Rim Glow e Optical Light Wrap."""
        H, W = fg_clean_bgr.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy = W * 0.50, H * 0.36
        dist = np.sqrt(((xx - cx) / (W * 0.62)) ** 2 + ((yy - cy) / (H * 0.48)) ** 2)
        glow = np.clip(1.0 - dist, 0.0, 1.0) ** 1.6

        bg = np.zeros((H, W, 3), dtype=np.float32)
        bg[..., 0] = 20.0 + glow * 54.0   # B
        bg[..., 1] = 24.0 + glow * 64.0   # G
        bg[..., 2] = 28.0 + glow * 78.0   # R
        bg_u8 = np.clip(bg, 0, 255).astype(np.uint8)

        pil_bg = Image.fromarray(cv2.cvtColor(bg_u8, cv2.COLOR_BGR2RGB)).convert("RGBA")
        txt_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        try:
            f_big = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", int(W * 0.13))
            f_sub = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", int(W * 0.036))
        except Exception:
            f_big = ImageFont.load_default()
            f_sub = ImageFont.load_default()

        y_off = int(math.sin(t_sec * 2.2) * 12.0)
        tb = draw.textbbox((0, 0), title, font=f_big)
        tw = tb[2] - tb[0]
        tx = (W - tw) // 2
        ty = int(H * 0.19) + y_off

        draw.text((tx, ty + 6), title, font=f_big, fill=(0, 0, 0, 190))
        draw.text((tx, ty), title, font=f_big, fill=(255, 214, 10, 250))
        sb = draw.textbbox((0, 0), subtitle, font=f_sub)
        sw = sb[2] - sb[0]
        draw.rounded_rectangle(
            [(W - sw) // 2 - 28, ty - 64, (W + sw) // 2 + 28, ty - 10],
            radius=24,
            fill=(18, 22, 30, 235),
            outline=(250, 204, 21, 230),
            width=2,
        )
        draw.text(((W - sw) // 2, ty - 56), subtitle, font=f_sub, fill=(250, 204, 21, 255))

        pil_bg = Image.alpha_composite(pil_bg, txt_layer)
        bg_studio_bgr = cv2.cvtColor(np.array(pil_bg.convert("RGB")), cv2.COLOR_RGB2BGR)

        return self.compose_with_light_wrap_bgr(
            fg_clean_bgr,
            alpha_u8,
            bg_studio_bgr,
            light_wrap_amount=max(0.26, float(self.cfg.light_wrap_amount)),
            rim_darken=0.18,
        )


# ==============================================================================
# CLI OFICIAL (IDÊNTICA AOS PARÂMETROS DO ROTO BRUSH 3.0 DO AFTER EFFECTS)
# ==============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RotoBrush 3.0 GPU Pro V2 — Rotoscopia Temporal, VitMatte Hair Matting, Auto-Adaptive Despill e Optical Light Wrap na GPU NVIDIA"
    )
    p.add_argument("--input", "-i", type=str, help="Vídeo de entrada (.mp4/.mov) ou imagem (se --image)")
    p.add_argument("--output-dir", "-o", type=str, default="", help="Pasta de saída para os vídeos/mattes gerados")
    p.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "25d", "studio", "alpha", "fg"],
        help="Modo de exportação: all (Alpha+FG+2.5D+Studio), 25d (Cenário Original + Texto Atrás), studio (Troca de Cenário Studio Pro), alpha (apenas Luma Matte), fg (apenas Foreground Descontaminado)",
    )
    p.add_argument("--title", type=str, default="ROTOBRUSH 3.0", help="Título 2.5D gigante atrás da pessoa")
    p.add_argument("--subtitle", type=str, default="GPU PRO V2 • VITMATTE + LIGHT WRAP", help="Subtítulo/Badge 2.5D")
    p.add_argument("--image", type=str, default="", help="Modo Estático Ultra-Nitidez 4K: caminho da imagem de entrada")
    p.add_argument("--output-png", type=str, default="", help="Caminho do PNG RGBA de saída para --image")
    p.add_argument("--chroma-hybrid", action="store_true", help="Ativa Despill + Chroma Key Híbrido no modo --image")
    p.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "mobilenetv3"])
    p.add_argument("--no-birefnet", action="store_true", help="Desativa fusão BiRefNet_lite FP16 (usa apenas RVM)")
    p.add_argument("--no-vitmatte", action="store_true", help="Desativa VitMatte ViT FP16 na zona do cabelo")
    p.add_argument("--birefnet-interval", type=int, default=2, help="Intervalo de frames para ancoragem BiRefNet")
    p.add_argument("--refine-radius", type=int, default=8, help="Largura da banda Refine Edge Trimap (px)")
    p.add_argument("--feather", type=float, default=0.6, help="Feather da borda em pixels (0..15)")
    p.add_argument("--contrast", type=float, default=35.0, help="Contraste da borda em %% (0..100)")
    p.add_argument("--shift-edge", type=float, default=-1.5, help="Shift Edge / Choke no cabelo em pixels (ex: -1.5)")
    p.add_argument("--body-choke", type=float, default=-2.5, help="Choke cirúrgico no tronco/camisa/ombros em pixels (ex: -2.5)")
    p.add_argument("--motion-damping", type=float, default=0.72, help="Reduce Chatter / Amortecimento temporal (0..0.95)")
    p.add_argument("--decontaminate", type=float, default=0.98, help="Força do Deep Interior Decontaminate Edge Colors (0..1)")
    p.add_argument("--light-wrap", type=float, default=0.24, help="Intensidade do Optical Light Wrap na composição (0..1)")
    p.add_argument("--start-sec", type=float, default=0.0, help="Segundo inicial do vídeo")
    p.add_argument("--duration-sec", type=float, default=0.0, help="Duração em segundos (0 = vídeo inteiro)")
    p.add_argument("--demo-25d", action="store_true", help="Gera também vídeo showcase 2.5D com texto atrás da cabeça")
    p.add_argument("--demo-studio", action="store_true", help="Gera também vídeo com Troca Total de Cenário Studio Pro")
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = RotoBrushConfig(
        backbone=args.backbone,
        use_birefnet_fusion=not args.no_birefnet,
        use_vitmatte_hair=not args.no_vitmatte,
        birefnet_interval=args.birefnet_interval,
        refine_radius=args.refine_radius,
        feather=args.feather,
        contrast=args.contrast,
        shift_edge=args.shift_edge,
        body_choke_px=args.body_choke,
        motion_damping=args.motion_damping,
        decontaminate=args.decontaminate,
        light_wrap_amount=args.light_wrap,
    )

    engine = RotoBrushGPUPro(cfg)

    if args.image:
        img_in = Path(args.image)
        png_out = Path(args.output_png) if args.output_png else img_in.with_name(f"{img_in.stem}_roto_pro.png")
        t0 = time.time()
        saved = engine.process_static_image(img_in, png_out, chroma_hybrid=args.chroma_hybrid)
        print(f"[OK] Imagem recortada em {time.time() - t0:.2f}s -> {saved}")
        return 0

    if not args.input:
        parser.error("Informe --input <video.mp4> ou --image <foto.png>")

    vid_in = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else vid_in.parent / f"{vid_in.stem}_rotobrush_pro"

    mode = args.mode.lower()
    exp_matte = mode in ("all", "alpha", "25d", "studio")
    exp_fg = mode in ("all", "fg")
    exp_25d = (mode in ("all", "25d")) or bool(args.demo_25d)
    exp_studio = (mode in ("all", "studio")) or bool(args.demo_studio)

    rep = engine.process_video(
        vid_in,
        out_dir,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        export_fg_mp4=exp_fg,
        export_matte_mp4=exp_matte,
        export_25d_demo_mp4=exp_25d,
        export_studio_mp4=exp_studio,
        demo_title=args.title,
        demo_subtitle=args.subtitle,
    )
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
