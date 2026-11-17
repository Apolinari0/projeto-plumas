"""Atributos visuais (aparentes, em pixels) de uma máscara de pluma.

Todos os valores descrevem a PROJEÇÃO da pluma na imagem. Sem calibração da
câmera, não são medidas físicas (metros, m/s).
"""
from __future__ import annotations

import math

import numpy as np
from skimage.measure import regionprops


def mask_features(mask: np.ndarray) -> dict | None:
    """Atributos de uma máscara binária (H, W). Devolve None se estiver vazia.

    - area_px: área aparente em pixels
    - width_px / height_px: largura e altura da caixa envolvente da máscara
    - aspect_ratio: largura / altura
    - cx, cy: centroide (u, v) em coordenadas de imagem (x para a direita, y para baixo)
    - orientation_deg: ângulo do eixo principal em relação ao eixo horizontal da
      imagem, em graus, no intervalo (-90, 90]; positivo = inclinado para baixo
      no sentido horário visto na imagem (y para baixo)
    - frac_image: fração da imagem ocupada pela máscara
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    h, w = m.shape
    props = regionprops(m)[0]
    min_r, min_c, max_r, max_c = props.bbox
    # skimage: orientation = ângulo entre o eixo das linhas (vertical) e o eixo
    # principal, em (-pi/2, pi/2]. Convertemos para ângulo em relação à horizontal.
    theta_from_vertical = props.orientation
    ang = math.degrees(theta_from_vertical)
    # eixo vertical -> 90°; convertemos para referência horizontal (x para a direita, y para baixo)
    ang_h = 90.0 - ang
    if ang_h > 90:
        ang_h -= 180
    elif ang_h <= -90:
        ang_h += 180
    return {
        "area_px": int(props.area),
        "width_px": int(max_c - min_c),
        "height_px": int(max_r - min_r),
        "aspect_ratio": float((max_c - min_c) / max(1, (max_r - min_r))),
        "cx": float(props.centroid[1]),
        "cy": float(props.centroid[0]),
        "orientation_deg": float(ang_h),
        "frac_image": float(props.area / (h * w)),
    }


def iou_dice(pred: np.ndarray, gt: np.ndarray) -> dict:
    """IoU e Dice entre duas máscaras binárias."""
    p = np.asarray(pred) > 0
    g = np.asarray(gt) > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    s = p.sum() + g.sum()
    return {
        "iou": float(inter / union) if union else 1.0,
        "dice": float(2 * inter / s) if s else 1.0,
    }
