"""Associação de detecções entre quadros consecutivos (rastreamento simples)."""
from __future__ import annotations

import numpy as np


def iou_boxes(a, b) -> float:
    """IoU entre caixas (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def associate(prev_boxes, cur_boxes, iou_threshold: float = 0.2):
    """Associação gulosa por IoU. Devolve lista de pares (i_prev, j_cur).

    Detecções sem par (nascimentos/mortes) ficam de fora; quem chama decide
    o que fazer com elas.
    """
    cand = []
    for i, p in enumerate(prev_boxes):
        for j, c in enumerate(cur_boxes):
            v = iou_boxes(p, c)
            if v >= iou_threshold:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used_i, used_j, pairs = set(), set(), []
    for _, i, j in cand:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        pairs.append((i, j))
    return pairs


def find_gaps(times_s, max_gap_s: float):
    """Índices k tais que times_s[k+1] - times_s[k] > max_gap_s.

    Nesses pontos a trajetória deve ser interrompida: não se interpola
    intervalos longos sem observação.
    """
    t = np.asarray(times_s, dtype=float)
    return [int(k) for k in np.where(np.diff(t) > max_gap_s)[0]]


def apparent_displacement(c_t, c_t1):
    """Δc = c(t+1) - c(t), em pixels (u para a direita, v para baixo)."""
    return (float(c_t1[0] - c_t[0]), float(c_t1[1] - c_t[1]))
