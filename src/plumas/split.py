"""Divisão treino/validação/teste sem vazamento entre quadros parecidos."""
from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def group_split(df: pd.DataFrame, group_col: str, ratios: dict, seed: int) -> pd.DataFrame:
    """Divide por GRUPO (sequência, local ou data): todos os quadros de um
    grupo ficam no mesmo conjunto. Requer pelo menos 3 grupos.
    """
    groups = np.array(sorted(df[group_col].unique()))
    if len(groups) < 3:
        raise ValueError(
            f"Só há {len(groups)} grupo(s) em '{group_col}'. Use block_split() "
            "para uma única sequência ou grave mais sequências."
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_val = max(1, int(round(n * ratios["val"])))
    n_test = max(1, int(round(n * ratios["test"])))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("Grupos insuficientes para as proporções pedidas.")
    assign = {}
    for g in groups[:n_train]:
        assign[g] = "train"
    for g in groups[n_train:n_train + n_val]:
        assign[g] = "val"
    for g in groups[n_train + n_val:]:
        assign[g] = "test"
    out = df.copy()
    out["split"] = out[group_col].map(assign)
    return out


def block_split(df: pd.DataFrame, order_col: str, ratios: dict, gap_frames: int) -> pd.DataFrame:
    """Para uma única sequência: blocos contíguos no tempo (treino, val, teste)
    com `gap_frames` quadros descartados (split='gap') entre eles.
    """
    out = df.sort_values(order_col).reset_index(drop=True).copy()
    n = len(out)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    labels = ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    out["split"] = labels
    # marca como 'gap' os quadros em torno de cada fronteira
    for boundary in (n_train, n_train + n_val):
        lo = max(0, boundary - gap_frames // 2)
        hi = min(n, boundary + (gap_frames - gap_frames // 2))
        out.loc[lo:hi - 1, "split"] = "gap"
    return out


def check_no_leakage(df: pd.DataFrame, group_col: str) -> None:
    """Garante que nenhum grupo aparece em mais de um conjunto."""
    sub = df[df["split"].isin(SPLITS)]
    bad = sub.groupby(group_col)["split"].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise AssertionError("Vazamento: grupos em mais de um conjunto: " + ", ".join(map(str, bad.index)))
