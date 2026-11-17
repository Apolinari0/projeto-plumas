"""Leitura do config.yaml, caminhos das etapas e registro de experimentos."""
from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REQUIRED_METADATA = [
    "source",                 # origem (URL, órgão, autor, "gravado por nós")
    "capture_datetime_utc",   # ISO 8601 em UTC, ex.: 2026-11-20T15:30:00Z
    "original_timezone",      # fuso original, se conhecido (ex.: America/Sao_Paulo)
    "latitude",
    "longitude",
    "location_uncertainty_m", # precisão/incerteza da localização
    "camera",
    "resolution",             # ex.: "1920x1080"
    "fps",
    "camera_azimuth_deg",     # para onde a câmera aponta (0=N, 90=L), sentido horário
    "capture_conditions",     # luz, distância aproximada, fundo, clima
]


def find_repo_root(start: str | os.PathLike | None = None) -> Path:
    """Sobe a partir de `start` até achar config.yaml."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "config.yaml").exists():
            return cand
    raise FileNotFoundError("config.yaml não encontrado a partir de " + str(p))


def load_config(path: str | os.PathLike | None = None) -> dict:
    root = find_repo_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["drive_root"] = os.environ.get("PLUMAS_DRIVE_ROOT", cfg["drive_root"])
    cfg["_repo_root"] = str(root)
    return cfg


def stage_dir(cfg: dict, stage: str) -> Path:
    """Pasta de dados da etapa (no Drive). Cria se não existir."""
    d = Path(cfg["drive_root"]) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _git_hash(repo_root: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def log_experiment(cfg: dict, stage: str, record: dict) -> Path:
    """Acrescenta uma linha JSON em <drive_root>/runs.jsonl.

    Guarda data, etapa, versão do código (commit) e os campos informados
    (parâmetros, métricas, caminhos de pesos/saídas, decisões tomadas).
    """
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "git_commit": _git_hash(cfg["_repo_root"]),
        **record,
    }
    out = Path(cfg["drive_root"]) / "runs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    return out


def validate_metadata(meta: dict) -> list[str]:
    """Retorna a lista de campos obrigatórios ausentes ou vazios."""
    missing = []
    for k in REQUIRED_METADATA:
        v = meta.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(k)
    return missing
