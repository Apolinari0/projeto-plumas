"""Extração de quadros de vídeo com horário em UTC."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import pandas as pd


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_utc(s: str) -> datetime:
    """Aceita '2026-11-20T15:30:00Z' ou com offset; devolve datetime com tz UTC."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("Informe o horário com fuso (Z ou +00:00): " + s)
    return dt.astimezone(timezone.utc)


def extract_frames(video_path: str | Path, out_dir: str | Path, every_s: float,
                   seq_id: str, t0_utc: str) -> pd.DataFrame:
    """Salva um quadro a cada `every_s` segundos e devolve a tabela de quadros.

    Colunas: frame (nome do arquivo), sequence, frame_idx, t_s, t_utc.
    O vídeo original não é modificado.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError("Não foi possível abrir o vídeo: " + str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        raise ValueError("FPS inválido no vídeo; informe manualmente.")
    step = max(1, int(round(every_s * fps)))
    t0 = parse_utc(t0_utc)

    rows, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t_s = idx / fps
            name = f"{seq_id}_{idx:06d}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            rows.append({
                "frame": name, "sequence": seq_id, "frame_idx": idx,
                "t_s": round(t_s, 3),
                "t_utc": (t0 + timedelta(seconds=t_s)).isoformat(timespec="milliseconds"),
            })
        idx += 1
    cap.release()
    return pd.DataFrame(rows)
