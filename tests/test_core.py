import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plumas import config as C
from plumas import features as F
from plumas import split as S
from plumas import tracking as T
from plumas import video as V
from plumas import wind as W


# ---------- vento ----------
def test_wind_from_direction_cardinals():
    assert W.wind_from_deg(1, 0) == pytest.approx(270)   # vento DE oeste (sopra para leste)
    assert W.wind_from_deg(-1, 0) == pytest.approx(90)   # DE leste
    assert W.wind_from_deg(0, 1) == pytest.approx(180)   # DE sul (sopra para norte)
    assert W.wind_from_deg(0, -1) == pytest.approx(0)    # DE norte
    assert W.wind_to_deg(1, 0) == pytest.approx(90)


def test_lateral_component():
    # câmera olhando para o norte, vento para leste -> empurra para a direita
    assert W.lateral_wind_component(5, 0, 0) == pytest.approx(5)
    # câmera olhando para o leste, vento para o norte -> esquerda da imagem
    assert W.lateral_wind_component(0, 5, 90) == pytest.approx(-5)
    # vento ao longo da visada não tem componente lateral
    assert W.lateral_wind_component(0, 5, 0) == pytest.approx(0, abs=1e-9)


def test_sign_agreement():
    assert W.sign_agreement(10, 2) == "concorda"
    assert W.sign_agreement(-10, 2) == "discorda"
    assert W.sign_agreement(0.2, 2) == "indefinido"


# ---------- atributos da máscara ----------
def test_mask_features_rectangle():
    m = np.zeros((100, 200), np.uint8)
    m[20:40, 50:110] = 1          # 20 linhas x 60 colunas
    f = F.mask_features(m)
    assert f["area_px"] == 20 * 60
    assert f["width_px"] == 60 and f["height_px"] == 20
    assert f["aspect_ratio"] == pytest.approx(3.0)
    assert f["cx"] == pytest.approx(79.5) and f["cy"] == pytest.approx(29.5)
    assert f["frac_image"] == pytest.approx(1200 / 20000)
    assert abs(f["orientation_deg"]) < 1.0       # alongado na horizontal


def test_mask_features_empty():
    assert F.mask_features(np.zeros((10, 10))) is None


def test_orientation_diagonals():
    down_right = np.zeros((100, 100), np.uint8)
    cv2.line(down_right, (10, 10), (90, 90), 1, 5)   # "\" na imagem (y para baixo)
    up_right = np.zeros((100, 100), np.uint8)
    cv2.line(up_right, (10, 90), (90, 10), 1, 5)     # "/"
    assert F.mask_features(down_right)["orientation_deg"] == pytest.approx(45, abs=3)
    assert F.mask_features(up_right)["orientation_deg"] == pytest.approx(-45, abs=3)


def test_vertical_orientation():
    m = np.zeros((100, 100), np.uint8)
    m[10:90, 45:55] = 1
    assert abs(F.mask_features(m)["orientation_deg"]) == pytest.approx(90, abs=1)


def test_iou_dice():
    a = np.zeros((10, 10)); a[:5] = 1
    b = np.zeros((10, 10)); b[:5, :5] = 1
    r = F.iou_dice(a, b)
    assert r["iou"] == pytest.approx(0.5)
    assert r["dice"] == pytest.approx(2 * 25 / 75)


# ---------- tracking ----------
def test_iou_and_associate():
    assert T.iou_boxes((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert T.iou_boxes((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    prev = [(0, 0, 10, 10), (50, 50, 60, 60)]
    cur = [(52, 52, 62, 62), (1, 1, 11, 11), (200, 200, 210, 210)]
    pairs = sorted(T.associate(prev, cur, 0.3))
    assert pairs == [(0, 1), (1, 0)]


def test_gaps_and_displacement():
    assert T.find_gaps([0, 1, 2, 10, 11], max_gap_s=5) == [2]
    assert T.apparent_displacement((10, 20), (13, 18)) == (3.0, -2.0)


# ---------- divisão sem vazamento ----------
def _df(n_seq=6, per=10):
    return pd.DataFrame({
        "frame": [f"s{s}_{i}" for s in range(n_seq) for i in range(per)],
        "sequence": [f"s{s}" for s in range(n_seq) for _ in range(per)],
        "t_s": [i for _ in range(n_seq) for i in range(per)],
    })


def test_group_split_no_leakage():
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    out = S.group_split(_df(), "sequence", ratios, seed=1)
    S.check_no_leakage(out, "sequence")
    assert set(out["split"]) == {"train", "val", "test"}
    again = S.group_split(_df(), "sequence", ratios, seed=1)
    assert (out["split"] == again["split"]).all()   # determinístico


def test_group_split_needs_groups():
    with pytest.raises(ValueError):
        S.group_split(_df(n_seq=2), "sequence", {"train": .7, "val": .15, "test": .15}, 0)


def test_leakage_detected():
    out = S.group_split(_df(), "sequence", {"train": .7, "val": .15, "test": .15}, 1)
    out.loc[out.index[0], "split"] = "test" if out.loc[out.index[0], "split"] != "test" else "train"
    with pytest.raises(AssertionError):
        S.check_no_leakage(out, "sequence")


def test_block_split_has_gaps_and_order():
    df = _df(n_seq=1, per=100)
    out = S.block_split(df, "t_s", {"train": .7, "val": .15, "test": .15}, gap_frames=10)
    order = [s for s in out["split"].tolist()]
    # nunca 'train' depois de 'val'/'test', nem 'val' depois de 'test'
    first = {s: order.index(s) for s in ("train", "val", "test")}
    last = {s: len(order) - 1 - order[::-1].index(s) for s in ("train", "val", "test")}
    assert last["train"] < first["val"] < last["val"] < first["test"]
    assert (out["split"] == "gap").sum() == 20


# ---------- metadados ----------
def test_validate_metadata():
    missing = C.validate_metadata({"source": "x", "latitude": -23.5})
    assert "capture_datetime_utc" in missing and "source" not in missing
    full = {k: "v" for k in C.REQUIRED_METADATA}
    full["latitude"] = 0.0
    assert C.validate_metadata(full) == []


# ---------- extração de quadros ----------
def test_extract_frames(tmp_path):
    vid = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(50):                       # 5 s a 10 fps
        vw.write(np.full((48, 64, 3), i * 4, np.uint8))
    vw.release()
    df = V.extract_frames(vid, tmp_path / "fr", every_s=1.0, seq_id="seqA",
                          t0_utc="2026-11-20T15:30:00Z")
    assert len(df) == 5
    assert df["t_s"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert df["t_utc"].iloc[1].startswith("2026-11-20T15:30:01")
    assert (tmp_path / "fr" / df["frame"].iloc[0]).exists()
    assert len(V.sha256_file(vid)) == 64
    with pytest.raises(ValueError):
        V.parse_utc("2026-11-20T15:30:00")   # sem fuso


# ---------- config ----------
def test_load_config(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setenv("PLUMAS_DRIVE_ROOT", "/tmp/plumas_test")
    cfg = C.load_config()
    assert cfg["drive_root"] == "/tmp/plumas_test"
    assert cfg["class_names"] == ["pluma"]
    d = C.stage_dir(cfg, "01_dados")
    assert d.exists()
    p = C.log_experiment(cfg, "01_dados", {"nota": "teste"})
    assert p.exists()
