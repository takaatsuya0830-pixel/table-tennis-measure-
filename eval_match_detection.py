"""
試合映像でのAI検出性能を評価する  ── 2026-08-04

BlurBallデータセットの正解アノテーション(ボール中心座標)を基準に、
学習したモデルが試合の引き映像でボールを検出できるかを測る。

指標:
  検出率(Recall) = 正解位置の近傍(許容 TOL_PX)に検出枠の中心が来たフレーム / 可視フレーム
  誤検出率       = 正解と対応しない検出 / 全検出
  位置誤差       = 対応した検出の中心と正解座標の距離[px]

学習に使っていない動画で評価するため、build_match_dataset.py と同じ seed で
シャッフルした並びの後ろ側(学習に使わなかった分)から動画を選ぶ。
"""
import argparse
import csv as csvmod
import io
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ZIP_PATH = Path(r"C:\Users\bi23043\Downloads\blurball_dataset.zip")
TMP = Path(r"C:\Users\bi23043\AppData\Local\Temp\bb_eval")
OUT_TXT = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\match_eval.txt")
TOL_PX = 20          # 正解とみなす距離(球が約10pxなので中心から20px以内)
TRAIN_VIDEOS = 150   # build_match_dataset.py で学習に使った本数

_LINES = []


def emit(s=""):
    _LINES.append(s)


def load_annotations(z):
    inner = z.read("blurball_dataset/all_csv_midpoint_annotations.zip")
    iz = zipfile.ZipFile(io.BytesIO(inner))
    ann = {}
    for n in iz.namelist():
        if not n.endswith(".csv"):
            continue
        base = n.split("/")[-1].replace(".csv", "")
        parts = base.split("_")
        rows = []
        for r in csvmod.DictReader(io.StringIO(iz.read(n).decode("utf-8", "replace"))):
            try:
                rows.append((int(r["Frame"]), int(r["Visibility"]),
                             float(r["X"]), float(r["Y"])))
            except (ValueError, KeyError):
                continue
        ann[(parts[0], parts[-1])] = rows
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--videos", type=int, default=12, help="評価する動画本数(未学習分)")
    ap.add_argument("--stride", type=int, default=6, help="評価フレームの間引き")
    args = ap.parse_args()

    from ai_detect import BallDetector, DEFAULT_ONNX
    det = BallDetector(args.onnx or DEFAULT_ONNX, conf=args.conf, box_scale=1.0)

    z = zipfile.ZipFile(ZIP_PATH)
    ann = load_annotations(z)
    vids = [n for n in z.namelist() if n.endswith(".mp4")]
    random.seed(0)
    random.shuffle(vids)
    holdout = vids[TRAIN_VIDEOS:]          # 学習に使っていない動画

    TMP.mkdir(parents=True, exist_ok=True)
    emit(f"モデル: {Path(args.onnx or DEFAULT_ONNX).name}  conf={args.conf}  許容={TOL_PX}px")
    emit(f"{'動画':<14}{'評価fr':>7}{'検出率':>9}{'誤検出率':>10}{'位置誤差px':>11}")
    emit("-" * 55)

    T = {"vis": 0, "hit": 0, "det": 0, "fp": 0, "err": []}
    used = 0
    for vpath in holdout:
        if used >= args.videos:
            break
        parts = vpath.split("/")
        key = (parts[1], parts[-1].replace(".mp4", ""))
        rows = ann.get(key)
        if not rows:
            continue
        local = TMP / "cur.mp4"
        local.write_bytes(z.read(vpath))
        cap = cv2.VideoCapture(str(local))
        if not cap.isOpened():
            continue

        vis_frames = [(f, x, y) for f, v, x, y in rows if v == 1][::args.stride]
        if len(vis_frames) < 5:
            cap.release(); continue

        v = {"vis": 0, "hit": 0, "det": 0, "fp": 0, "err": []}
        for f, gx, gy in vis_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            v["vis"] += 1
            dets = det.detect(fr)
            v["det"] += len(dets)
            best = None
            for cx, cy, w, h, cf in dets:
                d = np.hypot(cx - gx, cy - gy)
                if d <= TOL_PX and (best is None or d < best):
                    best = d
            if best is not None:
                v["hit"] += 1
                v["err"].append(best)
                v["fp"] += len(dets) - 1
            else:
                v["fp"] += len(dets)
        cap.release()
        used += 1
        if v["vis"] == 0:
            continue
        rec = v["hit"] / v["vis"] * 100
        fpr = v["fp"] / v["det"] * 100 if v["det"] else 0
        me = np.mean(v["err"]) if v["err"] else float("nan")
        emit(f"  {key[0]}/{key[1]:<10}{v['vis']:>7}{rec:>8.0f}%{fpr:>9.0f}%{me:>11.1f}")
        for k in ("vis", "hit", "det", "fp"):
            T[k] += v[k]
        T["err"] += v["err"]

    emit("-" * 55)
    if T["vis"]:
        emit(f"  {'合計':<12}{T['vis']:>7}{T['hit']/T['vis']*100:>8.0f}%"
             f"{(T['fp']/T['det']*100 if T['det'] else 0):>9.0f}%"
             f"{(np.mean(T['err']) if T['err'] else float('nan')):>11.1f}")
    emit("\n検出率=正解位置±20px以内に検出できた割合 / 誤検出率=正解に対応しない検出の割合")
    emit("評価は学習に使っていない動画(holdout)のみ。参考: 既存Roboflowモデルは試合で本物57%。")


if __name__ == "__main__":
    main()
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(_LINES), encoding="utf-8")
    print("saved", OUT_TXT)
