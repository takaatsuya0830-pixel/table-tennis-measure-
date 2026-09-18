"""
AI検出(YOLOv8 ONNX) vs 古典CV の検出率比較  ── 2026-08-04

分母の定義: 各動画で古典CVが「ボールが連続して映る」と判定した最長区間 [f0,f1] の
長さ(f1-f0+1)フレーム。実際に μr/rps を算出した窓と同一で、手や影だけが動く区間は含まない。
（AIが古典より広い区間を見つける可能性もあるため、AI側の検出フレーム総数も併記する）

比較対象:
  古典CV = 各計測スクリプトと同じ検出（mu:背景差分+輝度 / side:フレーム差分 / hit:HSV白球）
  AI     = ai_detect.BallDetector（動画によらず同一のモデル・同一閾値＝チューニング不要）
"""
import glob
import io
import sys
from pathlib import Path

import cv2
import numpy as np

from ai_detect import BallDetector
import build_dataset as BD          # 古典CV検出（各計測スクリプトと同一パラメータ）

VDIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos")
ONNX = Path(r"c:\Users\bi23043\Documents\4年前期\models\ball_yolo8.onnx")
OUT_TXT = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\ai_vs_classical.txt")

# build_dataset の import で stdout が閉じられるため、結果は行バッファに貯めてファイル出力する
_LINES = []


def print(*args, **kwargs):          # noqa: A001  (モジュール内のみ差し替え)
    _LINES.append(" ".join(str(a) for a in args))

GROUPS = [
    ("μr斜め(6/9)", "PXL_20260609_135*.mp4", "bright", None),
    ("横アングル(6/9)", "PXL_20260609_1437*.mp4", "framed", None),
    ("打球(6/23)", "PXL_20260623_064552307.mp4", "white", None),
    ("μr(5/22)", "PXL_20260522_0105*_h264.mp4", "bright", (1800, 25, 90)),
    ("俯瞰(6/1)", "PXL_20260601_09204*_h264.mp4", "bright", (1000, 18, 60)),
]


def classical_window(video_path, mode, bparams):
    """古典CVの検出結果から最長連続区間を返す。(f0, f1, 検出フレーム数)"""
    if mode == "bright" and bparams:
        det, n = BD.detect_bright(video_path, *bparams)
    else:
        det, n = BD.DETECTORS[mode](video_path)
    if not det:
        return None
    run = BD.longest_run(list(det.keys()))
    if len(run) < 8:
        return None
    return run[0], run[-1], len(run)


def ai_detect_window(video_path, det_ai, f0, f1):
    """区間[f0,f1]でAIが検出できたフレーム数と、動画全体での検出フレーム数。"""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_win = 0
    total = 0
    fno = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if det_ai.detect_best(fr) is not None:
            total += 1
            if f0 <= fno <= f1:
                in_win += 1
        fno += 1
    cap.release()
    return in_win, total, n


def main():
    det_ai = BallDetector(ONNX, conf=0.25)
    print(f"{'動画':<26}{'窓':>12}{'在view':>7}{'古典':>12}{'AI':>12}{'AI全体':>8}")
    print("-" * 80)
    tot = {"span": 0, "cls": 0, "ai": 0}
    for gname, pat, mode, bparams in GROUPS:
        print(f"[{gname}]")
        for vp in sorted(VDIR.glob(pat)):
            w = classical_window(vp, mode, bparams)
            tag = vp.stem.replace("PXL_2026", "").replace("_h264", "")
            if w is None:
                print(f"  {tag:<24}  古典が窓を作れず → skip")
                continue
            f0, f1, ncls = w
            span = f1 - f0 + 1
            ai_in, ai_tot, nframes = ai_detect_window(vp, det_ai, f0, f1)
            print(f"  {tag:<24}{f'{f0}-{f1}':>12}{span:>7}"
                  f"{ncls}/{span}({ncls/span*100:>3.0f}%){'':2}"
                  f"{ai_in}/{span}({ai_in/span*100:>3.0f}%){ai_tot:>8}")
            tot["span"] += span; tot["cls"] += ncls; tot["ai"] += ai_in
    print("-" * 80)
    if tot["span"]:
        print(f"{'合計':<26}{'':>12}{tot['span']:>7}"
              f"{tot['cls']}/{tot['span']}({tot['cls']/tot['span']*100:>3.0f}%){'':2}"
              f"{tot['ai']}/{tot['span']}({tot['ai']/tot['span']*100:>3.0f}%)")
    print("\n分母=古典CVの最長連続窓。AI全体=動画全体でAIが検出したフレーム数(窓外も含む)。")
    print("AIは全動画で同一モデル・同一閾値(conf=0.25)＝動画ごとのチューニング不要。")


if __name__ == "__main__":
    main()
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(_LINES), encoding="utf-8")
