"""
6/9撮影分のボール検出率を「正確な分母」で測定する  ── 2026-06-10

分母の定義（重要）:
  各動画について、実際に μr / rps を算出したときと同一の検出ロジックで
  「ボールが連続して映っている最長区間 [f0, f1]」を求め、その区間長
  (f1 - f0 + 1) フレームを分母 = 「在view（ボールが実際に映っているフレーム）」とする。
  → 手・影など、ボール本体が画面に入る前後の動きは区間外なので分母に含まれない。

  ・μr用(斜め+定規, 1348-1351): mu_0609.detect_track（背景差分+輝度+円形度）の最長連続区間
  ・回転用(横アングル, 1437-1438): spin_sideview.track（連続フレーム差分）の最長連続区間

各方式の在view検出率:
  ・背景差分/フレーム差分方式 = その区間内で実際に検出できたフレーム数 / 区間長
  ・Hough方式 = 区間内で、実測ボール半径に合わせた半径レンジ(0.7r〜1.35r)で
                ボール位置近傍に円を検出できたフレーム数 / 区間長
"""
import glob
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np

import mu_0609 as MU
import spin_sideview as SV

VDIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos")
OUT = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detrate_0609.txt")
_LINES = []


def emit(s=""):
    _LINES.append(s)


def hough_rate(video_path, f0, f1, fx, fy, rmed):
    """区間[f0,f1]で、半径レンジを実測球径に合わせたHough検出率。
    fx,fy は区間内各フレームの補間ボール中心。"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    minR = max(15, int(rmed * 0.7))
    maxR = int(rmed * 1.35)
    hit = 0
    for i, f in enumerate(range(f0, f1 + 1)):
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(g, (9, 9), 2)
        circ = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=400,
                                param1=100, param2=30, minRadius=minR, maxRadius=maxR)
        if circ is not None:
            for cx, cy, cr in circ[0]:
                if abs(cx - fx[i]) < rmed and abs(cy - fy[i]) < rmed:
                    hit += 1
                    break
    cap.release()
    return hit, minR, maxR


def process_mu(video_path):
    F, X, Y, D = MU.detect_track(video_path)
    if len(F) < 10:
        return None
    idx = MU.longest_run(F)
    F, X, Y, D = F[idx], X[idx], Y[idx], D[idx]
    f0, f1 = int(F[0]), int(F[-1])
    span = f1 - f0 + 1
    rmed = float(np.median(D)) / 2.0           # D=直径(短辺) → 半径
    allf = np.arange(f0, f1 + 1)
    fx = np.interp(allf, F, X)
    fy = np.interp(allf, F, Y)
    bg_hit = len(F)                            # 区間内で背景差分が検出したフレーム数
    h_hit, minR, maxR = hough_rate(video_path, f0, f1, fx, fy, rmed)
    return dict(span=span, rmed=rmed, bg=bg_hit, hough=h_hit, minR=minR, maxR=maxR, f=(f0, f1))


def process_sv(video_path):
    rec = SV.track(video_path)
    if len(rec) < 10:
        return None
    fnos = [r[0] for r in rec]
    lo, hi = SV.longest_run(fnos)
    rec = rec[lo:hi + 1]
    F = np.array([r[0] for r in rec])
    X = np.array([r[1] for r in rec], float)
    Y = np.array([r[2] for r in rec], float)
    R = np.array([r[3] for r in rec], float)
    f0, f1 = int(F[0]), int(F[-1])
    span = f1 - f0 + 1
    rmed = float(np.median(R))
    allf = np.arange(f0, f1 + 1)
    fx = np.interp(allf, F, X)
    fy = np.interp(allf, F, Y)
    bg_hit = len(F)
    h_hit, minR, maxR = hough_rate(video_path, f0, f1, fx, fy, rmed)
    return dict(span=span, rmed=rmed, bg=bg_hit, hough=h_hit, minR=minR, maxR=maxR, f=(f0, f1))


def main():
    mu_vids = sorted(glob.glob(str(VDIR / "PXL_20260609_1348*.mp4"))
                     + glob.glob(str(VDIR / "PXL_20260609_1349*.mp4"))
                     + glob.glob(str(VDIR / "PXL_20260609_1350*.mp4"))
                     + glob.glob(str(VDIR / "PXL_20260609_1351*.mp4")))
    sv_vids = sorted(glob.glob(str(VDIR / "PXL_20260609_1437*.mp4"))
                     + glob.glob(str(VDIR / "PXL_20260609_1438*.mp4")))

    for title, vids, proc in [("μr用（斜め＋定規）背景差分窓", mu_vids, process_mu),
                              ("回転用（横アングル）フレーム差分窓", sv_vids, process_sv)]:
        emit(f"\n=== {title} ===")
        emit(f"{'動画':<14}{'窓[f0-f1]':>14}{'在view':>7}{'径px':>6}{'背景差分':>11}{'Hough(半径)':>16}")
        emit("-" * 74)
        tot_span = tot_bg = tot_h = 0
        for v in vids:
            r = proc(Path(v))
            tag = Path(v).stem.replace("PXL_20260609_", "")
            if r is None:
                emit(f"{tag:<14}  ボール無し/不足（位置合わせ用クリップ）")
                continue
            f0, f1 = r["f"]
            emit(f"{tag:<14}{f'{f0}-{f1}':>14}{r['span']:>7}{2*r['rmed']:>6.0f}"
                  f"{r['bg']}/{r['span']} ({r['bg']/r['span']*100:>3.0f}%){'':1}"
                  f"{r['hough']}/{r['span']} ({r['hough']/r['span']*100:>3.0f}%) r{r['minR']}-{r['maxR']}")
            tot_span += r["span"]; tot_bg += r["bg"]; tot_h += r["hough"]
        if tot_span:
            emit("-" * 74)
            emit(f"{'合計':<14}{'':>14}{tot_span:>7}{'':>6}"
                  f"{tot_bg}/{tot_span} ({tot_bg/tot_span*100:>3.0f}%){'':1}"
                  f"{tot_h}/{tot_span} ({tot_h/tot_span*100:>3.0f}%)")
    emit("\n分母 = 各動画でボールが連続して映る最長区間の長さ(f1-f0+1)。手・影が動くだけの区間は含まない。")


if __name__ == "__main__":
    main()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(_LINES), encoding="utf-8")
