"""
6/9 再撮影分(大きく鮮明な白球+定規)の μr を計測する  ── 2026-06-09

retab_mu_0522.py と同じ手法:
  背景差分でボール検出 → 中心と真の直径(minAreaRect短辺) → 減速区間抽出
  → フレーム毎の局所px/cm(=真直径/4)でパース補正 → 放物線フィット → μr=a/g
6/9分はボールが大きい(r~70px)・明るいので検出は容易。脚/手の誤検出は
円形度と半径レンジで除外する。
"""
import glob
import io
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import curve_fit

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPS = 240.0
G = 9.81
BALL_DIAM_CM = 4.0
VDIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos")


def parabola(t, s0, v0, a):
    return s0 + v0 * t - 0.5 * a * t * t


def detect_track(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samp = []
    for f in range(0, n, 50):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            samp.append(fr)
    bg = np.median(np.array(samp), axis=0).astype(np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    F, X, Y, D = [], [], [], []
    fno = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        dg = cv2.cvtColor(cv2.absdiff(fr, bg), cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        m = cv2.bitwise_and((dg > 30).astype(np.uint8), (g > 130).astype(np.uint8)) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 3000:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            circ = a / (np.pi * r * r)
            if 40 < r < 130 and circ > 0.6:   # 円形度で脚/手を除外
                if best is None or a > best[0]:
                    (cx, cy), (w, h), _ = cv2.minAreaRect(c)
                    best = (a, x, y, min(w, h))
        if best:
            F.append(fno); X.append(best[1]); Y.append(best[2]); D.append(best[3])
        fno += 1
    cap.release()
    return np.array(F), np.array(X, float), np.array(Y, float), np.array(D, float)


def longest_run(F, max_gap=4):
    if len(F) == 0:
        return []
    runs = [[0]]
    for i in range(1, len(F)):
        if F[i] - F[runs[-1][-1]] <= max_gap + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return max(runs, key=len)


def process(video_path):
    F, X, Y, D = detect_track(video_path)
    if len(F) < 20:
        return {"v": video_path.stem, "note": f"検出{len(F)}点で不足"}
    idx = longest_run(F)
    F, X, Y, D = F[idx], X[idx], Y[idx], D[idx]
    if len(F) < 20:
        return {"v": video_path.stem, "note": f"連続窓{len(F)}点で不足"}

    pts = np.column_stack([X, Y])
    _, sv, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    s_axis = (pts - pts.mean(0)) @ vt[0]
    if s_axis[-1] < s_axis[0]:
        s_axis = -s_axis
    linearity = sv[0] / sv[1] if sv[1] > 0 else 999

    # 減速フェーズ抽出
    t_all = (F - F[0]) / FPS
    sm = np.convolve(s_axis, np.ones(5) / 5, mode="same")
    v_inst = np.gradient(sm, t_all)
    kpk = int(np.argmax(v_inst))
    j = kpk; end = kpk; vmax = v_inst[kpk]
    while j < len(v_inst):
        if v_inst[j] > 0.02 * vmax:
            end = j; j += 1
        else:
            break
    lo, hi = kpk, max(end, kpk + 1)
    if hi - lo >= 15:
        F, X, Y, D, s_axis = F[lo:hi + 1], X[lo:hi + 1], Y[lo:hi + 1], D[lo:hi + 1], s_axis[lo:hi + 1]
    if len(F) < 15:
        return {"v": video_path.stem, "note": "減速区間が短い"}

    dcoef = np.polyfit(s_axis, D, 1)
    ppcm = np.polyval(dcoef, s_axis) / BALL_DIAM_CM
    dpx = np.hypot(np.diff(X), np.diff(Y))
    ppcm_mid = (ppcm[:-1] + ppcm[1:]) / 2
    s_cm = np.concatenate([[0.0], np.cumsum(dpx / ppcm_mid)])
    t = (F - F[0]) / FPS

    mask = np.ones(len(t), bool)
    popt, pcov = curve_fit(parabola, t, s_cm, p0=[0, 150, 100])
    for _ in range(3):
        r = s_cm - parabola(t, *popt)
        sd = r[mask].std()
        if sd == 0:
            break
        nm = np.abs(r) <= 2.5 * sd
        if nm.sum() == mask.sum() or nm.sum() < 10:
            break
        mask = nm
        popt, pcov = curve_fit(parabola, t[mask], s_cm[mask], p0=popt)
    perr = np.sqrt(np.diag(pcov))
    s0, v0, a = popt
    rms_mm = np.std((s_cm - parabola(t, *popt))[mask]) * 10
    return {
        "v": video_path.stem, "n": int(mask.sum()),
        "v0_kmh": v0 / 100 * 3.6, "a_ms": a / 100, "mu": a / 100 / G,
        "mu_err": perr[2] / 100 / G, "lin": linearity, "rms_mm": rms_mm,
        "ppcm": f"{ppcm[0]:.0f}->{ppcm[-1]:.0f}", "diam_px": float(np.median(D)), "note": "OK",
    }


def main():
    vids = sorted(glob.glob(str(VDIR / "PXL_20260609_*.mp4")))
    print(f"{'動画':<22}{'n':>4}{'径px':>6}{'v0[km/h]':>9}{'a[m/s²]':>9}{'μr':>8}{'±σ':>7}{'直線性':>7}{'RMS[mm]':>8}  判定")
    print("-" * 92)
    adopted = []
    for v in vids:
        r = process(Path(v))
        tag = Path(v).stem.replace("PXL_20260609_", "")
        if r.get("note") != "OK":
            print(f"{tag:<22}  {r.get('note','')}")
            continue
        ok = (r["lin"] > 5) and (1.0 <= r["v0_kmh"] <= 12.0) and (r["rms_mm"] < 15) and (0 < r["mu"] < 0.3)
        flag = "採用" if ok else "除外"
        if ok:
            adopted.append(r["mu"])
        print(f"{tag:<22}{r['n']:>4}{r['diam_px']:>6.0f}{r['v0_kmh']:>9.2f}{r['a_ms']:>9.3f}"
              f"{r['mu']:>8.4f}{r['mu_err']:>7.4f}{r['lin']:>7.1f}{r['rms_mm']:>8.2f}  {flag}")
    print("-" * 92)
    if adopted:
        a = np.array(adopted)
        print(f"採用 {len(a)} 本  μr = {a.mean():.4f} ± {a.std(ddof=1):.4f}  (6/9再撮影・パース補正・位置フィット)")


if __name__ == "__main__":
    main()
