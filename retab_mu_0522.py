"""
5/22 セッション全試行を新手法(パース補正・位置フィット)で再集計する  ── 2026-06-09

各動画から直接:
  背景差分でボールを検出 → 中心と「真の直径(minAreaRectの短辺=ブラー非依存)」を毎フレーム取得
  → 転がり窓(最長連続区間のうち単調に進む区間)を選び
  → フレーム毎の局所px/cm(=真直径/4)で進行距離をcm換算(パース補正)
  → s(t)=s0+v0t-½at² を放物線フィット → μr=a/g
旧 calc_velocity(中心差分)+single px/cm を置き換える。

採否は v0 と直線性・残差で機械的に判定し、採用試行の μr 平均±SD を出す。
"""
import io
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

# 5/22 セッションのロール動画
VIDEOS = [
    "PXL_20260522_010519552_h264.mp4",
    "PXL_20260522_010535801_h264.mp4",
    "PXL_20260522_010548402_h264.mp4",
    "PXL_20260522_010558346_h264.mp4",
    "PXL_20260522_020308082_h264.mp4",
    "PXL_20260522_020323568_h264.mp4",
    "PXL_20260522_020337655_h264.mp4",
    "PXL_20260522_020347238_h264.mp4",
    "PXL_20260522_020356332_h264.mp4",
    "PXL_20260522_020406465_h264.mp4",
    "PXL_20260522_020414820_h264.mp4",
]


def parabola(t, s0, v0, a):
    return s0 + v0 * t - 0.5 * a * t * t


def detect_track(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samp = []
    for f in range(0, n, 30):
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
        m = cv2.bitwise_and((dg > 35).astype(np.uint8), (g > 150).astype(np.uint8)) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cand = [c for c in cnts if cv2.contourArea(c) > 1500]
        if cand:
            c = max(cand, key=cv2.contourArea)
            (cx, cy), (w, h), _ = cv2.minAreaRect(c)
            if min(w, h) > 0:
                F.append(fno); X.append(cx); Y.append(cy); D.append(min(w, h))
        fno += 1
    cap.release()
    return np.array(F), np.array(X), np.array(Y), np.array(D)


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


def process(video_name):
    F, X, Y, D = detect_track(VDIR / video_name)
    if len(F) < 20:
        return {"v": Path(video_name).stem, "note": f"検出{len(F)}点で不足"}
    idx = longest_run(F)
    F, X, Y, D = F[idx], X[idx], Y[idx], D[idx]
    if len(F) < 20:
        return {"v": Path(video_name).stem, "note": f"連続窓{len(F)}点で不足"}

    # PCA 主軸へ射影
    pts = np.column_stack([X, Y])
    _, sv, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    s_axis = (pts - pts.mean(0)) @ vt[0]
    if s_axis[-1] < s_axis[0]:
        s_axis = -s_axis
    linearity = sv[0] / sv[1] if sv[1] > 0 else 999

    # ── 減速フェーズだけを切り出す(投擲/加速/検出ジャンプを除外) ──
    # 主軸射影の速度を平滑化し、ピーク以降の単調減速・前進区間を採用
    t_all = (F - F[0]) / FPS
    if len(F) >= 7:
        sm = np.convolve(s_axis, np.ones(5) / 5, mode="same")
        v_inst = np.gradient(sm, t_all)
        kpk = int(np.argmax(v_inst))            # 速度ピーク = 転がり開始付近
        # ピーク以降、速度が正で概ね下降している連続区間
        j = kpk
        end = kpk
        vmax = v_inst[kpk]
        while j < len(v_inst):
            if v_inst[j] > 0.02 * vmax:
                end = j
                j += 1
            else:
                break
        lo, hi = kpk, max(end, kpk + 1)
        if hi - lo >= 20:
            F, X, Y, D, s_axis = F[lo:hi + 1], X[lo:hi + 1], Y[lo:hi + 1], D[lo:hi + 1], s_axis[lo:hi + 1]
    if len(F) < 20:
        return {"v": Path(video_name).stem, "note": "減速区間が短い"}

    # 真直径を進行位置の1次関数でモデル化 → 局所px/cm
    dcoef = np.polyfit(s_axis, D, 1)
    ppcm = np.polyval(dcoef, s_axis) / BALL_DIAM_CM

    # パース補正進行距離[cm]
    dpx = np.hypot(np.diff(X), np.diff(Y))
    ppcm_mid = (ppcm[:-1] + ppcm[1:]) / 2
    s_cm = np.concatenate([[0.0], np.cumsum(dpx / ppcm_mid)])
    t = (F - F[0]) / FPS

    # 放物線フィット(残差で外れ値除去)
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
        "v": Path(video_name).stem, "n": int(mask.sum()),
        "v0_kmh": v0 / 100 * 3.6, "a_ms": a / 100, "mu": a / 100 / G,
        "mu_err": perr[2] / 100 / G, "lin": linearity, "rms_mm": rms_mm,
        "ppcm": f"{ppcm[0]:.0f}->{ppcm[-1]:.0f}", "note": "OK",
    }


def main():
    print(f"{'動画':<30}{'n':>4}{'v0[km/h]':>9}{'a[m/s²]':>9}{'μr':>8}{'±σ':>7}{'直線性':>7}{'RMS[mm]':>8}{'px/cm':>9}")
    print("-" * 100)
    adopted = []
    for v in VIDEOS:
        r = process(v)
        if r.get("note") != "OK":
            print(f"{r['v']:<30}  {r.get('note','')}")
            continue
        # 採否: 直線性が高く、v0が常識的(2〜8km/h)、残差小
        ok = (r["lin"] > 5) and (2.0 <= r["v0_kmh"] <= 8.0) and (r["rms_mm"] < 15) and (0 < r["mu"] < 0.3)
        flag = "採用" if ok else "除外"
        if ok:
            adopted.append(r["mu"])
        print(f"{r['v']:<30}{r['n']:>4}{r['v0_kmh']:>9.2f}{r['a_ms']:>9.3f}"
              f"{r['mu']:>8.4f}{r['mu_err']:>7.4f}{r['lin']:>7.1f}{r['rms_mm']:>8.2f}{r['ppcm']:>9}  {flag}")
    print("-" * 100)
    if adopted:
        a = np.array(adopted)
        print(f"採用 {len(a)} 試行  μr = {a.mean():.4f} ± {a.std(ddof=1):.4f}  (パース補正・位置フィット)")
        print(f"  旧主結果: 0.081 ± 0.012 (中心差分・単一px/cm・fps233)")


if __name__ == "__main__":
    main()
