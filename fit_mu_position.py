"""
転がり摩擦係数 μr を「位置の放物線フィット」で求める  ── 2026-06-09

従来 (calc_velocity.py + final_plot.py) は
  位置 → 中心差分で速度 → 線形減速フィット v(t)=v0-a·t → μr=a/g
だったが、微分が位置ノイズ(Hough中心ジッタ・画素量子化)を増幅していた。

本スクリプトは微分を経ず、位置を直接フィットする:
  1. 検出座標(x,y)を主運動軸(PCA)へ射影して1次元の進行距離 s(t) を作る
     → speed=√(vx²+vy²) の「ノイズ整流バイアス(速度過大評価)」を排除
  2. 転がり区間で s(t)=s0+v0·t-½·a·t² を最小二乗フィット(残差で外れ値反復除去)
  3. μr = a / g。curve_fitの共分散から σ_a → σ_μr を算出
  4. px→cm は「ボール直径(既知40mm)による自己校正」を既定とし、
     ルーラー値(--px-per-cm)も与えれば両者でμrを出して校正感度を提示

使い方:
  python fit_mu_position.py --csv detections_20260522_103205.csv
  python fit_mu_position.py --csv <file> --px-per-cm 31.33   # ルーラー値も併記
  python fit_mu_position.py --csv <file> --f-start 180 --f-end 430
"""

import argparse
import io
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EFFECTIVE_FPS = 240.0          # capture.fps メタデータより確定
G = 9.81                        # m/s²
BALL_DIAM_CM = 4.0              # 卓球ボール直径 40mm = 4cm
CSV_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detections")


def parabola(t, s0, v0, a):
    return s0 + v0 * t - 0.5 * a * t * t


def perspective_corrected(video_path, f_lo, f_hi, fps):
    """動画から各フレームのボール中心と「真の直径(運動と直交＝minAreaRectの短辺)」を測り、
    フレーム毎の局所スケール(px/cm)で進行距離をcm換算してパース歪みを補正する。

    低オブリーク撮影では px/cm が画面位置で大きく変化する(本データで21→16)。
    水平方向の見かけ径はモーションブラーで伸びるため、短辺=真の直径を用いる。
    Returns dict or None.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samp = []
    for f in range(0, n, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            samp.append(fr)
    bg = np.median(np.array(samp), axis=0).astype(np.uint8)

    F, X, Y, D = [], [], [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_lo)
    for fno in range(f_lo, f_hi + 1):
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
        if not cand:
            continue
        c = max(cand, key=cv2.contourArea)
        (cx, cy), (w, h), _ = cv2.minAreaRect(c)
        true_diam = min(w, h)            # 短辺 = ブラー非依存の真の直径
        F.append(fno); X.append(cx); Y.append(cy); D.append(true_diam)
    cap.release()

    if len(F) < 8:
        return None
    F = np.array(F); X = np.array(X); Y = np.array(Y); D = np.array(D)
    t = (F - F[0]) / fps

    # 局所スケール: 真の直径を進行位置の滑らかな関数(1次)でモデル化しノイズ低減
    pts = np.column_stack([X, Y])
    _, _, vt = np.linalg.svd(pts - pts.mean(0), full_matrices=False)
    s_axis = (pts - pts.mean(0)) @ vt[0]
    dcoef = np.polyfit(s_axis, D, 1)
    diam_model = np.polyval(dcoef, s_axis)
    ppcm = diam_model / BALL_DIAM_CM          # フレーム毎 px/cm

    # 局所スケールで各ステップ変位を cm 換算して累積(パース補正後の進行距離)
    dpx = np.hypot(np.diff(X), np.diff(Y))
    ppcm_mid = (ppcm[:-1] + ppcm[1:]) / 2
    s_cm = np.concatenate([[0.0], np.cumsum(dpx / ppcm_mid)])

    popt, pcov = curve_fit(parabola, t, s_cm, p0=[0, 150, 100])
    perr = np.sqrt(np.diag(pcov))
    a_cm = popt[2]
    mu = a_cm / 100.0 / G
    mu_err = perr[2] / 100.0 / G
    return {
        "t": t, "s_cm": s_cm, "popt": popt,
        "v0_kmh": popt[1] / 100 * 3.6, "a_ms": a_cm / 100, "mu": mu, "mu_err": mu_err,
        "ppcm_lo": ppcm[0], "ppcm_hi": ppcm[-1], "n": len(F),
    }


def load_positions(csv_path, source):
    """検出CSVから (frame, x, y, r) を返す。source優先・他へフォールバック。"""
    df = pd.read_csv(csv_path)
    fr = df["frame_number"].to_numpy()
    n = len(df)
    x = np.full(n, np.nan)
    y = np.full(n, np.nan)
    r = np.full(n, np.nan)

    def col(name):
        return df[name].to_numpy(dtype=float) if name in df else np.full(n, np.nan)

    hx, hy, hr = col("hough_x"), col("hough_y"), col("hough_r")
    cx, cy, cr = col("color_x"), col("color_y"), col("color_r")

    if source == "color":
        prim = (cx, cy, cr); sec = (hx, hy, hr)
    else:
        prim = (hx, hy, hr); sec = (cx, cy, cr)

    prim_valid = ~np.isnan(prim[0])
    for i in range(n):
        if not np.isnan(prim[0][i]):
            x[i], y[i], r[i] = prim[0][i], prim[1][i], prim[2][i]
        elif not np.isnan(sec[0][i]):
            x[i], y[i], r[i] = sec[0][i], sec[1][i], sec[2][i]
    return fr, x, y, r, prim_valid


def robust_fit(t, s, n_iter=3, k=2.5):
    """放物線フィット + 残差による外れ値反復除去。(popt, pcov, mask)を返す。"""
    mask = np.ones(len(t), dtype=bool)
    popt, pcov = curve_fit(parabola, t, s, p0=[s[0], (s[-1] - s[0]) / (t[-1] - t[0]), 1.0])
    for _ in range(n_iter):
        resid = s - parabola(t, *popt)
        sd = resid[mask].std()
        if sd == 0:
            break
        newmask = np.abs(resid) <= k * sd
        if newmask.sum() == mask.sum() or newmask.sum() < 5:
            break
        mask = newmask
        popt, pcov = curve_fit(parabola, t[mask], s[mask],
                               p0=popt)
    return popt, pcov, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--source", choices=["color", "hough"], default="color",
                    help="座標ソース(既定:color=minEnclosingCircle中心は比較的安定)")
    ap.add_argument("--fps", type=float, default=EFFECTIVE_FPS)
    ap.add_argument("--px-per-cm", type=float, default=None,
                    help="ルーラー等の外部px/cm(併記用)。未指定ならボール自己校正のみ")
    ap.add_argument("--f-start", type=int, default=None, help="窓の開始フレーム")
    ap.add_argument("--f-end", type=int, default=None, help="窓の終了フレーム")
    ap.add_argument("--video", type=str, default=None,
                    help="元動画パス。指定すると毎フレームのボール径でパース補正したμrを算出(推奨)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = CSV_DIR / csv_path
    if not csv_path.exists():
        print(f"エラー: {csv_path} が見つかりません")
        sys.exit(1)

    fr, x, y, r, prim_valid = load_positions(csv_path, args.source)

    valid = ~np.isnan(x)
    if prim_valid.sum() < 8:
        print("主ソースの検出点が少なすぎます")
        sys.exit(1)

    # 転がり窓: 指定がなければ主ソース検出の最初〜最後(投擲/手のノイズを避ける)
    f_lo = args.f_start if args.f_start is not None else int(fr[prim_valid][0])
    f_hi = args.f_end if args.f_end is not None else int(fr[prim_valid][-1])
    win = valid & (fr >= f_lo) & (fr <= f_hi)
    fw, xw, yw, rw = fr[win], x[win], y[win], r[win]
    tw = fw / args.fps
    tw = tw - tw[0]                      # 窓内相対時刻

    # ── PCA: 主運動軸へ射影して 1次元進行距離 s_px ──
    pts = np.column_stack([xw, yw])
    centroid = pts.mean(0)
    u, sv, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    axis = vt[0]                         # 主成分ベクトル
    s_px = (pts - centroid) @ axis
    if s_px[-1] < s_px[0]:               # 進行方向を正に
        s_px = -s_px
        axis = -axis
    linearity = sv[0] / sv[1] if sv[1] > 0 else float("inf")

    # ── ロバスト放物線フィット ──
    popt, pcov, fitmask = robust_fit(tw, s_px)
    s0, v0_px, a_px = popt
    perr = np.sqrt(np.diag(pcov))
    a_px_err = perr[2]
    v0_px_err = perr[1]

    # ── px/cm 候補 ──
    diam_px = np.nanmedian(rw) * 2.0
    ppcm_ball = diam_px / BALL_DIAM_CM
    candidates = [("ボール自己校正", ppcm_ball)]
    if args.px_per_cm:
        candidates.append(("ルーラー指定", args.px_per_cm))

    print(f"入力       : {csv_path.name}")
    print(f"座標ソース : {args.source}   fps: {args.fps}")
    print(f"窓         : frame {f_lo}-{f_hi}  点数 {win.sum()} (フィット採用 {fitmask.sum()})")
    print(f"軌跡直線性 : 主軸が副軸の {linearity:.1f} 倍 (1次元射影の妥当性)")
    print(f"検出半径   : 中央値 {np.nanmedian(rw):.1f}px → 直径 {diam_px:.1f}px")
    print()
    print(f"  位置フィット v(t)=v0-a·t:")
    print(f"    v0 = {v0_px:.1f} ± {v0_px_err:.1f} px/s")
    print(f"    a  = {a_px:.1f} ± {a_px_err:.1f} px/s²")
    print()
    print(f"  {'px/cm の取り方':<16}{'px/cm':>8}{'v0[km/h]':>10}{'a[m/s²]':>10}{'μr':>9}{'±σ(fit)':>9}")
    print("  " + "-" * 62)
    results = []
    for label, ppcm in candidates:
        ppm = ppcm * 100.0
        v0_kmh = v0_px / ppcm / 100 * 3.6 * 10   # px/s→m/s→km/h: /ppcm/100=m/s ; *3.6
        v0_ms = v0_px / ppm
        a_ms = a_px / ppm
        mu = a_ms / G
        mu_err = a_px_err / ppm / G
        results.append((label, ppcm, v0_ms * 3.6, a_ms, mu, mu_err))
        print(f"  {label:<16}{ppcm:>8.2f}{v0_ms*3.6:>10.2f}{a_ms:>10.3f}{mu:>9.4f}{mu_err:>9.4f}")
    print()
    # 校正感度
    if len(results) >= 2:
        mu_a, mu_b = results[0][4], results[1][4]
        print(f"  校正感度: px/cm を {results[0][1]:.1f}↔{results[1][1]:.1f} で μr={mu_a:.4f}↔{mu_b:.4f} "
              f"({abs(mu_a-mu_b)/max(mu_a,mu_b)*100:.0f}% 差) → 単一px/cmだと校正がμr誤差の主因")

    # ── パース補正(動画があれば: フレーム毎ボール径で局所スケール) ──
    persp = None
    if args.video:
        persp = perspective_corrected(Path(args.video), f_lo, f_hi, args.fps)
        if persp:
            print()
            print(f"  ★パース補正(動画 {persp['n']}点, px/cm {persp['ppcm_lo']:.1f}→{persp['ppcm_hi']:.1f}):")
            print(f"    v0 = {persp['v0_kmh']:.2f} km/h   a = {persp['a_ms']:.3f} m/s²")
            print(f"    μr = {persp['mu']:.4f} ± {persp['mu_err']:.4f}  ← 推奨値(パース歪みを補正)")
        else:
            print("\n  (動画からのパース補正は検出不足で算出できませんでした)")

    # ── プロット(自己校正値で図示) ──
    ppcm = ppcm_ball
    mu_self = results[0][4]
    tf = np.linspace(tw.min(), tw.max(), 300)
    fig, ax = plt.subplots(3, 1, figsize=(9, 11))
    # (1) position fit
    ax[0].scatter(tw[fitmask], s_px[fitmask] / ppcm, s=14, color="steelblue", label="detected (used)")
    if (~fitmask).any():
        ax[0].scatter(tw[~fitmask], s_px[~fitmask] / ppcm, s=14, color="red", marker="x", label="outlier (rejected)")
    ax[0].plot(tf, parabola(tf, *popt) / ppcm, "r--", label="parabola fit")
    _title_mu = (f"mu={persp['mu']:.3f} (perspective-corrected)" if persp
                 else f"mu={mu_self:.3f} (ppcm=ball, no perspective corr.)")
    ax[0].set_ylabel("distance s [cm]"); ax[0].set_title(f"Position fit — {csv_path.stem}  {_title_mu}")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    # (2) velocity: fit-derivative vs numerical difference (reference)
    ax[1].scatter(tw, np.gradient(s_px, tw) / (ppcm * 100) * 3.6, s=10, color="gray", alpha=0.6, label="central diff (ref)")
    ax[1].plot(tf, (v0_px - a_px * tf) / (ppcm * 100) * 3.6, "r--", label="v(t) from fit")
    ax[1].set_ylabel("speed [km/h]"); ax[1].set_title("Velocity: fit-derivative vs numerical diff")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    # (3) residuals
    resid_cm = (s_px - parabola(tw, *popt)) / ppcm
    ax[2].scatter(tw[fitmask], resid_cm[fitmask] * 10, s=14, color="steelblue")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_ylabel("residual [mm]"); ax[2].set_xlabel("time [s]")
    ax[2].set_title(f"Fit residuals (RMS={np.std(resid_cm[fitmask])*10:.2f} mm)")
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    out = csv_path.parent / f"{csv_path.stem.replace('detections','mufit')}.png"
    plt.savefig(str(out), dpi=140); plt.close()
    print(f"\n  図: {out}")


if __name__ == "__main__":
    main()
