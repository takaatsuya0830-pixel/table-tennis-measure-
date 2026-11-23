"""
回転(rps)暫定救出スクリプト  ── 2026-06-09

既存の俯瞰動画(PXL_20260601_142xxxxx)から、転がるボールの黒点角度変化を
できる限り救い出して暫定的な rps を算出する。

detect_spin.py の問題点を修正:
  1. 背景差分でボールの「動き」を要求 → 右端の明るい固定物の誤検出を排除
  2. 半径レンジを実測値(約12〜48px)に合わせる
  3. 黒点検出をボール平均輝度に対する適応閾値に変更
  4. 黒点が連続して見える最大の窓だけを使って線形フィット

【限界】俯瞰視点では単一赤道黒点が半回転ごとに隠れ、投影も非線形になるため、
得られる rps は短い弧からの暫定値（大きな不確かさを伴う）。横アングル再撮影が本筋。
"""

import csv
import io
import math
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EFFECTIVE_FPS = 240.0  # 撮影メタデータ com.android.capture.fps=240 より確定(旧233は誤り)
VIDEO_DIR = Path(r"C:\Users\bi23043\Documents\4年前期\卒論\videos")
OUT_DIR = Path(r"C:\Users\bi23043\Documents\4年前期\卒論\frames\spin_salvage")
VIDEOS = [
    "PXL_20260601_142552791_h264.mp4",
    "PXL_20260601_142607743_h264.mp4",
    "PXL_20260601_142707105_h264.mp4",
    "PXL_20260601_142717570_h264.mp4",
    "PXL_20260601_142729147_h264.mp4",
]

# 検出パラメータ
DIFF_THRESH = 35      # 背景差分の動き閾値
BRIGHT_THRESH = 120   # ボールの明るさ閾値
R_MIN, R_MAX = 12, 48
DOT_K = 1.3           # 黒点閾値 = ball_mean - DOT_K*ball_std
DOT_MIN_AREA = 12     # 黒点の最小面積px²
DOT_MIN_CONTRAST = 1.5  # 黒点の最小コントラスト (ball_mean - dot_mean)/ball_std
DOT_MAX_DARKEST = 90  # 黒点の最暗画素がこれ以下(真っ黒)であることを要求(微弱陰影を排除)
MAX_GAP = 4           # 窓内で許容する欠損フレーム数
MIN_RUN = 6           # フィットに必要な最小点数


def imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        open(str(path), "wb").write(buf.tobytes())


def build_background(cap, n_frames, step=40):
    samp = []
    for f in range(0, n_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if ok:
            samp.append(fr)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.array(samp), axis=0).astype(np.uint8)


def detect_ball(frame, bg):
    """背景差分(動き) + 明るさ でボールを検出。"""
    dg = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY)
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = cv2.bitwise_and((dg > DIFF_THRESH).astype(np.uint8),
                        (g > BRIGHT_THRESH).astype(np.uint8)) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 150:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if R_MIN < r < R_MAX and a / (np.pi * r * r) > 0.45:
            if best is None or a > best[0]:
                best = (a, int(x), int(y), int(r))
    return best  # (area, cx, cy, r) or None


def detect_dot(gray, cx, cy, r):
    """ボール内部の黒点を適応閾値で検出。Returns angle_deg or None."""
    rr = int(r * 0.8)
    y0, y1 = cy - rr, cy + rr
    x0, x1 = cx - rr, cx + rr
    if y0 < 0 or x0 < 0 or y1 > gray.shape[0] or x1 > gray.shape[1]:
        return None
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return None
    mask = np.zeros(roi.shape, np.uint8)
    cv2.circle(mask, (rr, rr), rr, 255, -1)
    vals = roi[mask > 0]
    bmean, bstd = vals.mean(), vals.std()
    th = bmean - DOT_K * bstd
    dk = ((roi < th) & (mask > 0)).astype(np.uint8) * 255
    dk = cv2.morphologyEx(dk, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dc, _ = cv2.findContours(dk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dc = [c for c in dc if cv2.contourArea(c) >= DOT_MIN_AREA]
    if not dc:
        return None
    c = max(dc, key=cv2.contourArea)
    area = cv2.contourArea(c)
    # 黒点が大きすぎる(ボール面積の60%超)のは影/影響で除外
    if area > math.pi * rr * rr * 0.6:
        return None
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    # 絶対暗度ゲート: 本物の黒マーカーは真っ黒(最暗画素が十分暗い)。
    # 黒点が裏側へ回った後に残る微弱な陰影(dmin>=110)を確実に排除する。
    cmask = np.zeros(roi.shape, np.uint8)
    cv2.drawContours(cmask, [c], -1, 255, -1)
    dot_vals = roi[cmask > 0]
    dot_min = dot_vals.min()
    dot_mean = dot_vals.mean()
    if dot_min > DOT_MAX_DARKEST:
        return None
    if (bmean - dot_mean) < DOT_MIN_CONTRAST * bstd:
        return None
    dx = M["m10"] / M["m00"] - rr
    dy = M["m01"] / M["m00"] - rr
    if math.hypot(dx, dy) > rr * 0.95:   # 縁すぎ = 影の可能性
        return None
    return math.degrees(math.atan2(dy, dx))


def longest_run(frames, max_gap):
    """フレーム番号リストから、欠損 max_gap 以内で繋がる最長の連続区間を返す。"""
    if not frames:
        return []
    runs = [[frames[0]]]
    for f in frames[1:]:
        if f - runs[-1][-1] <= max_gap + 1:
            runs[-1].append(f)
        else:
            runs.append([f])
    return max(runs, key=len)


def unwrap(angles):
    """連続フレーム間の角度差を ±180° に正規化して累積。"""
    cum = [angles[0]]
    for i in range(1, len(angles)):
        d = angles[i] - angles[i - 1]
        if d > 180:
            d -= 360
        elif d < -180:
            d += 360
        cum.append(cum[-1] + d)
    return cum


def process(video_name):
    path = VIDEO_DIR / video_name
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = build_background(cap, n)

    det = {}   # frame -> (time, cx, cy, r, angle)
    fno = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ball = detect_ball(fr, bg)
        if ball:
            _, cx, cy, r = ball
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            ang = detect_dot(g, cx, cy, r)
            if ang is not None:
                det[fno] = (fno / EFFECTIVE_FPS, cx, cy, r, ang)
        fno += 1
    cap.release()

    stem = Path(video_name).stem
    if len(det) < MIN_RUN:
        return {"video": stem, "n_dot": len(det), "rps": None,
                "note": "黒点検出が少なすぎる"}

    run = longest_run(sorted(det.keys()), MAX_GAP)
    if len(run) < MIN_RUN:
        return {"video": stem, "n_dot": len(det), "rps": None,
                "note": f"連続窓が短い(最長{len(run)}点)"}

    times = np.array([det[f][0] for f in run])
    raw = [det[f][4] for f in run]
    cum = np.array(unwrap(raw))

    # 線形フィット + 不確かさ
    A = np.vstack([times, np.ones_like(times)]).T
    coef, res, *_ = np.linalg.lstsq(A, cum, rcond=None)
    slope = coef[0]                       # deg/s
    pred = A @ coef
    ss_res = np.sum((cum - pred) ** 2)
    ss_tot = np.sum((cum - cum.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(times)
    # slope の標準誤差
    if n > 2:
        sigma2 = ss_res / (n - 2)
        sxx = np.sum((times - times.mean()) ** 2)
        slope_se = math.sqrt(sigma2 / sxx) if sxx > 0 else float("nan")
    else:
        slope_se = float("nan")

    rps = slope / 360.0
    rps_se = slope_se / 360.0
    arc = cum[-1] - cum[0]
    dur = times[-1] - times[0]

    # プロット
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(times, cum, s=25, color="steelblue", zorder=3,
               label=f"dot angle (n={n})")
    tf = np.linspace(times[0], times[-1], 100)
    ax.plot(tf, np.polyval(coef, tf), "r--",
            label=f"fit: {rps:+.2f} ± {rps_se:.2f} rps  (R²={r2:.2f})")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cumulative dot angle [deg]")
    ax.set_title(f"Spin salvage — {stem}\n"
                 f"arc={arc:+.0f}° over {dur*1000:.0f} ms")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f"{stem}_salvage.png"), dpi=140)
    plt.close()

    # CSV
    with open(OUT_DIR / f"{stem}_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_sec", "cx", "cy", "r", "raw_angle", "cum_angle"])
        for fr_i, ca in zip(run, cum):
            t, cx, cy, r, a = det[fr_i]
            w.writerow([fr_i, f"{t:.5f}", cx, cy, r, f"{a:.1f}", f"{ca:.1f}"])

    return {"video": stem, "n_dot": len(det), "n_run": n,
            "window": (run[0], run[-1]), "dur_ms": dur * 1000,
            "arc_deg": arc, "rps": rps, "rps_se": rps_se, "r2": r2,
            "note": "OK"}


def main():
    print(f"{'動画':<28}{'黒点':>5}{'窓点':>5}{'窓ms':>7}{'弧°':>7}{'rps':>9}{'±se':>7}{'R²':>6}  備考")
    print("-" * 90)
    results = []
    for v in VIDEOS:
        r = process(v)
        results.append(r)
        if r is None:
            print(f"{v:<28} 読み込み失敗")
            continue
        if r["rps"] is None:
            print(f"{r['video']:<28}{r['n_dot']:>5}{'-':>5}{'-':>7}{'-':>7}{'-':>9}{'-':>7}{'-':>6}  {r['note']}")
        else:
            print(f"{r['video']:<28}{r['n_dot']:>5}{r['n_run']:>5}"
                  f"{r['dur_ms']:>7.0f}{r['arc_deg']:>7.0f}"
                  f"{r['rps']:>+9.2f}{r['rps_se']:>7.2f}{r['r2']:>6.2f}  {r['note']}")
    print("-" * 90)
    print(f"出力: {OUT_DIR}")
    print("\n【注意】俯瞰視点・短い弧のため、これらは大きな不確かさを伴う暫定値。")


if __name__ == "__main__":
    main()
