"""
6/9 横アングル再撮影分の回転速度 rps を算出する  ── 2026-06-09

横アングル(カメラを床高さ・転がり方向に垂直)で撮影。黒点がほぼ常時見え円を描く。
各動画で:
  連続フレーム差分でボール検出(露出ドリフトに強い) → 中心・半径
  rps(rolling) = v/(2πr)  (すべりなし転がり、スケール非依存)
  黒点: ボール内の暗部重心の角度。2点マーカーが約180°対向で最大暗部が飛ぶため、
        連続フレームの角度差の「中央値」を回転率とする(飛びは外れ値として無視)。
  rps(dot) = |median(角度差)| × fps / 360
"""
import glob
import io
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPS = 240.0
VDIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos")


def track(video_path):
    cap = cv2.VideoCapture(str(video_path))
    prev = None
    rec = []   # (fno, x, y, r, [dot angles])
    fno = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is None:
            prev = g; fno += 1; continue
        d = cv2.absdiff(g, prev); prev = g
        m = (d > 28).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) >= 4000:
                (x, y), r = cv2.minEnclosingCircle(c)
                x, y, r = int(x), int(y), int(r)
                if 50 < r < 140:
                    rr = int(r * 0.82)
                    roi = g[max(0, y - rr):y + rr, max(0, x - rr):x + rr].astype(np.float32)
                    angles = []
                    if roi.size:
                        mask = np.zeros(roi.shape, np.uint8)
                        cv2.circle(mask, (roi.shape[1] // 2, roi.shape[0] // 2), rr, 255, -1)
                        dk = ((roi < 100) & (mask > 0)).astype(np.uint8) * 255
                        dk = cv2.morphologyEx(dk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                        dc, _ = cv2.findContours(dk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        big = [cc for cc in dc if cv2.contourArea(cc) >= 25]
                        if big:
                            cc = max(big, key=cv2.contourArea)
                            Mo = cv2.moments(cc)
                            if Mo["m00"]:
                                dx = Mo["m10"] / Mo["m00"] - roi.shape[1] // 2
                                dy = Mo["m01"] / Mo["m00"] - roi.shape[0] // 2
                                if math.hypot(dx, dy) < rr * 0.95:
                                    angles.append(math.degrees(math.atan2(dy, dx)))
                    rec.append((fno, x, y, r, angles))
        fno += 1
    cap.release()
    return rec


def longest_run(fnos, max_gap=4):
    if not fnos:
        return (0, 0)
    runs = [[0]]
    for i in range(1, len(fnos)):
        if fnos[i] - fnos[runs[-1][-1]] <= max_gap + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    r = max(runs, key=len)
    return r[0], r[-1]


def process(video_path):
    rec = track(video_path)
    if len(rec) < 15:
        return {"v": video_path.stem, "note": f"検出{len(rec)}点で不足"}
    fnos = [r[0] for r in rec]
    lo, hi = longest_run(fnos)
    rec = rec[lo:hi + 1]
    if len(rec) < 15:
        return {"v": video_path.stem, "note": "連続窓が短い"}

    F = np.array([r[0] for r in rec])
    X = np.array([r[1] for r in rec], float)
    Y = np.array([r[2] for r in rec], float)
    R = np.array([r[3] for r in rec], float)

    # rolling rps
    dt = np.diff(F) / FPS
    v_px = np.median(np.hypot(np.diff(X), np.diff(Y)) / dt)
    r_px = np.median(R)
    rps_roll = v_px / (2 * np.pi * r_px)
    v_kmh = v_px * (0.02 / r_px) * 3.6

    # dot rps via median consecutive angle step
    diffs = []
    n_seen = 0
    prev_ang = None
    prev_f = None
    for r in rec:
        if not r[4]:
            prev_ang = None
            continue
        n_seen += 1
        ang = r[4][0]
        if prev_ang is not None and r[0] - prev_f == 1:
            dd = ang - prev_ang
            while dd > 180:
                dd -= 360
            while dd < -180:
                dd += 360
            diffs.append(dd)
        prev_ang = ang
        prev_f = r[0]
    rps_dot = None
    if len(diffs) >= 10:
        rps_dot = abs(np.median(diffs)) * FPS / 360.0
    seen_pct = n_seen / len(rec) * 100

    return {"v": video_path.stem, "n": len(rec), "r_px": r_px,
            "v_kmh": v_kmh, "rps_roll": rps_roll, "rps_dot": rps_dot,
            "seen": seen_pct, "note": "OK"}


def main():
    vids = sorted(glob.glob(str(VDIR / "PXL_20260609_1437*.mp4"))
                  + glob.glob(str(VDIR / "PXL_20260609_1438*.mp4")))
    print(f"{'動画':<16}{'n':>4}{'r[px]':>6}{'v[km/h]':>8}{'rps(roll)':>10}{'rps(dot)':>10}{'点可視%':>8}  備考")
    print("-" * 74)
    rolls, dots = [], []
    for v in vids:
        r = process(Path(v))
        tag = Path(v).stem.replace("PXL_20260609_", "")
        if r.get("note") != "OK":
            print(f"{tag:<16}  {r.get('note','')}")
            continue
        dot = f"{r['rps_dot']:.2f}" if r['rps_dot'] else "-"
        print(f"{tag:<16}{r['n']:>4}{r['r_px']:>6.0f}{r['v_kmh']:>8.1f}"
              f"{r['rps_roll']:>10.2f}{dot:>10}{r['seen']:>8.0f}  OK")
        rolls.append(r["rps_roll"])
        if r["rps_dot"]:
            dots.append(r["rps_dot"])
    print("-" * 74)
    if rolls:
        rr = np.array(rolls)
        print(f"rps(rolling) 平均 = {rr.mean():.2f} ± {rr.std(ddof=1):.2f}  ({len(rr)}本)")
    if dots:
        dd = np.array(dots)
        print(f"rps(dot)     平均 = {dd.mean():.2f} ± {dd.std(ddof=1):.2f}  ({len(dd)}本)")
    print("rps(rolling)=v/(2πr) と rps(dot)=黒点角度の中央値ステップ。両者一致ですべりなし転がりを実証。")


if __name__ == "__main__":
    main()
