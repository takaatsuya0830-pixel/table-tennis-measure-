"""
すべりなし転がりからの回転速度 rps 導出 + 黒点rpsとの比較  ── 2026-06-09

回転用の俯瞰動画では、単一黒点のatan2は前縮み投影で過大評価になりがち
([[spin-detection-root-cause]])。本スクリプトは、より信頼できる経路として
「ボールの並進速度 v とボール半径 r からすべりなし転がり ω=v/r を用いて
 rps = v / (2πr) を算出」し、spin_salvage.py の黒点rpsと突き合わせる。

  - v, r は spin_salvage と同じ背景差分＋輝度でボールを検出して求める
    (r は運動方向・垂直方向ともほぼ等しく、ブラーによる伸びは無視できることを確認済み)
  - ball半径r_px を既知の実半径2cmに対応づけて v[m/s] を自己校正(ルーラー不要)

出力: 各動画の v, r, rps(rolling), rps(dot), 比 を一覧表示。
"""

import io
import sys
from pathlib import Path

import numpy as np

import cv2
import pandas as pd

import spin_salvage as ss  # build_background, detect_ball, EFFECTIVE_FPS を流用(import時にstdoutもUTF-8化)

BALL_RADIUS_CM = 2.0  # 卓球ボール半径 20mm
FPS = ss.EFFECTIVE_FPS
SALVAGE_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\spin_salvage")


def ball_track(video_path):
    """全フレームで移動ボールを検出し (frames, cx, cy, r) を返す。"""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg = ss.build_background(cap, n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fr, cx, cy, rr = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        b = ss.detect_ball(frame, bg)
        if b:
            _, x, y, r = b
            fr.append(i); cx.append(x); cy.append(y); rr.append(r)
        i += 1
    cap.release()
    return np.array(fr), np.array(cx, float), np.array(cy, float), np.array(rr, float)


def longest_run(frames, max_gap=3):
    if len(frames) == 0:
        return []
    runs = [[0]]
    for i in range(1, len(frames)):
        if frames[i] - frames[runs[-1][-1]] <= max_gap + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return max(runs, key=len)


def process(video_name):
    fr, cx, cy, rr = ball_track(ss.VIDEO_DIR / video_name)
    if len(fr) < 5:
        return {"video": Path(video_name).stem, "note": f"検出{len(fr)}点で不足"}
    idx = longest_run(fr)
    fw, xw, yw, rw = fr[idx], cx[idx], cy[idx], rr[idx]
    tw = (fw - fw[0]) / FPS

    # PCA で主運動軸へ射影 → 1次元進行距離(px)
    pts = np.column_stack([xw, yw])
    c = pts.mean(0)
    _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    s = (pts - c) @ vt[0]
    if s[-1] < s[0]:
        s = -s
    # 短窓なので速度は線形フィットの傾き(平均速度)で代表
    A = np.vstack([tw, np.ones_like(tw)]).T
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    v_px = coef[0]                      # px/s
    r_px = np.median(rw)
    # 自己校正: r_px = 2cm
    v_ms = v_px * (BALL_RADIUS_CM / 100.0) / r_px
    rps_roll = v_px / (2 * np.pi * r_px)   # スケール非依存(px/pxで相殺)

    # 黒点rps(あれば)
    rps_dot = None
    runcsv = SALVAGE_DIR / f"{Path(video_name).stem}_run.csv"
    if runcsv.exists():
        d = pd.read_csv(runcsv)
        if len(d) >= 6:
            tt = d.time_sec.values
            cc = d.cum_angle.values
            sl = np.polyfit(tt, cc, 1)[0]
            rps_dot = abs(sl) / 360.0

    return {"video": Path(video_name).stem, "n_run": len(idx),
            "v_ms": v_ms, "v_kmh": v_ms * 3.6, "r_px": r_px,
            "rps_roll": rps_roll, "rps_dot": rps_dot,
            "ratio": (rps_dot / rps_roll) if rps_dot else None, "note": "OK"}


def main():
    print(f"{'動画':<28}{'窓':>4}{'v[km/h]':>9}{'r[px]':>7}{'rps(roll)':>11}{'rps(dot)':>10}{'dot/roll':>9}")
    print("-" * 80)
    for v in ss.VIDEOS:
        r = process(v)
        if r.get("note") != "OK":
            print(f"{r['video']:<28}  {r.get('note','')}")
            continue
        dot = f"{r['rps_dot']:.1f}" if r['rps_dot'] else "-"
        ratio = f"{r['ratio']:.2f}" if r['ratio'] else "-"
        print(f"{r['video']:<28}{r['n_run']:>4}{r['v_kmh']:>9.1f}{r['r_px']:>7.0f}"
              f"{r['rps_roll']:>11.1f}{dot:>10}{ratio:>9}")
    print("-" * 80)
    print("rps(roll)=v/(2πr): すべりなし転がりからの信頼値。")
    print("rps(dot): 黒点atan2(俯瞰投影で過大評価しやすい)。dot/roll>1 は投影バイアス or トップスピン。")


if __name__ == "__main__":
    main()
