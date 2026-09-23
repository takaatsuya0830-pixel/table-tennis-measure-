"""
ローリングシャッター読み出し時間 τ の較正  ── 2026-09-18

スマホのセンサーは画面上の行から順に読み出すため、行 y の実撮影時刻は
  t = n/fps + τ·(y/H)
だけ遅れる。縦方向に動く球では有効なフレーム間隔が伸縮し、加速度に偏りが出る。

■ 落下動画による推定(本スクリプト)
  自由落下(重力+空気抵抗)の y(t) を、τ を変えながらフィットし、残差と推定 g を出す。
  限界: 純粋な時刻ずれは放物線に3次の歪みを作るが、その信号は小さく、
  照明のちらつき(奇偶フレームの露光差)による残差に埋もれて τ を一意に決められない。
  さらに「幅から求めたスケール」の誤差と縮退する(τ と px/cm の両方が g を動かす)。
  → この方法だけでは τ は確定しない。結果は「τ の範囲と、スケール誤差との切り分け」に使う。

■ 推奨する独立較正(ちらつき照明法)
  60Hz地域の蛍光灯/LEDは120Hzで明滅する。それを240fpsで撮ると、読み出し中に明暗が
  変わるので画面に横縞が出る。縞1周期の行数 R から  τ = H·(1/120) / R  で求まる。
  これはスケールと無関係で、1分で較正できる。求めた τ を hit_speed.py の
  --readout-ms に渡す。

実測(2026-09-18, 落下動画 PXL_20260804_233126876):
  抗力込みで τ=0 のとき g=11.49 (+17%)。τ=3-4ms で +5〜8%、g=9.81 には τ=5.75ms が
  必要だが1フレーム(4.17ms)を超えるので不可能 → 偏りは「τ(半分)+幅スケール(半分)」の合算。
  また奇偶フレームで露光が交互に変わる現象は、120Hz照明×240fps の同期で説明できる。
  撮影時は非ちらつき光源(太陽光・DC点灯LED)を使うか、この効果を織り込むこと。
"""
import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FPS = 240.0
BALL_DIAM_CM = 4.0
K_M = 1.2 * 0.45 * np.pi * 0.02 ** 2 / (2 * 0.0027)   # 抗力係数 ρCdA/2m [1/m]


def track_fall(video, f0, f1):
    cap = cv2.VideoCapture(str(video))
    H = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samp = []
    for f in list(range(max(0, f0 - 130), max(0, f0 - 10), 20)) + list(range(min(n - 1, f1 + 50), min(n - 1, f1 + 170), 20)):
        cap.set(1, f); ok, fr = cap.read()
        if ok: samp.append(fr)
    bg = np.median(np.array(samp), 0).astype(np.uint8)
    rows = []
    for fno in range(f0, f1 + 1):
        cap.set(1, fno); ok, fr = cap.read()
        if not ok: break
        dg = cv2.cvtColor(cv2.absdiff(fr, bg), cv2.COLOR_BGR2GRAY)
        m = (dg > 20).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 1500: continue
        x0, y0, w, h = cv2.boundingRect(c); M = cv2.moments(c)
        rows.append((fno, M["m01"] / M["m00"], w))
    cap.release()
    F = np.array([r[0] for r in rows], float); Y = np.array([r[1] for r in rows]); W = np.array([r[2] for r in rows])
    return F, Y, W, H


def fit_g(F, Y, ppcm, H, tau_ms):
    t = (F - F[0]) / FPS + (tau_ms / 1000.0) * Y / H
    t = t - t[0]
    k_cm = K_M / 100.0
    def sim(p):
        g, y0, v0 = p
        sol = solve_ivp(lambda _t, s: [s[1], g - k_cm * s[1] * abs(s[1]) / ppcm],
                        (t[0], t[-1]), [y0, v0], t_eval=t, rtol=1e-8, atol=1e-8)
        return sol.y[0]
    r = least_squares(lambda p: sim(p) - Y, x0=[50000, Y[0], (Y[1] - Y[0]) / (t[1] - t[0])])
    return r.x[0], float(np.sqrt(np.mean((sim(r.x) - Y) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--f-start", type=int, required=True)
    ap.add_argument("--f-end", type=int, required=True)
    ap.add_argument("--tau-max", type=float, default=4.2, help="探索上限[ms](1/fps以下が物理的)")
    a = ap.parse_args()
    F, Y, W, H = track_fall(Path(a.video), a.f_start, a.f_end)
    ppcm = np.median(W) / BALL_DIAM_CM
    print(f"追跡 {len(F)} フレーム  幅中央値 {np.median(W):.0f}px → px/cm(幅) = {ppcm:.2f}")
    print(f"{'tau[ms]':>7} {'g_px':>8} {'残差px':>7} {'g[m/s2]':>8} {'誤差%':>6}")
    for tau in np.arange(0, a.tau_max + 1e-9, 0.5):
        g, rms = fit_g(F, Y, ppcm, H, tau)
        gm = g / ppcm / 100
        print(f"{tau:7.1f} {g:8.0f} {rms:7.2f} {gm:8.2f} {(gm-9.81)/9.81*100:+6.1f}")
    print("\n残差が平坦なら τ は形からは決まらない。ちらつき照明法で独立に較正すること(docstring参照)。")


if __name__ == "__main__":
    main()
