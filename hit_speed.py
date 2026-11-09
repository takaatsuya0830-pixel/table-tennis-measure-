"""
打球速度 計測スクリプト  ── 2026-06-23

打った/投げた卓球ボールが飛ぶ動画から、飛球の速度(m/s, km/h)を算出する。

手法:
  1. 各フレームで白球を検出（HSVで低彩度・高輝度 → 肌色の手は彩度が高く自動除外）
  2. 検出が連続する最長区間（=1回の飛行）を切り出し、半径の外れ値(ブラーで潰れた点)を除去
  3. ボール直径(既知40mm)による自己校正で px→cm（飛球面の局所スケール、定規不要）
  4. 座標を主運動軸へ射影(PCA)し、進行距離 s(t) を直線(等速)フィット → 速度
     ※飛行は0.1秒程度で空気抵抗は小さく、ほぼ等速。区間ごとの速度範囲も併記

使い方:
  python hit_speed.py --video PXL_20260623_064552307.mp4
  python hit_speed.py --video <path> --v-low 150 --s-high 60   # 白球HSV閾値の調整
  python hit_speed.py --video <path> --f-start 555 --f-end 590  # 区間を手動指定

【限界】1台のカメラでは奥行き方向の速度は測れない。飛球線に対しカメラを
垂直（球が画面を横切る）に置くこと。回転(スピン)は別計測（240fpsは120rpsまで）。
"""

import argparse
import csv
import io
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

EFFECTIVE_FPS = 240.0          # capture.fps メタデータより
BALL_DIAM_CM = 4.0             # 卓球ボール直径 40mm
# ── 空気抵抗モデルの物理定数 ──
# 卓球ボールは軽い(2.7g)ので飛行中の減速が大きい。抗力 F=½ρv²CdA より
# 減速度 a = k·v²,  k = ρCdA/(2m)。20km/hでも0.1秒で約7%、60km/hでは約21%減速する。
# 「等速フィット」は飛行中の平均速度を返すため、打球の初速を高速ほど過小評価する。
BALL_MASS_KG = 0.0027
AIR_DENSITY = 1.2              # kg/m^3
DRAG_CD = 0.56                 # 2026-09-23 自由落下2本から実測(0.51, 0.62の平均。文献0.4-0.6)
# 卓球ボールは軽いので空気の浮力と付加質量(押しのける空気の半分)が効く:
#   g_eff = g(1-ρV/m)/(1+ρV/2m) = 9.59 m/s²,  抗力も 1/(1+ρV/2m) 倍になる
# (9/23の落下検証で、この g_eff に対し -1.1%, -3.2% で一致)
_BALL_V = 4.0 / 3.0 * np.pi * (BALL_DIAM_CM / 200.0) ** 3
_ADDED = 1.0 + 0.5 * AIR_DENSITY * _BALL_V / BALL_MASS_KG
G_CM = 981.0 * (1.0 - AIR_DENSITY * _BALL_V / BALL_MASS_KG) / _ADDED   # 実効重力 cm/s^2
FOCAL_PX = 1250.0              # 1080p動画の焦点距離[px](Pixel主カメラ、水平画角~75°)。3D再構成の奥行きのみに効く
VIDEO_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos")
OUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\hit_speed")


def imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        open(str(path), "wb").write(buf.tobytes())


def detect_white_ball(frame, v_low, s_high, r_min, r_max, min_area):
    """HSVで白球(低彩度・高輝度)を検出。最大の円形blobの(x,y,r)を返す。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, v_low), (180, s_high, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r_min < r < r_max and a / (np.pi * r * r) > 0.55:
            if best is None or a > best[0]:
                best = (a, float(x), float(y), float(r))
    return best[1:] if best else None


def detect_ball_streak(frame, v_low, s_high, r_min, r_max, min_area, max_aspect=8.0):
    """高速球むけ検出。運動ブラーで伸びた球(ストリーク)を受け入れる。

    高速になるほど球は円でなく線になるため、円形度で選ぶ従来法は破綻する
    (露光4.2ms・240fpsでは 10km/h 程度からブラーが直径の3割を超える)。
    ブラーは運動方向にしか伸びないので:
      真の直径 = minAreaRect の短辺  (落下動画で速度が増えても180-190pxで安定と実証)
      位置     = 輪郭の重心          (露光中の平均位置。速度推定には一貫していれば良い)
    Returns (x, y, r) で r は短辺/2。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, v_low), (180, s_high, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        (cx, cy), (s1, s2), _ = cv2.minAreaRect(c)
        minor, major = min(s1, s2), max(s1, s2)
        if minor <= 0:
            continue
        if not (r_min < minor / 2 < r_max):
            continue
        if major / minor > max_aspect:      # 伸びすぎ=球でない(手・背景の帯など)
            continue
        # ストリークは細長いが「太さ方向には詰まっている」ので充填率で偽物を除く
        if a / (minor * major) < 0.55:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        x, y = M["m10"] / M["m00"], M["m01"] / M["m00"]
        if best is None or a > best[0]:
            best = (a, float(x), float(y), float(minor / 2))
    return best[1:] if best else None


def longest_run(frames, max_gap=6):
    """フレーム番号リストから欠損 max_gap 以内で繋がる最長区間のインデックスを返す。"""
    if not frames:
        return []
    runs = [[0]]
    for i in range(1, len(frames)):
        if frames[i] - frames[runs[-1][-1]] <= max_gap + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return max(runs, key=len)


def robust_linear(t, s, n_iter=3, k=2.5):
    """進行距離 s(t) を等速直線フィット(残差で外れ値除去)。slope[px/s], mask を返す。"""
    mask = np.ones(len(t), bool)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A[mask], s[mask], rcond=None)
    for _ in range(n_iter):
        resid = s - (A @ coef)
        sd = resid[mask].std()
        if sd == 0:
            break
        nm = np.abs(resid) <= k * sd
        if nm.sum() == mask.sum() or nm.sum() < 4:
            break
        mask = nm
        coef, *_ = np.linalg.lstsq(A[mask], s[mask], rcond=None)
    return coef[0], coef, mask


def drag_fit(t, s_cm, g_along_cm, k_cm):
    """空気抵抗を考慮した飛行モデルで進行距離 s(t) をフィットし、初速 v0 を返す。

    主軸方向の運動方程式:  dv/dt = g_along − k·v·|v|
      g_along: 重力の主軸成分[cm/s^2](軌道が下向きなら正、上向きなら負)
      k      : ρCdA/(2m) を cm 単位にしたもの[1/cm]
    自由パラメータは s0 と v0 のみ(物理定数は既知)。
    Returns dict(v0, s0, v_end, resid_rms) — v は cm/s。
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares

    def simulate(v0, s0):
        def rhs(_t, y):
            s, v = y
            return [v, g_along_cm - k_cm * v * abs(v)]
        sol = solve_ivp(rhs, (t[0], t[-1]), [s0, v0], t_eval=t, rtol=1e-8, atol=1e-8)
        return sol.y[0], sol.y[1]

    v_guess = (s_cm[-1] - s_cm[0]) / max(t[-1] - t[0], 1e-6)

    def resid(p):
        s_pred, _ = simulate(p[0], p[1])
        return s_pred - s_cm

    res = least_squares(resid, x0=[v_guess, s_cm[0]])
    v0, s0 = res.x
    s_pred, v_pred = simulate(v0, s0)
    rms = float(np.sqrt(np.mean((s_pred - s_cm) ** 2)))
    return {"v0": float(v0), "s0": float(s0), "v_end": float(v_pred[-1]),
            "resid_rms": rms, "s_pred": s_pred, "v_pred": v_pred}


def bgsub_candidates(fr, bg):
    """背景差分でボール候補を探す。Returns [(cx, cy, 直径px, 画面端で欠けているか), ...]

    直径 = 差分マスクの minAreaRect 短辺。打球の追跡(track_bgsub)と静止球の較正
    (calib_static.py)で同じ定義を使うための共通関数。定義を変えると較正値 b が無効になる。
    """
    H, W = fr.shape[:2]
    dg = cv2.cvtColor(cv2.absdiff(fr, bg), cv2.COLOR_BGR2GRAY)
    m = (dg > 30).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 150:
            continue
        (_, _), (s1, s2), _ = cv2.minAreaRect(c)
        mi, ma = min(s1, s2), max(s1, s2)
        if not (12 < mi < 300) or ma / max(mi, 1) > 6 or a / (mi * ma) < 0.5:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        edge = bx <= 1 or by <= 1 or bx + bw >= W - 1 or by + bh >= H - 1   # 画面端で欠けた球
        M = cv2.moments(c)
        out.append((M["m10"] / M["m00"], M["m01"] / M["m00"], mi, edge))
    return out


def track_bgsub(path, f0, f1, gate=90.0, max_lost=3, gate_first=300.0):
    """背景差分 + 予測近傍追跡で飛球を追う(暖色照明・白飛びでも使える)。

    HSVの「白」判定は暖色光で球が黄色く写ると球の一部しか拾えず、
    白飛びした打球では直径170pxの球を104pxと誤る(9/23動画)。背景差分なら色に依らない。
    各フレームの候補blobを予測位置(前フレーム位置+速度)に最も近いものへ繋ぎ、
    複数の軌跡(人・ラケットも含む)のうち画面上の移動量が最大のものを飛球とする。
    直径 = minAreaRect の短辺(ブラーは運動方向にしか伸びないため)。
    つなぐ範囲(gate): 速度がまだ分からない2点目は gate_first まで広げ、その後は
    予測位置から max(gate, 0.6×1フレームの移動量) 以内。50km/h級の打球は1フレームで
    100px以上進むため、固定の90pxでは追跡が始まらず前半を取りこぼしていた(10/20動画)。
    Returns F, X, Y, D (D=直径px), 画像幅, 画像高さ
    """
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    samp = []
    for f in np.linspace(0, n - 1, 40).astype(int):      # 球は各地点に一瞬しか居ない→中央値が背景
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if ok:
            samp.append(fr)
    bg = np.median(np.array(samp), axis=0).astype(np.uint8)

    active, done = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for f in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        cands = [[cx, cy, mi, edge, False] for cx, cy, mi, edge in bgsub_candidates(fr, bg)]
        for tr in active:
            lf, px, py = tr["rows"][-1][:3]
            dt = f - lf
            qx, qy = px + tr["vx"] * dt, py + tr["vy"] * dt
            if len(tr["rows"]) == 1:
                g_now = gate_first * dt
            else:
                g_now = max(gate, 0.6 * np.hypot(tr["vx"], tr["vy"]) * dt)
            best = None
            for c in cands:
                if c[4]:
                    continue
                e = np.hypot(c[0] - qx, c[1] - qy)
                if e < g_now and (best is None or e < best[0]):
                    best = (e, c)
            if best is None:
                tr["lost"] += 1
                continue
            c = best[1]; c[4] = True
            tr["vx"], tr["vy"] = (c[0] - px) / dt, (c[1] - py) / dt
            tr["lost"] = 0
            tr["rows"].append((f, c[0], c[1], c[2], c[3]))
        for c in cands:
            if not c[4]:
                active.append({"rows": [(f, c[0], c[1], c[2], c[3])], "vx": 0.0, "vy": 0.0, "lost": 0})
        done += [t for t in active if t["lost"] > max_lost]
        active = [t for t in active if t["lost"] <= max_lost]
    cap.release()
    done += active
    done = [t for t in done if len(t["rows"]) >= 8]
    if not done:
        return None
    # 飛球の選び方: 画面上を十分長く動いた軌跡(最長の半分以上)のうち、最も早いもの。
    # 壁で跳ね返って戻ってくる球は打球より長く写ることがあり、「最長」で選ぶと
    # 跳ね返りを打球と取り違える(10/20 041940482 で 178km/h という誤った値になった)。
    disp = [np.hypot(t["rows"][-1][1] - t["rows"][0][1], t["rows"][-1][2] - t["rows"][0][2]) for t in done]
    long_ = [t for t, dd in zip(done, disp) if dd >= 0.5 * max(disp)]
    best = min(long_, key=lambda t: t["rows"][0][0])
    R = np.array([r for r in best["rows"] if not r[4]], float)    # 端で欠けた球は除外
    if len(R) == 0:
        return None
    return R[:, 0], R[:, 1], R[:, 2], R[:, 3], W, H


def diameter_outlier_mask(D, win=7, tol=0.25):
    """前後 win 点の中央値から tol(25%)以上、または全体の中央値から50%以上ずれた直径を外れ値とする。

    球が下げ振りのおもり等と重なると直径が大きく、物の陰や画面の端で一部しか
    写らないと小さく出る(10/20 040155468: 約60pxの中に101pxと13px)。
    こうした点が数点あるだけで初期値が崩れ、フィットが発散した。
    """
    D = np.asarray(D, float)
    h = win // 2
    med = np.array([np.median(D[max(0, i - h):i + h + 1]) for i in range(len(D))])
    gm = np.median(D)          # 端で数点まとまって小さい場合は近所の中央値も小さいので、全体の中央値でも判定
    return (np.abs(D - med) <= tol * med) & (np.abs(D - gm) <= 0.5 * gm)


def fit_3d(t, x, y, d, focal_px, cx, cy):
    """球の見かけ直径から3次元軌道を再構成し、抗力+重力の運動方程式でフィットする。

    ピンホールカメラ:  Z = f·D/d,  X = (x-cx)·D/d,  Y = (y-cy)·D/d   (D=4cm)
    横・縦の位置は f に依らず、奥行き Z だけが f に比例する。
    → カメラに近づく/遠ざかる成分も含めた「本当の速さ」が1台のカメラで得られる。
    モデル: dv/dt = g·ĝ − k|v|v。ĝ=重力の向き(カメラのピッチ・ロールで表す)。
    g の大きさも自由にして、既知の g_eff と比べることで結果全体の検証に使う
    (直径の定義やスケールが狂っていれば g が合わない)。
    Returns dict: v0vec, v0, v_end, g, g_err, pitch, roll, rms[m]
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
    # 直径のノイズを抑える: 1/d(∝奥行き)を時間の2次で平滑化(外れ値は3σで除去)
    keep = np.ones(len(t), bool)
    for _ in range(3):
        c = np.polyfit(t[keep], 1 / d[keep], 2)
        r = 1 / d - np.polyval(c, t)
        nk = np.abs(r) < 3 * np.std(r[keep]) + 1e-12
        if (nk == keep).all():
            break
        keep = nk
    t, x, y = t[keep], x[keep], y[keep]
    inv_d = np.polyval(c, t)
    Dm = BALL_DIAM_CM / 100.0
    P = np.column_stack([(x - cx) * Dm * inv_d, (y - cy) * Dm * inv_d, focal_px * Dm * inv_d])
    k = AIR_DENSITY * DRAG_CD * np.pi * (Dm / 2) ** 2 / (2 * BALL_MASS_KG) / _ADDED

    def gdir(th, ph):
        return np.array([np.sin(ph) * np.cos(th), np.cos(ph) * np.cos(th), np.sin(th)])

    def sim(p):
        def rhs(_t, q):
            v = q[3:]
            return np.concatenate([v, p[6] * gdir(p[7], p[8]) - k * np.linalg.norm(v) * v])
        z = solve_ivp(rhs, (t[0], t[-1]), p[:6], t_eval=t, rtol=1e-9, atol=1e-9)
        return z.y[:3].T, z.y[3:].T

    j = min(3, len(t) - 1)
    x0 = np.concatenate([P[0], (P[j] - P[0]) / (t[j] - t[0]), [G_CM / 100.0, 0.0, 0.0]])
    res = least_squares(lambda p: (sim(p)[0] - P).ravel(), x0=x0)
    J = res.jac
    cov = np.linalg.pinv(J.T @ J) * (res.fun @ res.fun / max(1, len(res.fun) - len(res.x)))
    Pf, V = sim(res.x)
    # p0vec: (外れ値除去後の)最初の時刻における3D位置。fit_image の初期値に使う
    return {"p0vec": Pf[0], "v0vec": V[0], "v0": float(np.linalg.norm(V[0])), "v_end": float(np.linalg.norm(V[-1])),
            "g": float(res.x[6]), "g_err": float(np.sqrt(cov[6, 6])),
            "pitch": float(np.degrees(res.x[7])), "roll": float(np.degrees(res.x[8])),
            "rms": float(np.sqrt(np.mean(res.fun ** 2))), "n": int(len(t)), "d_med": float(np.median(1 / inv_d))}


def fit_image(t, x, y, d, focal_px, cx, cy, init=None, d_bias=0.0, fix_g=False, grav=None):
    """画像空間で直接フィットする(fit_3d の改良版)。

    fit_3d は観測 d から Z=f·D/d を作ってから3D位置をフィットするため、
    d の誤差(数%)が X,Y,Z すべてに相関して入り、x,y の高精度(~1px)を活かせない。
    ここでは逆に、3D運動モデルを画像へ投影して観測と比べる:
        x_pred = f·X/Z + cx,  y_pred = f·Y/Z + cy,  d_pred = f·D/Z + d_bias
        残差 = ((x-x_pred)/σxy, (y-y_pred)/σxy, (d-d_pred)/σd)
    σxy, σd は残差から推定して1回再フィットする(各チャンネルを正しい重みで使う)。
    外れ値は soft_l1 ロスで抑える(しきい値で捨てない)。
    d_bias: 直径の測り方の系統的な偏り[px](静止球の距離較正で求めたものを与える)。
    fix_g : g の大きさを g_eff に固定する。速い球は飛行が短く(60km/hで~0.1s)落下が
            ほとんど写らないため g が決まらず(合成データで±46%)、初速も道連れでぶれる。
            その場合は g を既知として固定し、検証には使わない。
    grav  : 下げ振り(plumb.gravity_constraint)から得た重力の向きの条件。None なら向きも推定。
    Returns fit_3d と同じキー + v0_err(初速の1σ, m/s), sig_xy, sig_d, grav_mode
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
    Dm = BALL_DIAM_CM / 100.0
    k = AIR_DENSITY * DRAG_CD * np.pi * (Dm / 2) ** 2 / (2 * BALL_MASS_KG) / _ADDED
    if init is None:
        init = fit_3d(t, x, y, d - d_bias, focal_px, cx, cy)

    def gdir(th, ph):
        return np.array([np.sin(ph) * np.cos(th), np.cos(ph) * np.cos(th), np.sin(th)])

    # 重力の向きの扱い(grav):
    #   None            → カメラのピッチ・ロール(2角)を未知数として推定
    #   ("plane", n)    → 下げ振り1本: 向きは法線 n の平面内。平面内の角度1つだけ推定
    #   ("fixed", dvec) → 下げ振り2本: 向きは既知。角度は推定しない
    # 速い球は飛行が短く向きがほとんど決まらないので、糸で向きを与えると安定する。
    mode = "free" if grav is None else grav[0]
    if mode == "plane":
        nrm = np.asarray(grav[1], float); nrm = nrm / np.linalg.norm(nrm)
        u0 = np.array([0.0, 1.0, 0.0]) - nrm[1] * nrm; u0 /= np.linalg.norm(u0)   # 平面内で画像の真下に最も近い向き
        w0 = np.cross(nrm, u0)
    if mode == "fixed":
        dfix = np.asarray(grav[1], float); dfix = dfix / np.linalg.norm(dfix)
    na = {"free": 2, "plane": 1, "fixed": 0}[mode]
    G0 = G_CM / 100.0

    def decode(q):
        """未知数ベクトル → (初期状態6, gの大きさ, 重力の向き)"""
        i = 6
        g = G0 if fix_g else q[i]
        i += 0 if fix_g else 1
        if mode == "free":
            dv = gdir(q[i], q[i + 1])
        elif mode == "plane":
            dv = np.cos(q[i]) * u0 + np.sin(q[i]) * w0
        else:
            dv = dfix
        return q[:6], g, dv

    def sim(q):
        s0, g, dv = decode(q)
        def rhs(_t, z):
            v = z[3:]
            return np.concatenate([v, g * dv - k * np.linalg.norm(v) * v])
        z = solve_ivp(rhs, (t[0], t[-1]), s0, t_eval=t, rtol=1e-9, atol=1e-9)
        return z.y[:3].T, z.y[3:].T

    def project(q):
        P, _ = sim(q)
        Z = P[:, 2]
        return focal_px * P[:, 0] / Z + cx, focal_px * P[:, 1] / Z + cy, focal_px * Dm / Z + d_bias

    # 初期値: 3D法の解。角度は ±45° に制限する(速い球で角度が暴走して初速が発散した: 10/20 040155468)
    ANG = np.radians(45.0)
    clipr = lambda a: float(np.clip(a, -0.95 * ANG, 0.95 * ANG))
    d_init = gdir(np.radians(init["pitch"]), np.radians(init["roll"]))
    if mode == "free":
        ang0 = [clipr(np.radians(init["pitch"])), clipr(np.radians(init["roll"]))]
    elif mode == "plane":
        ang0 = [clipr(np.arctan2(d_init @ w0, d_init @ u0))]
    else:
        ang0 = []
    p0 = np.concatenate([init["p0vec"], init["v0vec"],
                         ([] if fix_g else [float(np.clip(init["g"], 3.0, 25.0))]) + ang0])
    lo = np.full(len(p0), -np.inf); hi = np.full(len(p0), np.inf)
    if na:
        lo[-na:], hi[-na:] = -ANG, ANG
    if not fix_g:
        lo[6], hi[6] = 2.0, 30.0
    sig_xy, sig_d = 1.0, 3.0
    for _ in range(2):
        def res(q, sxy=sig_xy, sd=sig_d):
            xp, yp, dp = project(q)
            return np.concatenate([(x - xp) / sxy, (y - yp) / sxy, (d - dp) / sd])
        r = least_squares(res, x0=p0, loss="soft_l1", f_scale=2.5, bounds=(lo, hi))
        p0 = r.x
        xp, yp, dp = project(r.x)
        rxy = np.concatenate([x - xp, y - yp]); rd = d - dp
        # 残差のロバストな広がり(MAD)で σ を更新
        sig_xy = max(1.4826 * np.median(np.abs(rxy - np.median(rxy))), 0.2)
        sig_d = max(1.4826 * np.median(np.abs(rd - np.median(rd))), 0.2)
    J = r.jac
    n_obs = len(r.fun); dof = max(1, n_obs - len(r.x))
    s2 = max(np.sum(r.fun ** 2) / dof, 1.0)       # 重みが楽観的なら共分散を膨らませる
    cov = np.linalg.pinv(J.T @ J) * s2
    _, g_fit, dv = decode(r.x)
    P, V = sim(r.x)
    v0n = float(np.linalg.norm(V[0]))
    grad = np.zeros(len(r.x)); grad[3:6] = V[0] / max(v0n, 1e-9)
    v0_err = float(np.sqrt(max(grad @ cov @ grad, 0)))
    Pm = np.column_stack([(x - cx) * Dm / (d - d_bias), (y - cy) * Dm / (d - d_bias),
                          focal_px * Dm / (d - d_bias)])
    return {"v0vec": V[0], "v0": v0n, "v0_err": v0_err, "v_end": float(np.linalg.norm(V[-1])),
            "g": float(g_fit), "g_err": 0.0 if fix_g else float(np.sqrt(cov[6, 6])), "g_fixed": fix_g,
            "pitch": float(np.degrees(np.arcsin(np.clip(dv[2], -1, 1)))),
            "roll": float(np.degrees(np.arctan2(dv[0], dv[1]))), "grav_mode": mode,
            "rms": float(np.sqrt(np.mean(np.sum((P - Pm) ** 2, 1)) / 3)),
            "n": int(len(t)), "d_med": float(np.median(d)), "sig_xy": float(sig_xy), "sig_d": float(sig_d),
            "p0vec": P[0]}


def run_3d(args, path):
    """--detector bgsub: 背景差分追跡 + 球径による3D再構成で打球初速を出す。"""
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    f0 = args.f_start if args.f_start is not None else 0
    f1 = args.f_end if args.f_end is not None else n - 1
    tr = track_bgsub(path, f0, f1)
    if tr is None:
        print("飛球の軌跡が見つかりません。--f-start/--f-end で飛行区間を指定してください。")
        sys.exit(1)
    F, X, Y, D, W, H = tr
    keep = diameter_outlier_mask(D)
    n_drop = int((~keep).sum())
    F, X, Y, D = F[keep], X[keep], Y[keep], D[keep]
    if len(F) < 10:
        print(f"追跡 {len(F)} 点で不足。")
        sys.exit(1)
    # ローリングシャッター: 行 y の撮影時刻は τ·y/H 遅れる(横長動画は縦に読み出す)
    t = (F - F[0]) / args.fps + (args.readout_ms / 1000.0) * (Y - Y[0]) / H
    # 下げ振り(糸)から重力の向き
    grav, plumb_msg = None, "使わない(--plumb none)"
    if args.plumb != "none":
        import plumb as pl
        lines = pl.parse_manual(args.plumb) if args.plumb != "auto" else pl.find_plumb_lines(pl.sample_frames(path))
        grav = pl.gravity_constraint(lines, args.focal_px, W / 2.0, H / 2.0)
        if grav is None:
            plumb_msg = "糸が見つからない → 向きも推定"
        elif grav[0] == "fixed":
            th, ph = pl.gravity_angles(grav[1])
            plumb_msg = f"糸{len(lines)}本で向きを固定(ピッチ{th:+.1f}° ロール{ph:+.1f}°)"
        else:
            L = lines[0]
            plumb_msg = (f"糸1本 x={(L[0]+L[2])/2:.0f}px(傾き{np.degrees(np.arctan2(L[2]-L[0], L[3]-L[1])):+.1f}°)"
                         "で向きを1つに絞る(2本あれば完全に決まる)")
    r3 = fit_3d(t, X, Y, D - args.d_bias, args.focal_px, W / 2.0, H / 2.0)
    r = r3 if args.fit == "3d" else fit_image(t, X, Y, D, args.focal_px, W / 2.0, H / 2.0,
                                               init=r3, d_bias=args.d_bias, fix_g=args.fix_g, grav=grav)
    v = r["v0vec"]
    g_dev = (r["g"] - G_CM / 100) / (G_CM / 100)
    print(f"動画        : {path.name}")
    _fname = "画像空間フィット" if args.fit == "image" else "3D位置フィット"
    print(f"検出方式    : 背景差分+予測追跡 → 球径で3D再構成・{_fname}"
          f" (f={args.focal_px:.0f}px, 径の偏り{args.d_bias:+.1f}px, Cd={DRAG_CD}, τ={args.readout_ms}ms)")
    print(f"追跡        : {len(F)} 点 (f{int(F[0])}-{int(F[-1])}) / フィット {r['n']} 点, 飛行 {t[-1]*1000:.0f} ms"
          + (f"  (直径が外れた {n_drop} 点を除外)" if n_drop else ""))
    print(f"重力の向き  : {plumb_msg}")
    print(f"球の見かけ径: {D.min():.0f}〜{D.max():.0f}px (中央 {r['d_med']:.0f}px = カメラから約 {args.focal_px*BALL_DIAM_CM/r['d_med']:.0f}cm)")
    print()
    if "v0_err" in r:
        print(f"  ★ 打球初速 = {r['v0']:.2f} ± {r['v0_err']:.2f} m/s = {r['v0']*3.6:.1f} ± {r['v0_err']*3.6:.1f} km/h"
              f"  (3次元・空気抵抗モデル・画像空間フィット)")
        print("    ※ ± は観測ノイズによる統計誤差のみ。焦点距離・直径の偏り・回転(マグヌス力)による"
              "系統誤差は含まない(9/23の打球では重力のずれから見て数%)")
        print(f"    観測ノイズ(推定): 位置 {r['sig_xy']:.2f}px, 直径 {r['sig_d']:.2f}px"
              f"   / 参考: 3D位置フィットの初速 {r3['v0']*3.6:.1f} km/h, g={r3['g']:.2f}")
    else:
        print(f"  ★ 打球初速 = {r['v0']:.2f} m/s = {r['v0']*3.6:.1f} km/h  (3次元・空気抵抗モデル)")
    print(f"    内訳: 横 {v[0]*3.6:+.1f} / 縦 {v[1]*3.6:+.1f} / 奥行き {v[2]*3.6:+.1f} km/h"
          f"  (奥行き成分 {abs(v[2])/r['v0']*100:.0f}%)")
    print(f"    飛行終端 {r['v_end']*3.6:.1f} km/h,  位置残差 {r['rms']*1000:.1f} mm")
    if r.get("g_fixed"):
        print(f"  重力: g = g_eff = {G_CM/100:.2f} m/s² に固定(--fix-g)。重力による検証はなし"
              f"  カメラ傾き ピッチ{r['pitch']:+.0f}° ロール{r['roll']:+.0f}°")
        return
    print(f"  検証: フィットした重力 g = {r['g']:.2f} ± {r['g_err']:.2f} m/s²"
          f"  (理論 g_eff={G_CM/100:.2f} との差 {g_dev*100:+.1f}%)"
          f"  カメラ傾き ピッチ{r['pitch']:+.0f}° ロール{r['roll']:+.0f}°")
    if abs(g_dev) > 0.10:
        print("    ⚠ 重力が10%超ずれています。球が近すぎてピンボケ/画面端で欠けている/"
              "強い回転(マグヌス力)の可能性。速度も同程度ずれている恐れ")
    else:
        print("    → 重力が一致: スケール(球径)・時間(fps)・物理モデルが整合")


def main():
    ap = argparse.ArgumentParser(description="打球速度 計測")
    ap.add_argument("--video", required=True, help="動画ファイル（VIDEO_DIR相対 or フルパス）")
    ap.add_argument("--fps", type=float, default=EFFECTIVE_FPS)
    ap.add_argument("--v-low", type=int, default=160, help="白判定の最小輝度V(0-255)")
    ap.add_argument("--s-high", type=int, default=55, help="白判定の最大彩度S(0-255、低いほど手・背景を除外)")
    ap.add_argument("--r-min", type=int, default=8, help="ボール最小半径px")
    ap.add_argument("--r-max", type=int, default=80, help="ボール最大半径px")
    ap.add_argument("--min-area", type=int, default=100, help="検出最小面積px^2")
    ap.add_argument("--f-start", type=int, default=None)
    ap.add_argument("--f-end", type=int, default=None)
    ap.add_argument("--detector", choices=["classical", "ai", "streak", "bgsub"], default="classical",
                    help="ボール検出方式。ai=学習済みYOLOv8(ONNX)。"
                         "streak=高速球むけ(運動ブラーで伸びた球を受け入れ、短辺を真の直径にする)。"
                         "bgsub=背景差分+予測追跡+球径による3D再構成(暖色照明・奥行き移動に強い、推奨)")
    ap.add_argument("--focal-px", type=float, default=FOCAL_PX,
                    help="bgsub の3D再構成に使う焦点距離[px]。奥行き成分にのみ効く(±30%%で初速±2%%程度)")
    ap.add_argument("--fit", choices=["image", "3d"], default="image",
                    help="bgsub のフィット方式。image=3D運動を画像に投影して(x,y,d)と比較(推奨)。"
                         "3d=直径からZを出して3D位置をフィット(旧方式)")
    ap.add_argument("--d-bias", type=float, default=0.0,
                    help="直径の測り方の系統的な偏り[px]。静止球を既知距離に置いた d=f·D/Z+b の較正値 b")
    ap.add_argument("--plumb", default="auto",
                    help="下げ振り(おもり付きの糸)で重力の向きを決める。auto=動画から自動で探す、none=使わない、"
                         "または糸の両端の座標 x1,y1,x2,y2 (2本なら ; で区切る)。左右の端に1本ずつあると向きが完全に決まる")
    ap.add_argument("--fix-g", action="store_true",
                    help="重力の大きさを g_eff に固定(速い球むけ。飛行が短いと g が決まらず初速がぶれる)。"
                         "この場合 g による検証はできない")
    ap.add_argument("--max-aspect", type=float, default=8.0,
                    help="streak検出で許す 長辺/短辺 の上限")
    ap.add_argument("--readout-ms", type=float, default=0.0,
                    help="ローリングシャッターの読み出し時間τ[ms]。行yの撮影時刻を τ·y/H 遅らせて補正。"
                         "rs_calib.py で較正した値を与える(既定0=補正なし)。240fpsなら4.17ms未満")
    ap.add_argument("--scale", choices=["auto", "gravity", "ball"], default="auto",
                    help="px/cmの決め方。gravity=自由落下のy(t)から重力で較正"
                         "(検出器に依存せず、高速球でブラーの影響を受けない)。"
                         "ball=ボール直径40mmから較正。auto=重力が使えれば重力")
    ap.add_argument("--onnx", default=None, help="AI検出のONNXパス(既定はai_detect.DEFAULT_ONNX)")
    ap.add_argument("--ai-conf", type=float, default=0.25, help="AI検出の信頼度閾値")
    args = ap.parse_args()

    path = Path(args.video)
    if not path.is_absolute():
        path = VIDEO_DIR / path
    if not path.exists():
        print(f"エラー: {path} が見つかりません")
        sys.exit(1)
    if args.detector == "bgsub":
        run_3d(args, path)
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("エラー: 動画を開けません")
        sys.exit(1)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr_h = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080.0   # ローリングシャッター補正用
    f0 = args.f_start if args.f_start is not None else 0
    f1 = args.f_end if args.f_end is not None else n - 1

    det_ai = None
    if args.detector == "ai":
        from ai_detect import BallDetector, DEFAULT_ONNX
        det_ai = BallDetector(args.onnx or DEFAULT_ONNX, conf=args.ai_conf)

    F, X, Y, R = [], [], [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    for fno in range(f0, f1 + 1):
        ok, fr = cap.read()
        if not ok:
            break
        if det_ai is not None:
            b = det_ai.detect_best(fr)      # (cx, cy, w, h, conf)
            d = (b[0], b[1], (b[2] + b[3]) / 4.0) if b else None   # 半径=(w+h)/2/2
        elif args.detector == "streak":
            d = detect_ball_streak(fr, args.v_low, args.s_high, args.r_min,
                                   args.r_max, args.min_area, args.max_aspect)
        else:
            d = detect_white_ball(fr, args.v_low, args.s_high,
                                  args.r_min, args.r_max, args.min_area)
        if d:
            F.append(fno); X.append(d[0]); Y.append(d[1]); R.append(d[2])
    cap.release()

    if len(F) < 4:
        print(f"検出 {len(F)} 点で不足。", end="")
        if det_ai is not None:
            print("--ai-conf を下げる(例 0.15)か、--detector classical を試してください。")
        else:
            print("--v-low を下げる/--s-high を上げる、または --detector ai を試してください。")
        sys.exit(1)

    F = np.array(F); X = np.array(X); Y = np.array(Y); R = np.array(R)
    # 最長飛行区間
    idx = longest_run(list(F))
    F, X, Y, R = F[idx], X[idx], Y[idx], R[idx]

    # 半径の外れ値除去(ブラーで潰れた点)→ 自己校正スケール
    rmed0 = np.median(R)
    keep = (R > 0.55 * rmed0) & (R < 1.8 * rmed0)
    rmed = np.median(R[keep]) if keep.sum() >= 3 else rmed0
    ppcm = 2 * rmed / BALL_DIAM_CM

    # ── ローリングシャッター補正 ──
    # センサーは画面上の行から順に読み出すため、行 y の実撮影時刻は
    #   t = n/fps + τ·(y/H)   (τ=1フレームの読み出し時間)
    # だけ遅れる。縦方向に動く球では有効なフレーム間隔が伸縮し、加速度に偏りが出る
    # (落下検証では抗力込みで+17%のうち、τ≈3-4msで半分程度を説明できる)。
    # 横方向の運動では影響は小さい。τ は rs_calib.py 等で較正して与える(既定0=無効)。
    if args.readout_ms > 0:
        t = (F - F[0]) / args.fps + (args.readout_ms / 1000.0) * (Y / fr_h)
        t = t - t[0]
    else:
        t = (F - F[0]) / args.fps

    # ── 重力によるスケール較正 ──
    # 飛球は自由落下中なので、y(t)の放物線フィットの2次係数が g[px/s^2] を与える。
    # g=9.81m/s^2 は検出器に依存しない物理定数なので px/cm = g_px/981 が得られる。
    # ボール径による較正は検出器ごとに10〜20%ずれる(最小外接円は過大、
    # ストリークの短辺は過小、AI枠はラベル定義由来で過大)のに対し、
    # 重力較正では3検出器の速度が±4%に収束することを実測で確認済み。
    # 高速球ほどブラーで球径較正は壊れるが、重力較正は影響を受けない。
    ppcm_g = None
    g_rel_err = None
    if keep.sum() >= 6:
        try:
            ycoef, ycov = np.polyfit(t[keep], Y[keep], 2, cov=True)
            g_px = 2 * ycoef[0]                      # y下向き正 → 重力は正
            g_se = 2 * np.sqrt(max(ycov[0, 0], 0))
            if g_px > 0 and g_se < 0.5 * g_px:
                ppcm_g = g_px / G_CM                 # 実効重力 959 cm/s^2
                g_rel_err = g_se / g_px
        except np.linalg.LinAlgError:
            pass

    # スケール源の決定
    scale_src = args.scale
    if scale_src == "auto":
        scale_src = "gravity" if (ppcm_g is not None and g_rel_err < 0.15) else "ball"
    if scale_src == "gravity" and ppcm_g is None:
        print("  ⚠ 重力較正に失敗(落下成分が弱い/検出点不足)。ボール径較正に切替")
        scale_src = "ball"

    if scale_src == "gravity":
        # 重力較正は1本の定数スケール(奥行き補正は使わない)
        ppcm_t = np.full(len(t), ppcm_g)
        depth_change = 0.0
    else:
        # フレーム毎の局所スケール: 半径を時間の1次でモデル化(奥行き移動に追従)
        if keep.sum() >= 3:
            rcoef = np.polyfit(t[keep], R[keep], 1)
        else:
            rcoef = np.array([0.0, rmed])
        r_model = np.clip(np.polyval(rcoef, t), 0.5 * rmed, 2.0 * rmed)
        ppcm_t = 2 * r_model / BALL_DIAM_CM      # 各フレームの px/cm
        depth_change = (r_model[-1] - r_model[0]) / rmed   # 径変化=奥行き移動の指標

    # PCA主軸へ射影 → 進行距離[px] → 局所スケールで cm 換算
    pts = np.column_stack([X, Y])
    c = pts.mean(0)
    _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis_u = vt[0].copy()
    s_px = (pts - c) @ axis_u
    if s_px[-1] < s_px[0]:
        s_px = -s_px
        axis_u = -axis_u          # 進行方向を正にした主軸の単位ベクトル
    linearity = sv[0] / sv[1] if sv[1] > 0 else float("inf")

    ppcm_mid = (ppcm_t[:-1] + ppcm_t[1:]) / 2
    s_cm = np.concatenate([[0.0], np.cumsum(np.diff(s_px) / ppcm_mid)])

    slope_cm, coef, mask = robust_linear(t, s_cm)   # cm/s
    v_ms = abs(slope_cm) / 100.0
    v_kmh = v_ms * 3.6

    # 区間速度の範囲(参考)
    seg = np.abs(np.diff(s_cm)) / np.diff(t) / 100.0
    seg = seg[np.isfinite(seg)]

    # ── 空気抵抗モデルによる初速推定 ──
    # 等速フィット(v_ms)は飛行中の「平均速度」。打球の初速 v0 を得るには
    # 抗力 k·v² と重力の主軸成分で減速する運動方程式でフィットする。
    area_m2 = np.pi * (BALL_DIAM_CM / 200.0) ** 2
    k_m = AIR_DENSITY * DRAG_CD * area_m2 / (2 * BALL_MASS_KG) / _ADDED   # [1/m] 付加質量込み
    k_cm = k_m / 100.0                                              # [1/cm]
    g_along = G_CM * axis_u[1]      # 画像y下向き=重力方向。主軸のy成分が重力の寄与
    drag = None
    try:
        drag = drag_fit(t[mask], s_cm[mask], g_along, k_cm)
    except Exception as e:      # フィット失敗時は等速結果のみ報告
        drag_err = str(e)
    # 比較用: 二次フィット(定加速度)から見かけの減速度
    qcoef = np.polyfit(t[mask], s_cm[mask], 2)
    a_quad = -2 * qcoef[0]          # cm/s^2 (正=減速)
    v0_quad = qcoef[1]              # cm/s
    a_drag_pred = k_cm * v0_quad ** 2 - g_along   # 抗力+重力が予測する減速度

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"動画        : {path.name}")
    _dname = {"ai": "AI(YOLOv8 ONNX)", "streak": "ストリーク(高速球むけ)",
              "classical": "古典CV(HSV白球)"}[args.detector]
    print(f"fps         : {args.fps}   検出方式: {_dname}")
    print(f"検出/採用   : {len(F)} 点 / フィット {int(mask.sum())} 点")
    print(f"軌跡直線性  : 主軸が副軸の {linearity:.1f} 倍")
    print(f"ボール半径  : 中央値 {rmed:.0f}px → 直径 {2*rmed:.0f}px,  px/cm={ppcm:.1f}(球径較正)")
    if scale_src == "gravity":
        print(f"★スケール源 : 重力較正 px/cm={ppcm_g:.1f} ± {ppcm_g*g_rel_err:.1f}"
              f"  (球径較正 {ppcm:.1f} との差 {(ppcm_g-ppcm)/ppcm*100:+.0f}%)")
    else:
        print(f"★スケール源 : 球径較正 px/cm {ppcm_t[0]:.1f}→{ppcm_t[-1]:.1f}"
              f" (フレーム毎、径変化 {depth_change*100:+.0f}%)")
    print(f"飛行時間    : {t[-1]*1000:.0f} ms")
    print()
    _sname = "重力較正" if scale_src == "gravity" else "球径較正(局所スケール)"
    if drag is not None:
        v0_ms = drag["v0"] / 100.0
        vend_ms = drag["v_end"] / 100.0
        print(f"  ★ 打球初速 = {v0_ms:.2f} m/s = {v0_ms*3.6:.1f} km/h  (空気抵抗モデル・{_sname})")
        print(f"    飛行終端 {vend_ms*3.6:.1f} km/h  /  平均(等速フィット) {v_kmh:.1f} km/h"
              f"  /  減速 {(1-vend_ms/v0_ms)*100:.0f}%  残差RMS {drag['resid_rms']*10:.1f}mm")
        # 抗力モデルの妥当性: 二次フィットの見かけ減速度 vs 物理予測
        print(f"    減速度の検証: 実測 {a_quad/100:.2f} m/s²  vs 抗力+重力の予測 {a_drag_pred/100:.2f} m/s²"
              f"  (Cd={DRAG_CD}, 重力の主軸成分 {g_along/100:+.2f} m/s²)")
        print(f"    参考: 減速度を自由にした二次フィットの初速 = {v0_quad/100*3.6:.1f} km/h"
              f"  (物理モデル {v0_ms*3.6:.1f} との差が「説明できない減速」の大きさ)")
        if a_quad > 0 and abs(a_quad - a_drag_pred) / max(a_quad, 1e-6) > 0.5:
            print("    ⚠ 実測減速が物理予測と50%超乖離。回転(マグヌス)や奥行き移動の可能性")
    else:
        print(f"  ★ 打球速度 = {v_ms:.2f} m/s = {v_kmh:.1f} km/h  (等速フィット・{_sname})")
        print(f"    (空気抵抗モデルのフィットに失敗: {drag_err})")
    if len(seg):
        print(f"    区間速度の範囲: {seg.min()*3.6:.1f} 〜 {seg.max()*3.6:.1f} km/h")
    if ppcm_g is not None and scale_src != "gravity":
        dev = (ppcm_g - ppcm) / ppcm * 100
        print(f"  重力チェック: g由来 px/cm = {ppcm_g:.1f} ± {ppcm_g*g_rel_err:.1f}"
              f"  (球径較正との差 {dev:+.0f}%)")
        if abs(dev) > 15:
            print("    ⚠ 15%超の乖離。--scale gravity の方が検出器に依存せず安定します")
        else:
            print("    → 校正とfpsはオーダーで整合(自由落下との一致。±10-15%は空気抵抗等で正常)")
    if abs(depth_change) > 0.15:
        print(f"  ⚠ 飛行中に見かけ径が {depth_change*100:+.0f}% 変化 = 奥行き移動あり。"
              "局所スケールで補正済みだが、飛球線に垂直なカメラ位置を推奨")
    if len(F) < 8:
        print("  ⚠ 検出点が少なめ。明るく短露光・球が全幅を横切る撮影で精度↑")
    if linearity < 4:
        print("  ⚠ 軌跡が直線的でない（奥行き移動の可能性）。飛球線に垂直なカメラ位置を推奨")

    # CSV
    stem = path.stem
    with open(OUT_DIR / f"{stem}_track.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x_px", "y_px", "r_px", "s_cm"])
        for i in range(len(F)):
            w.writerow([F[i], f"{t[i]:.5f}", f"{X[i]:.1f}", f"{Y[i]:.1f}",
                        f"{R[i]:.1f}", f"{s_cm[i]:.2f}"])

    # プロット
    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    ax[0].scatter(X[mask], Y[mask], s=20, c="steelblue", label="ball (used)")
    if (~mask).any():
        ax[0].scatter(X[~mask], Y[~mask], s=20, c="red", marker="x", label="outlier")
    ax[0].invert_yaxis(); ax[0].set_aspect("equal", "box")
    ax[0].set_xlabel("x [px]"); ax[0].set_ylabel("y [px]")
    ax[0].set_title(f"Trajectory — {stem}  ({v_kmh:.1f} km/h)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    tf = np.linspace(t.min(), t.max(), 100)
    ax[1].scatter(t, s_cm, s=20, c="steelblue")
    ax[1].plot(tf, np.polyval(coef, tf), "r--",
               label=f"const-speed: {v_kmh:.1f} km/h (mean)")
    if drag is not None:
        ax[1].plot(t[mask], drag["s_pred"], "g-", linewidth=1.5,
                   label=f"drag model: v0={drag['v0']/100*3.6:.1f} km/h, end={drag['v_end']/100*3.6:.1f}")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("distance [cm]")
    ax[1].set_title("Distance vs time: constant-speed vs air-drag model")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f"{stem}_speed.png"), dpi=140)
    plt.close()
    print(f"\n  出力: {OUT_DIR / (stem + '_speed.png')}")
    print(f"        {OUT_DIR / (stem + '_track.csv')}")


if __name__ == "__main__":
    main()
