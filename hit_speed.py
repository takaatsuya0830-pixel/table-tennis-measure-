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
DRAG_CD = 0.45                 # 球のCd(Re~1e4-1e5で0.4-0.5)
G_CM = 981.0                   # cm/s^2
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
    ap.add_argument("--detector", choices=["classical", "ai", "streak"], default="classical",
                    help="ボール検出方式。ai=学習済みYOLOv8(ONNX)。"
                         "streak=高速球むけ(運動ブラーで伸びた球を受け入れ、短辺を真の直径にする)。"
                         "打球では古典46%%に対しAI100%%と大幅に良好")
    ap.add_argument("--max-aspect", type=float, default=8.0,
                    help="streak検出で許す 長辺/短辺 の上限")
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

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print("エラー: 動画を開けません")
        sys.exit(1)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                ppcm_g = g_px / 981.0                # 981 cm/s^2
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
    k_m = AIR_DENSITY * DRAG_CD * area_m2 / (2 * BALL_MASS_KG)   # [1/m]
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
