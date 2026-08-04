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
    ap.add_argument("--detector", choices=["classical", "ai"], default="classical",
                    help="ボール検出方式。ai=学習済みYOLOv8(ONNX)。"
                         "打球では古典46%%に対しAI100%%と大幅に良好")
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

    # フレーム毎の局所スケール: 半径を時間の1次でモデル化(奥行き移動に追従)
    if keep.sum() >= 3:
        rcoef = np.polyfit(t[keep], R[keep], 1)
    else:
        rcoef = np.array([0.0, rmed])
    r_model = np.clip(np.polyval(rcoef, t), 0.5 * rmed, 2.0 * rmed)
    ppcm_t = 2 * r_model / BALL_DIAM_CM          # 各フレームの px/cm
    depth_change = (r_model[-1] - r_model[0]) / rmed   # 相対的な径変化=奥行き移動の指標

    # PCA主軸へ射影 → 進行距離[px] → 局所スケールで cm 換算
    pts = np.column_stack([X, Y])
    c = pts.mean(0)
    _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    s_px = (pts - c) @ vt[0]
    if s_px[-1] < s_px[0]:
        s_px = -s_px
    linearity = sv[0] / sv[1] if sv[1] > 0 else float("inf")

    ppcm_mid = (ppcm_t[:-1] + ppcm_t[1:]) / 2
    s_cm = np.concatenate([[0.0], np.cumsum(np.diff(s_px) / ppcm_mid)])

    slope_cm, coef, mask = robust_linear(t, s_cm)   # cm/s
    v_ms = abs(slope_cm) / 100.0
    v_kmh = v_ms * 3.6

    # 区間速度の範囲(参考)
    seg = np.abs(np.diff(s_cm)) / np.diff(t) / 100.0
    seg = seg[np.isfinite(seg)]

    # ── 重力クロスチェック ──
    # 飛球は自由落下中: y(t) の放物線フィットから g_px[px/s²] を推定し、
    # px/cm = g_px/981 をボール径校正と独立に得る。
    # ※空気抵抗・マグヌス力で±10〜15%程度は乖離しうる粗い検証(大きな校正ミスの検出用)。
    ppcm_g = None
    g_rel_err = None
    if keep.sum() >= 6:
        try:
            ycoef, ycov = np.polyfit(t[keep], Y[keep], 2, cov=True)
            g_px = 2 * ycoef[0]                      # y下向き正 → 重力は正
            g_se = 2 * np.sqrt(max(ycov[0, 0], 0))
            if g_px > 0 and g_se < 0.5 * g_px:
                ppcm_g = g_px / 981.0                # 981 cm/s²
                g_rel_err = g_se / g_px
        except np.linalg.LinAlgError:
            pass

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"動画        : {path.name}")
    print(f"fps         : {args.fps}   検出方式: "
          f"{'AI(YOLOv8 ONNX)' if det_ai is not None else '古典CV(HSV白球)'}")
    print(f"検出/採用   : {len(F)} 点 / フィット {int(mask.sum())} 点")
    print(f"軌跡直線性  : 主軸が副軸の {linearity:.1f} 倍")
    print(f"ボール半径  : 中央値 {rmed:.0f}px → 直径 {2*rmed:.0f}px,  px/cm={ppcm:.1f}(自己校正)")
    print(f"局所スケール: px/cm {ppcm_t[0]:.1f}→{ppcm_t[-1]:.1f} (フレーム毎、径変化 {depth_change*100:+.0f}%)")
    print(f"飛行時間    : {t[-1]*1000:.0f} ms")
    print()
    print(f"  ★ 打球速度 = {v_ms:.2f} m/s = {v_kmh:.1f} km/h  (等速フィット・局所スケール)")
    if len(seg):
        print(f"    区間速度の範囲: {seg.min()*3.6:.1f} 〜 {seg.max()*3.6:.1f} km/h")
    if ppcm_g is not None:
        dev = (ppcm_g - ppcm) / ppcm * 100
        print(f"  重力チェック: g由来 px/cm = {ppcm_g:.1f} ± {ppcm_g*g_rel_err:.1f}"
              f"  (ボール径校正との差 {dev:+.0f}%)")
        if abs(dev) > 30:
            print("    ⚠ 30%超の乖離: 校正かfpsに系統誤差の疑い(空気抵抗/回転でも±10-15%は乖離しうる)")
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
               label=f"{v_ms:.2f} m/s = {v_kmh:.1f} km/h")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("distance [cm]")
    ax[1].set_title("Distance vs time (constant-speed fit)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / f"{stem}_speed.png"), dpi=140)
    plt.close()
    print(f"\n  出力: {OUT_DIR / (stem + '_speed.png')}")
    print(f"        {OUT_DIR / (stem + '_track.csv')}")


if __name__ == "__main__":
    main()
