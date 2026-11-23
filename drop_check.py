"""
落下の高さで速さを答え合わせする  ── 2026-11-10

ボールを静かに(初速0で)離した高さを巻尺で測っておけば、画面を通るときの速さは
物理(重力+浮力+空気抵抗)だけで決まる。映像から測った速さがこれと合えば、
時間(240fps)と長さ(球の大きさ)の基準が正しいことの確認になる。

計算: 球の位置を hit_speed と同じ方法(背景差分の追跡・球の直径で長さを決める)で求め、
糸2本(plumb.py)で決めた真下の向きに沿って「レンズからの高さ」に直す。
それに「高さ H から時刻 t_r に静かに離した」落下の式を当てはめ、未知数は H と t_r の2つだけ。
  z(t) = H − ln(cosh(√(g k)·(t − t_r))) / k     (g=実効重力, k=抗力の係数)
映像から逆算した H と巻尺の高さ、画面に入ったときの速さ(映像)と高さからの計算値を比べる。

2026-11-09 の落下3本(離した高さ 床240cm・レンズ100cm → レンズ+140cm):
  映像から逆算 139.4 / 138.5 / 137.9 cm、速さの差 −0.3 / −0.7 / −0.9 %。
  高さを1cm測り間違えると速さは約0.5%変わる。

使い方:
  python drop_check.py --video PXL_20261109_102910714.mp4 --release-floor-cm 240 --lens-cm 100
  python drop_check.py --video <動画> --release-cm 140             (レンズからの高さを直接)
  python drop_check.py --video <動画> --release-cm 140 --depth-cm 116.5
      (球が糸の線の上を落ちたとき、糸までの奥行きで長さを決める=糸のものさし)
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import hit_speed as hs       # 読み込むと標準出力が UTF-8 になる
import plumb as pl


def drop_k_per_cm():
    """抗力の係数 k [1/cm] (減速度 = k·v²)。hit_speed の打球モデルと同じ値。"""
    k_si = hs.AIR_DENSITY * hs.DRAG_CD * np.pi * (hs.BALL_DIAM_CM / 200.0) ** 2 / (2 * hs.BALL_MASS_KG * hs._ADDED)
    return k_si / 100.0


def fall_dist(T, g, k):
    """静止から落ちて時間 T で進む距離[cm]"""
    T = np.maximum(T, 0.0)
    return np.log(np.cosh(np.sqrt(g * k) * T)) / k


def fall_speed(T, g, k):
    """静止から落ちて時間 T 後の速さ[cm/s]"""
    return np.sqrt(g / k) * np.tanh(np.sqrt(g * k) * np.maximum(T, 0.0))


def speed_after_drop(h, g, k):
    """静止から高さ h[cm] 落ちたときの速さ[cm/s]"""
    if h <= 0:
        return 0.0
    T = np.arccosh(np.exp(k * h)) / np.sqrt(g * k)
    return float(fall_speed(T, g, k))


def heights_above_lens(X, Y, D, W, H, focal_px, gdir, depth_cm=None):
    """画像上の位置をレンズからの高さ[cm]に直す。gdir は真下の向き(カメラ座標の単位ベクトル)。
    長さの基準(1pxが何cmか)は、球の直径(中央値)から、または depth_cm(糸の線までの奥行き)から。"""
    s = (depth_cm / focal_px) if depth_cm else (hs.BALL_DIAM_CM / np.median(D))
    P = np.stack([(X - W / 2) * s, (Y - H / 2) * s, np.full_like(X, focal_px * s)], axis=1)
    return -(P @ gdir), s


def fit_release(t, z, g, k):
    """z(t) = H − fall(t − t_r) を当てはめる。Returns H, t_r, 残差rms[cm]"""
    res = lambda q: z - (q[0] - fall_dist(t - q[1], g, k))
    q = least_squares(res, [z[0] + 100.0, -0.5], loss="soft_l1", f_scale=0.3).x
    return float(q[0]), float(q[1]), float(np.sqrt(np.mean(res(q) ** 2)))


def main():
    ap = argparse.ArgumentParser(description="落下の高さで速さを答え合わせする")
    ap.add_argument("--video", required=True)
    ap.add_argument("--release-cm", type=float, default=None, help="離した位置のレンズからの高さ[cm]")
    ap.add_argument("--release-floor-cm", type=float, default=None, help="離した位置の床からの高さ[cm]")
    ap.add_argument("--lens-cm", type=float, default=None, help="レンズの床からの高さ[cm]")
    ap.add_argument("--depth-cm", type=float, default=None,
                    help="球が落ちた位置までの奥行き[cm](糸の線の上で落としたとき)。省略時は球の直径で長さを決める")
    ap.add_argument("--f-start", type=int, default=None)
    ap.add_argument("--f-end", type=int, default=None)
    ap.add_argument("--fps", type=float, default=hs.EFFECTIVE_FPS)
    ap.add_argument("--focal-px", type=float, default=hs.FOCAL_PX)
    ap.add_argument("--readout-ms", type=float, default=0.0,
                    help="ローリングシャッターの読み出し時間(+:上→下, −:下→上)。落下は縦の動きなので影響が大きい")
    args = ap.parse_args()

    if args.release_cm is None:
        if args.release_floor_cm is None or args.lens_cm is None:
            ap.error("--release-cm か、--release-floor-cm と --lens-cm の両方を指定してください")
        args.release_cm = args.release_floor_cm - args.lens_cm

    path = Path(args.video)
    if not path.is_absolute():
        path = hs.VIDEO_DIR / path
    import cv2
    cap = cv2.VideoCapture(str(path)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    f0 = args.f_start if args.f_start is not None else 0
    f1 = args.f_end if args.f_end is not None else n - 1

    tr = hs.track_bgsub(path, f0, f1, select="fall")
    if tr is None:
        print("落ちる球の軌跡が見つかりません。--f-start/--f-end で落下の区間を指定してください。")
        return
    F, X, Y, D, W, H = tr
    keep = hs.diameter_outlier_mask(D)          # True = 使う点
    F, X, Y, D = F[keep], X[keep], Y[keep], D[keep]
    if len(F) < 8:
        print(f"追跡 {len(F)} 点で不足。")
        return

    lines = pl.find_plumb_lines(pl.sample_frames(path))
    c = pl.gravity_constraint(lines, args.focal_px, W / 2, H / 2)
    if c is not None and c[0] == "fixed":
        gdir = c[1]
        th, ph = pl.gravity_angles(gdir)
        gtxt = f"糸2本で固定(ピッチ{th:+.1f}° ロール{ph:+.1f}°)"
    else:
        gdir = np.array([0.0, 1.0, 0.0])
        gtxt = "糸2本が見つからないので画面の真下と仮定(カメラの傾きの分だけ誤差が出ます)"

    g = hs.G_CM; k = drop_k_per_cm()
    t = (F - F[0]) / args.fps + (args.readout_ms / 1000.0) * (Y - H / 2) / H
    z, s = heights_above_lens(X, Y, D, W, H, args.focal_px, gdir, args.depth_cm)
    Hfit, tr_, rms = fit_release(t, z, g, k)

    v_video = fall_speed(-tr_, g, k)                              # 最初の点での速さ(映像から)
    v_phys = speed_after_drop(args.release_cm - z[0], g, k)       # 巻尺の高さから
    d_med = np.median(D)
    scale_txt = (f"糸の線までの奥行き {args.depth_cm:.1f}cm" if args.depth_cm
                 else f"球の直径 {d_med:.1f}px (カメラから約{args.focal_px * hs.BALL_DIAM_CM / d_med:.0f}cm)")

    print(f"動画        : {path.name}")
    print(f"追跡        : {len(F)} 点 (f{int(F[0])}-{int(F[-1])}), 飛行 {1000 * (t[-1] - t[0]):.0f} ms, "
          f"レンズからの高さ {z[0]:+.1f} → {z[-1]:+.1f} cm")
    print(f"真下の向き  : {gtxt}")
    print(f"長さの基準  : {scale_txt} → 1px = {10 * s:.3f} mm  (f={args.focal_px:.0f}px, τ={args.readout_ms}ms)")
    print(f"当てはまり  : 位置の残差 {10 * rms:.2f} mm")
    print()
    print(f"  離した高さ: 映像から {Hfit:.1f} cm  /  巻尺 {args.release_cm:.1f} cm  (差 {Hfit - args.release_cm:+.1f} cm)")
    dv = 100 * (v_video / v_phys - 1) if v_phys > 0 else float("nan")
    print(f"  ★ 画面に入ったときの速さ: 映像 {0.036 * v_video:.2f} km/h  /  高さからの計算 {0.036 * v_phys:.2f} km/h  (差 {dv:+.1f}%)")
    print(f"    ※ 高さを1cm測り間違えると速さは約{50 / max(args.release_cm - z[0], 1):.1f}%変わる。"
          f"読み出しの向きの仮定(--readout-ms ±3.88)で数%動く")


if __name__ == "__main__":
    main()
