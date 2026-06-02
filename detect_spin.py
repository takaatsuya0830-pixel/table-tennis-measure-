"""
ボール回転検出スクリプト

白いピンポン球に描いた黒い点の角度変化から回転数(rps)を計測する。

使い方:
  python detect_spin.py --video <動画パス>
  python detect_spin.py --video <動画パス> --min-r 40 --max-r 80 --param2 20
  python detect_spin.py --video <動画パス> --dark-thresh 60 --roi 0 100 1920 600

出力:
  frames/spin/spin_*.csv         : 各フレームの検出データ
  frames/spin/spin_*_plot.png    : 累積角度グラフ
  frames/spin/<動画名>/frame_*.png : 可視化フレーム画像
"""

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ─── 定数 ────────────────────────────────────────────────────
EFFECTIVE_FPS = 233

OUTPUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames")
# ─────────────────────────────────────────────────────────────


def imwrite(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite の日本語パス対応版。"""
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        open(str(path), "wb").write(buf.tobytes())


# ─── ボール検出（detect_ball.py から流用） ────────────────────

def _filter_roi(circles, roi, min_r, max_r):
    result = []
    for x, y, r in circles:
        if r < min_r or r > max_r:
            continue
        if roi is not None:
            x0, y0, x1, y1 = roi
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
        result.append((x, y, r))
    return result


def detect_by_hough(gray, roi=None, min_r=40, max_r=80, param2=30):
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
        param1=100, param2=param2,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return []
    raw = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    return _filter_roi(raw, roi, min_r, max_r)


def largest(circles):
    return max(circles, key=lambda c: c[2]) if circles else None


def nearest(circles, prev_x, prev_y, max_jump):
    if not circles:
        return None
    best = min(circles, key=lambda c: (c[0] - prev_x) ** 2 + (c[1] - prev_y) ** 2)
    dist = math.hypot(best[0] - prev_x, best[1] - prev_y)
    return best if dist <= max_jump else None


def pick(circles, prev, max_jump):
    """前フレーム追従方式で採用円を選ぶ。"""
    if prev is None:
        return largest(circles)
    return nearest(circles, prev[0], prev[1], max_jump)


# ─── 黒点検出 ─────────────────────────────────────────────────

def detect_black_dot(frame, cx, cy, radius, dark_thresh=60, min_area=8):
    """ボール内部の最大の暗いブロブ(黒点)を検出する。

    ボール外縁部(10%)を除いた内側だけを見る。
    Returns: (dot_x, dot_y) in original frame coordinates, or None
    """
    h, w = frame.shape[:2]
    r = max(1, int(radius * 0.90))  # 周縁部ノイズを除外
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(w, cx + r), min(h, cy + r)
    if x1 <= x0 or y1 <= y0:
        return None

    roi_img = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    # 円マスク: ボール外を 255(白)にして暗点として検出しないようにする
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx - x0, cy - y0), r, 255, -1)
    inv_mask = cv2.bitwise_not(mask)

    masked = cv2.bitwise_and(gray, gray, mask=mask)
    masked = cv2.add(masked, inv_mask)  # 円外 = 255(白)

    # 暗い領域を二値化
    _, thresh = cv2.threshold(masked, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    # ノイズ除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None

    largest_cnt = max(valid, key=cv2.contourArea)
    M = cv2.moments(largest_cnt)
    if M["m00"] == 0:
        return None

    dot_x = int(M["m10"] / M["m00"]) + x0
    dot_y = int(M["m01"] / M["m00"]) + y0
    return dot_x, dot_y


# ─── 角度アンラップ ───────────────────────────────────────────

def unwrap_step(new_angle, prev_angle, cum):
    """1ステップのアンラップ: 差を -180〜+180 に正規化して累積する。

    ※ 連続する有効フレーム間の差が ±180° 以内の場合のみ正しく動作する。
      点が長時間見えない場合は累積がずれる可能性がある。
    """
    diff = new_angle - prev_angle
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return cum + diff


# ─── 可視化 ──────────────────────────────────────────────────

def draw_frame(frame, ball, dot_xy, angle_deg, cum_deg, frame_idx, time_sec):
    vis = frame.copy()
    if ball is not None:
        cx, cy, r = ball
        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
        if dot_xy is not None:
            dx, dy = dot_xy
            cv2.circle(vis, (dx, dy), 6, (0, 0, 255), -1)
            cv2.line(vis, (cx, cy), (dx, dy), (0, 0, 255), 2)
            txt = f"{angle_deg:.1f}deg  cum={cum_deg:.0f}deg"
            cv2.putText(vis, txt, (cx - 90, cy + r + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(vis, f"f={frame_idx}  t={time_sec:.4f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    return vis


# ─── メイン ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ボール回転検出スクリプト")
    parser.add_argument("--video", required=True, help="動画ファイルパス")
    parser.add_argument("--fps", type=float, default=float(EFFECTIVE_FPS))
    parser.add_argument("--start", type=int, default=0, help="開始フレーム番号")
    parser.add_argument("--count", type=int, default=None, help="処理フレーム数")
    parser.add_argument("--min-r", type=int, default=40, help="ボール最小半径px")
    parser.add_argument("--max-r", type=int, default=80, help="ボール最大半径px")
    parser.add_argument("--param2", type=int, default=30, help="Hough param2")
    parser.add_argument("--max-jump", type=float, default=150.0,
                        help="前フレームからの最大追従距離px")
    parser.add_argument("--dark-thresh", type=int, default=60,
                        help="黒点検出の輝度閾値 0-255 (低=厳格, デフォルト:60)")
    parser.add_argument("--min-dot-area", type=int, default=8,
                        help="黒点の最小面積px² (デフォルト:8)")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        default=None)
    parser.add_argument("--save-every", type=int, default=10,
                        help="N フレームごとに可視化画像を保存 (デフォルト:10)")
    parser.add_argument("--preview", action="store_true", help="リアルタイムプレビュー")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"エラー: {video_path} が見つかりません")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("エラー: 動画を開けませんでした")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    process_count = args.count if args.count else total_frames - args.start

    roi = tuple(args.roi) if args.roi else None
    nyquist_rps = args.fps / 2.0

    # 出力ディレクトリ
    spin_dir = OUTPUT_DIR / "spin"
    spin_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = spin_dir / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = spin_dir / f"spin_{timestamp}.csv"
    plot_path = spin_dir / f"spin_{timestamp}_plot.png"

    print(f"動画        : {video_path.name}")
    print(f"実効fps     : {args.fps}")
    print(f"半径範囲    : {args.min_r} - {args.max_r} px")
    print(f"暗点閾値    : {args.dark_thresh}  最小面積: {args.min_dot_area} px^2")
    print(f"処理範囲    : フレーム {args.start} - {args.start + process_count - 1}")
    if roi:
        print(f"ROI         : x={roi[0]}-{roi[2]}  y={roi[1]}-{roi[3]}")
    print(f"ナイキスト  : {nyquist_rps:.1f} rps (これ以上は折り返し警告)")
    print(f"CSV出力     : {csv_path.name}")
    print()

    # ─── メインループ ───────────────────────────────────────
    records = []  # (frame_idx, time_sec, ball, dot_xy, angle_deg, cum_deg)
    prev_ball = None
    prev_angle = None
    cum_deg = 0.0
    ball_count = 0
    dot_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_number", "time_sec",
            "ball_cx", "ball_cy", "ball_r",
            "dot_x", "dot_y",
            "angle_deg", "cumulative_angle_deg",
        ])

        for i in range(process_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = args.start + i
            time_sec = frame_idx / args.fps

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            circles = detect_by_hough(gray, roi, args.min_r, args.max_r, args.param2)
            ball = pick(circles, prev_ball, args.max_jump)
            if ball is not None:
                prev_ball = ball
                ball_count += 1

            dot_xy = None
            angle_deg = None
            cum_this = None

            if ball is not None:
                cx, cy, r = ball
                dot_xy = detect_black_dot(frame, cx, cy, r,
                                          args.dark_thresh, args.min_dot_area)
                if dot_xy is not None:
                    dx, dy = dot_xy
                    # 画像座標系: x右が正、y下が正 → 時計回りが正角度
                    angle_deg = math.degrees(math.atan2(dy - cy, dx - cx))
                    if prev_angle is None:
                        cum_deg = angle_deg
                    else:
                        # 連続フレーム間の差が ±180° を超える場合は警告
                        diff = angle_deg - prev_angle
                        if diff > 180:
                            diff -= 360
                        elif diff < -180:
                            diff += 360
                        if abs(diff) > 90:
                            print(f"\n  注意: frame {frame_idx} で角度が {diff:.1f}° 急変 (ナイキスト付近の可能性)")
                        cum_deg = cum_deg + diff
                    prev_angle = angle_deg
                    cum_this = cum_deg
                    dot_count += 1

            def fmt(v, decimals=4):
                return f"{v:.{decimals}f}" if v is not None else ""
            def fmti(v):
                return str(int(v)) if v is not None else ""

            writer.writerow([
                frame_idx, f"{time_sec:.6f}",
                fmti(ball[0]) if ball else "", fmti(ball[1]) if ball else "",
                fmti(ball[2]) if ball else "",
                fmti(dot_xy[0]) if dot_xy else "", fmti(dot_xy[1]) if dot_xy else "",
                fmt(angle_deg), fmt(cum_this),
            ])
            records.append((frame_idx, time_sec, ball, dot_xy, angle_deg, cum_this))

            # フレーム画像保存（検出ありは必ず、その他は N フレームごと）
            if (i % args.save_every == 0) or (dot_xy is not None):
                vis = draw_frame(frame, ball, dot_xy,
                                 angle_deg or 0.0, cum_deg, frame_idx, time_sec)
                imwrite(frame_dir / f"frame_{frame_idx:05d}.png", vis)

            if i % 50 == 0:
                pct = i / process_count * 100
                print(
                    f"  [{pct:5.1f}%] f={frame_idx}"
                    f"  ball={'Y' if ball else 'N'}"
                    f"  dot={'Y' if dot_xy else 'N'}"
                    f"  cum={cum_deg:+.1f}deg",
                    end="\r",
                )

            if args.preview:
                vis = draw_frame(frame, ball, dot_xy,
                                 angle_deg or 0.0, cum_deg, frame_idx, time_sec)
                cv2.imshow("Spin Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if args.preview:
        cv2.destroyAllWindows()

    print()
    print(f"\n処理完了: {process_count} フレーム")
    print(f"  ボール検出: {ball_count} フレーム ({ball_count / process_count * 100:.1f}%)")
    print(f"  黒点検出 : {dot_count} フレーム ({dot_count / process_count * 100:.1f}%)")

    # ─── RPS 算出 ────────────────────────────────────────────
    dot_rec = [(r[1], r[5]) for r in records if r[5] is not None]
    if len(dot_rec) < 10:
        print("\n  警告: 検出点が少なすぎて rps を算出できません。")
        print("  --dark-thresh を上げる (例: --dark-thresh 100) か、")
        print("  --min-dot-area を下げる (例: --min-dot-area 4) を試してください。")
        return

    times_arr = np.array([r[0] for r in dot_rec])
    cum_arr = np.array([r[1] for r in dot_rec])

    coeffs = np.polyfit(times_arr, cum_arr, 1)
    slope_dps = coeffs[0]        # deg / sec
    rps = slope_dps / 360.0
    rpm = abs(rps) * 60.0
    frames_per_rev = args.fps / abs(rps) if rps != 0 else float("inf")

    print(f"\n  ─── 回転数フィット結果 ───")
    print(f"    傾き            : {slope_dps:+.2f} deg/s")
    print(f"    rps             : {rps:+.3f} rps")
    print(f"    rpm             : {rpm:.1f} rpm")
    print(f"    1周のフレーム数 : {frames_per_rev:.1f} frames/rev")
    direction = "正転 (画像座標で時計回り)" if rps > 0 else "逆転 (画像座標で反時計回り)"
    print(f"    回転方向        : {direction}")

    if abs(rps) > nyquist_rps:
        print(f"\n  ⚠ 警告: rps ({abs(rps):.1f}) がナイキスト限界 ({nyquist_rps:.1f} rps)"
              f" を超えています。折り返し(エイリアシング)の可能性があります。")

    # ─── プロット ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上パネル: 累積角度 + フィット直線
    ax = axes[0]
    ax.scatter(times_arr, cum_arr, s=6, color="steelblue", alpha=0.7, zorder=3,
               label=f"cumulative angle  (n={len(dot_rec)})")
    t_fit = np.linspace(times_arr[0], times_arr[-1], 300)
    ax.plot(t_fit, np.polyval(coeffs, t_fit), "r--", linewidth=1.5,
            label=f"linear fit: {rps:+.3f} rps ({rpm:.1f} rpm)\n"
                  f"1 rev = {frames_per_rev:.1f} frames")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cumulative angle [deg]")
    ax.set_title(f"Ball Spin — {video_path.stem}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 下パネル: 生の角度 (折り返し前)
    raw_rec = [(r[1], r[4]) for r in records if r[4] is not None]
    if raw_rec:
        rt = [r[0] for r in raw_rec]
        ra = [r[1] for r in raw_rec]
        axes[1].scatter(rt, ra, s=6, color="tomato", alpha=0.7)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Raw angle [deg]")
    axes[1].set_title("Raw angle per frame (atan2, -180 to +180)")
    axes[1].set_ylim(-195, 195)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(plot_path), dpi=150)
    plt.close()

    print(f"\n  Plot : {plot_path}")
    print(f"  CSV  : {csv_path}")
    print(f"  フレーム画像: {frame_dir}")


if __name__ == "__main__":
    main()
