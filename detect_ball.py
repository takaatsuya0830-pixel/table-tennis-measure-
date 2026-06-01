"""
卓球ボール検出スクリプト（白ボール対応）

出力:
  frames/detections/detections_YYYYMMDD_HHMMSS.csv  ← 全フレーム1行ずつ
  frames/detections/frame_NNNNN.png                 ← 検出結果画像

CSV列:
  frame_number, time_sec,
  hough_x, hough_y, hough_r,   ← 未検出は空欄
  color_x, color_y, color_r,   ← 未検出は空欄
  num_hough, num_color

複数検出時は「前フレームで採用したボール位置に最も近い円」を採用（前フレーム追従方式）。
最初の検出フレームまたは直前が見失い状態のときは最大円で初期化。
両手法の結果を別列に保存するので、後で比較可能。

使い方:
  python detect_ball.py                          # 全フレーム処理
  python detect_ball.py --start 100 --count 300
  python detect_ball.py --ball orange            # オレンジボール
  python detect_ball.py --fps 247                # 実効fps上書き
  python detect_ball.py --save-every 30          # 画像保存間隔（デフォルト10）
  python detect_ball.py --max-jump 200           # 追従最大距離px（デフォルト150）
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def imwrite(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite の日本語パス対応版。"""
    ext = path.suffix  # e.g. ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(str(path))


# ── 設定定数 ──────────────────────────────────────────────
VIDEO_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\PXL_20260514_154725948_h264.mp4"
)
OUTPUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames")

EFFECTIVE_FPS = 233  # 実効fps（校正値が変わったらここを変更）

# Hough円検出パラメータ
# ボールが画面内で直径30〜100px程度になる距離を想定
# 実際のサイズを sample_frame001.png で確認してから調整する
HOUGH_MIN_R = 10   # px（ボール最小半径の目安）
HOUGH_MAX_R = 80   # px（ボール最大半径の目安）
HOUGH_PARAM2 = 30  # 検出閾値（下げると感度UP、誤検出も増える）

# 白ボール用HSV範囲（H: 全色相, S: 低彩度, V: 高輝度）
WHITE_HSV_LOW = np.array([0, 0, 180])
WHITE_HSV_HIGH = np.array([180, 60, 255])

# オレンジボール用HSV範囲
ORANGE_HSV_LOW = np.array([5, 120, 120])
ORANGE_HSV_HIGH = np.array([25, 255, 255])
# ─────────────────────────────────────────────────────────


def _in_roi(
    x: int, y: int, roi: tuple[int, int, int, int] | None
) -> bool:
    """ROI (x0,y0,x1,y1) 内に点があるか。roi=None なら常に True。"""
    if roi is None:
        return True
    x0, y0, x1, y1 = roi
    return x0 <= x <= x1 and y0 <= y <= y1


def _filter_roi(
    circles: list[tuple[int, int, int]],
    roi: tuple[int, int, int, int] | None,
    min_r: int,
    max_r: int,
) -> list[tuple[int, int, int]]:
    """ROI範囲外・サイズ範囲外の検出を除去する。"""
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


def detect_by_hough(
    gray: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    min_r: int = HOUGH_MIN_R,
    max_r: int = HOUGH_MAX_R,
    param2: int = HOUGH_PARAM2,
) -> list[tuple[int, int, int]]:
    """Hough変換で円を検出する。(x, y, r) のリストを返す。
    フル解像度で検出してからROI・サイズでフィルタする。"""
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=100,
        param2=param2,
        minRadius=HOUGH_MIN_R,
        maxRadius=HOUGH_MAX_R,
    )
    if circles is None:
        return []
    raw = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    return _filter_roi(raw, roi, min_r, max_r)


def detect_by_color(
    hsv: np.ndarray,
    ball_color: str,
    roi: tuple[int, int, int, int] | None = None,
    min_r: int = HOUGH_MIN_R,
    max_r: int = HOUGH_MAX_R,
) -> list[tuple[int, int, int]]:
    """色マスクで円形領域を検出する。(x, y, r) のリストを返す。"""
    if ball_color == "white":
        mask = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)
    else:
        mask = cv2.inRange(hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    raw = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if r < 1:
            continue
        circularity = area / (np.pi * r * r)
        if circularity > 0.5:
            raw.append((int(cx), int(cy), int(r)))
    return _filter_roi(raw, roi, min_r, max_r)


def largest(
    circles: list[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    """複数検出から半径最大の円を返す。リストが空なら None。"""
    if not circles:
        return None
    return max(circles, key=lambda c: c[2])


def nearest(
    circles: list[tuple[int, int, int]],
    prev_x: int,
    prev_y: int,
    max_jump: float,
) -> tuple[int, int, int] | None:
    """前フレーム位置(prev_x, prev_y)に最も近い円を返す。
    全候補が max_jump を超える場合は None（見失い）を返す。"""
    if not circles:
        return None
    best = min(circles, key=lambda c: (c[0] - prev_x) ** 2 + (c[1] - prev_y) ** 2)
    dist = ((best[0] - prev_x) ** 2 + (best[1] - prev_y) ** 2) ** 0.5
    return best if dist <= max_jump else None


def pick(
    circles: list[tuple[int, int, int]],
    prev: tuple[int, int, int] | None,
    max_jump: float,
) -> tuple[int, int, int] | None:
    """追従方式で採用円を選ぶ。prev=None なら最大円で初期化。"""
    if prev is None:
        return largest(circles)
    return nearest(circles, prev[0], prev[1], max_jump)


def draw_detections(
    frame: np.ndarray,
    hough: list,
    color: list,
    frame_idx: int,
    time_sec: float,
    roi: tuple | None = None,
) -> np.ndarray:
    vis = frame.copy()
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(vis, (x0, y0), (x1, y1), (200, 200, 0), 1)
    for x, y, r in hough:
        cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
    for x, y, r in color:
        cv2.circle(vis, (x, y), r, (0, 100, 255), 2)
        cv2.circle(vis, (x, y), 3, (0, 100, 255), -1)
    label = (
        f"frame {frame_idx}  t={time_sec:.4f}s"
        f"  H:{len(hough)} C:{len(color)}"
    )
    cv2.putText(
        vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        vis, "green=Hough  orange=color", (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--count", type=int, default=None, help="処理フレーム数（省略時: 全フレーム）"
    )
    parser.add_argument("--ball", choices=["white", "orange"], default="white")
    parser.add_argument(
        "--fps", type=float, default=float(EFFECTIVE_FPS),
        help=f"実効fps（デフォルト: {EFFECTIVE_FPS}）",
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="N フレームごとに画像保存（検出ありは必ず保存）",
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="検出ROI（例: --roi 200 550 1900 900）。手や左端偽陽性を除外できる",
    )
    parser.add_argument(
        "--min-r", type=int, default=HOUGH_MIN_R,
        help=f"ボール最小半径px（デフォルト: {HOUGH_MIN_R}）",
    )
    parser.add_argument(
        "--max-r", type=int, default=HOUGH_MAX_R,
        help=f"ボール最大半径px（デフォルト: {HOUGH_MAX_R}）",
    )
    parser.add_argument(
        "--param2", type=int, default=HOUGH_PARAM2,
        help=f"HoughCircles param2（小さいほど感度UP、デフォルト: {HOUGH_PARAM2}）",
    )
    parser.add_argument(
        "--max-jump", type=float, default=150.0,
        help="前フレームからの最大追従距離px。超えると見失い扱い（デフォルト: 150）",
    )
    args = parser.parse_args()

    roi = tuple(args.roi) if args.roi else None

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"エラー: {video_path} が見つかりません。")
        print("先に convert_hevc.py を実行してH.264変換を完了してください。")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("エラー: 動画を開けませんでした。")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    if args.count is not None:
        process_count = args.count
    else:
        process_count = total_frames - args.start

    detect_dir = OUTPUT_DIR / "detections"
    detect_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = detect_dir / f"detections_{timestamp}.csv"

    print(f"動画       : {video_path.name}")
    print(f"ボール色   : {args.ball}")
    print(f"実効fps    : {args.fps}")
    if roi:
        print(f"ROI        : x={roi[0]}-{roi[2]}  y={roi[1]}-{roi[3]}")
    print(f"半径範囲   : {args.min_r} 〜 {args.max_r} px")
    print(f"最大追従距離: {args.max_jump} px")
    print(f"処理範囲   : フレーム {args.start} 〜 {args.start + process_count - 1}")
    print(f"CSV出力    : {csv_path.name}")
    print()

    hough_hits = 0
    color_hits = 0
    prev_hough: tuple[int, int, int] | None = None
    prev_color: tuple[int, int, int] | None = None

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_number", "time_sec",
            "hough_x", "hough_y", "hough_r",
            "color_x", "color_y", "color_r",
            "num_hough", "num_color",
        ])

        for i in range(process_count):
            ret, frame = cap.read()
            if not ret:
                print(f"\nフレーム {args.start + i} でデコード終了")
                break

            frame_idx = args.start + i
            time_sec = frame_idx / args.fps

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            hough_all = detect_by_hough(
                gray, roi=roi, min_r=args.min_r, max_r=args.max_r,
                param2=args.param2,
            )
            color_all = detect_by_color(
                hsv, args.ball, roi=roi,
                min_r=args.min_r, max_r=args.max_r,
            )

            # 前フレーム追従方式で採用円を選択
            hb = pick(hough_all, prev_hough, args.max_jump)
            cb = pick(color_all, prev_color, args.max_jump)
            prev_hough = hb
            prev_color = cb

            writer.writerow([
                frame_idx,
                f"{time_sec:.6f}",
                hb[0] if hb else "", hb[1] if hb else "", hb[2] if hb else "",
                cb[0] if cb else "", cb[1] if cb else "", cb[2] if cb else "",
                len(hough_all),
                len(color_all),
            ])

            if hough_all:
                hough_hits += 1
            if color_all:
                color_hits += 1

            save = (
                (i % args.save_every == 0)
                or bool(hough_all)
                or bool(color_all)
            )
            if save:
                vis = draw_detections(
                    frame, hough_all, color_all, frame_idx, time_sec, roi
                )
                imwrite(detect_dir / f"frame_{frame_idx:05d}.png", vis)

            if i % 50 == 0:
                pct = i / process_count * 100
                print(
                    f"  [{pct:5.1f}%] frame {frame_idx}"
                    f"  Hough:{hough_hits}  Color:{color_hits}",
                    end="\r",
                )

            if args.preview:
                vis = draw_detections(
                    frame, hough_all, color_all, frame_idx, time_sec, roi
                )
                cv2.imshow("Ball Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if args.preview:
        cv2.destroyAllWindows()

    h_pct = hough_hits / process_count * 100
    c_pct = color_hits / process_count * 100
    print(f"\n\n処理完了: {process_count} フレーム")
    print(f"  Hough検出あり : {hough_hits} フレーム  ({h_pct:.1f}%)")
    print(f"  色検出あり    : {color_hits} フレーム  ({c_pct:.1f}%)")
    print(f"  CSV           : {csv_path}")
    print()

    if hough_hits == 0 and color_hits == 0:
        print("=== 検出ゼロ: チューニングのヒント ===")
        print("  手順1: detect_dir/frame_00000.png を開いてボールの直径(px)を測る")
        print("         -> HOUGH_MIN_R と HOUGH_MAX_R をボール半径に合わせる")
        print("  手順2: HOUGH_PARAM2 を 30 -> 20 に下げて感度UP")
        print("  手順3: 白が暗い場合は WHITE_HSV_LOW の V を 180 -> 150 に下げる")
    else:
        print("次のステップ:")
        print(
            f"  python calc_velocity.py"
            f" --csv {csv_path.name} --px-per-cm <値>"
        )


if __name__ == "__main__":
    main()
