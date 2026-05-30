"""
px/cm キャリブレーションスクリプト

使い方:
  python calibrate.py

操作:
  1. 画像ウィンドウが開いたら、定規の「0cm」の位置を左クリック
  2. 次に「30cm」の位置を左クリック
  3. px_per_cm が表示される

マウスが使えない場合は --manual モードで直接座標を入力できる:
  python calibrate.py --manual --x0 120 --x1 1050 --dist-cm 30
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

CALIB_IMAGE = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\calib_frame.png"
)

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
        clicks.append((x, y))
        cv2.circle(param, (x, y), 8, (0, 0, 255), -1)
        label = "0cm" if len(clicks) == 1 else "30cm"
        cv2.putText(
            param, f"{label} ({x},{y})",
            (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2,
        )
        cv2.imshow("Calibration", param)
        if len(clicks) == 2:
            dx = clicks[1][0] - clicks[0][0]
            dy = clicks[1][1] - clicks[0][1]
            dist_px = math.sqrt(dx**2 + dy**2)
            print(f"\n点1: {clicks[0]}")
            print(f"点2: {clicks[1]}")
            print(f"ピクセル距離: {dist_px:.1f} px")


def calc_result(p1, p2, dist_cm: float) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist_px = math.sqrt(dx**2 + dy**2)
    px_per_cm = dist_px / dist_cm
    return px_per_cm, dist_px


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", action="store_true",
                        help="座標を手動入力するモード")
    parser.add_argument("--x0", type=int, help="点1のX座標")
    parser.add_argument("--y0", type=int, default=0, help="点1のY座標（横定規なら省略可）")
    parser.add_argument("--x1", type=int, help="点2のX座標")
    parser.add_argument("--y1", type=int, default=0, help="点2のY座標（横定規なら省略可）")
    parser.add_argument("--dist-cm", type=float, default=30.0,
                        help="2点間の実距離(cm)。デフォルト30cm")
    args = parser.parse_args()

    if not CALIB_IMAGE.exists():
        print(f"エラー: {CALIB_IMAGE} が見つかりません。")
        sys.exit(1)

    if args.manual:
        if args.x0 is None or args.x1 is None:
            print("--manual モードでは --x0 と --x1 が必要です。")
            sys.exit(1)
        p1 = (args.x0, args.y0)
        p2 = (args.x1, args.y1)
        px_per_cm, dist_px = calc_result(p1, p2, args.dist_cm)
        print(f"点1: {p1}")
        print(f"点2: {p2}")
        print(f"ピクセル距離: {dist_px:.1f} px")
        print(f"実距離: {args.dist_cm} cm")
        print(f"\n→ px_per_cm = {px_per_cm:.3f}")
        print(f"\n速度計算コマンド:")
        print(
            f"  python calc_velocity.py"
            f" --csv <ファイル名>.csv --px-per-cm {px_per_cm:.3f}"
        )
        return

    # クリックモード
    img = cv2.imdecode(
        np.fromfile(str(CALIB_IMAGE), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if img is None:
        print("画像を読み込めませんでした。")
        sys.exit(1)

    # 表示用にリサイズ（1920x1080は大きすぎる場合がある）
    h, w = img.shape[:2]
    scale = min(1.0, 1280 / w)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)))

    print("操作手順:")
    print(f"  1. 定規の「0cm」の位置をクリック")
    print(f"  2. 定規の「{args.dist_cm}cm」の位置をクリック")
    print(f"  q キーで終了")
    print()

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse, disp)
    cv2.imshow("Calibration", disp)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or key == 27:
            break
        if len(clicks) == 2:
            # スケール補正（表示縮小分を元の座標系に戻す）
            p1 = (int(clicks[0][0] / scale), int(clicks[0][1] / scale))
            p2 = (int(clicks[1][0] / scale), int(clicks[1][1] / scale))
            px_per_cm, dist_px = calc_result(p1, p2, args.dist_cm)
            print(f"実画像での点1: {p1}")
            print(f"実画像での点2: {p2}")
            print(f"実距離: {args.dist_cm} cm  →  {dist_px:.1f} px")
            print(f"\n→ px_per_cm = {px_per_cm:.3f}")
            print(f"\n速度計算コマンド:")
            print(
                f"  python calc_velocity.py"
                f" --csv <ファイル名>.csv --px-per-cm {px_per_cm:.3f}"
            )
            print("\n（q を押して終了）")
            # 結果を画像に描画
            cv2.line(disp, clicks[0], clicks[1], (0, 255, 0), 2)
            cv2.putText(
                disp,
                f"{dist_px/scale:.0f}px = {args.dist_cm}cm"
                f"  -> {px_per_cm:.2f}px/cm",
                (20, disp.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            cv2.imshow("Calibration", disp)
            clicks.clear()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
