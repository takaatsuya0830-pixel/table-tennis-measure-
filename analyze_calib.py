"""キャリブレーション画像から定規の範囲を自動推定する"""
import cv2
import numpy as np
from pathlib import Path

IMG_PATH = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\calib_frame.png")

img = cv2.imdecode(
    np.fromfile(str(IMG_PATH), dtype=np.uint8), cv2.IMREAD_COLOR
)
h, w = img.shape[:2]
print(f"画像サイズ: {w}x{h}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定規が映っている行を探す（明るいピクセルが横に並んでいる行）
best_y, best_left, best_right, best_count = 0, 0, 0, 0
for y in range(400, 750):
    row = gray[y, :]
    bright_px = np.where(row > 180)[0]
    if len(bright_px) > best_count:
        best_count = len(bright_px)
        best_y = y
        best_left = int(bright_px[0])
        best_right = int(bright_px[-1])

print(f"\n定規が最も明瞭な行: y={best_y}")
print(f"  左端: x={best_left}  右端: x={best_right}")
print(f"  幅: {best_right - best_left} px")

# 定規の周辺列でサンプリング（目盛り位置の推定）
row = gray[best_y, best_left:best_right]
# 目盛り部分は局所的に暗い（印刷部分）→ エッジ検出で等間隔を探す
edges = np.diff(row.astype(int))
peaks = np.where(np.abs(edges) > 30)[0]
if len(peaks) > 2:
    intervals = np.diff(peaks)
    # 最頻値のインターバル（目盛り間隔 = 1mm単位の繰り返し）
    from collections import Counter
    ic = Counter(intervals.tolist())
    common_interval = ic.most_common(1)[0][0]
    print(f"  目盛り間隔: {common_interval} px（推定）")
    # 5mm目盛り = common_interval * 5、10mm = common_interval * 10
    print(f"  → 1cm = {common_interval * 10} px (推定)")
    print(f"  → px_per_cm ≒ {common_interval * 10:.1f}")

# 定規が画像内で何cmか（30cmとして計算）
span_px = best_right - best_left
px_per_cm_30 = span_px / 30.0
print(f"\n定規全体={span_px}px を 30cm と仮定した場合:")
print(f"  px_per_cm = {px_per_cm_30:.2f}")

print("\n※ この推定値はあくまで目安です。")
print("  calibrate.py を実行してクリックで正確に測定することを推奨します。")
print(
    f"  手動入力の場合: python calibrate.py --manual"
    f" --x0 {best_left} --x1 {best_right} --dist-cm 30"
)
