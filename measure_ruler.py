"""
定規の目盛り間隔をピクセルで測定して px_per_cm を精密に算出する。
1cmの目盛り（長い線）を等間隔検出で求める。
"""
import cv2
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

IMG_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\calib_frame.png"
)
OUT_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\ruler_profile.png"
)

Y_RULER = 749   # analyze_calib.py で特定した定規の行

img = cv2.imdecode(
    np.fromfile(str(IMG_PATH), dtype=np.uint8), cv2.IMREAD_COLOR
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定規の行の水平輝度プロファイル（定規内の範囲だけ抽出）
X0, X1 = 300, 1350  # 定規が映っている範囲
profile = gray[Y_RULER, X0:X1].astype(float)

# 目盛りは暗い（黒い線）→ 輝度の谷を検出
inverted = 255.0 - profile

# 1cmの目盛りは他より長い（=谷が深い）ので threshold を高めに設定
peaks, props = find_peaks(
    inverted,
    height=60,       # 谷の深さ閾値（輝度の低さ）
    distance=25,     # 隣接ピーク最小距離（1mmの目盛り間隔より小さく）
    prominence=30,   # 際立ち度
)

print(f"検出した目盛り数: {len(peaks)}")
if len(peaks) > 2:
    intervals = np.diff(peaks)
    median_interval = float(np.median(intervals))
    print(f"目盛り間隔（中央値）: {median_interval:.2f} px")
    # Pixel 9aで30cm定規を撮影 → 1mm刻みなら約3px間隔が多い
    # 1cm目盛り（10mm）なら約30px間隔
    # 検出間隔が3px前後 → 1mm間隔として解釈
    # 検出間隔が30px前後 → 1cm間隔として解釈
    if median_interval < 10:
        px_per_cm = median_interval * 10
        unit = "1mm目盛り"
    else:
        px_per_cm = median_interval
        unit = "1cm目盛り"
    print(f"目盛り単位: {unit}")
    print(f"\n→ px_per_cm = {px_per_cm:.3f}")

# プロファイルグラフを保存
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(np.arange(len(profile)) + X0, profile, linewidth=0.5)
for p in peaks:
    ax.axvline(p + X0, color="red", linewidth=0.5, alpha=0.5)
ax.set_xlabel("x [px]")
ax.set_ylabel("輝度")
ax.set_title(f"定規プロファイル (y={Y_RULER})")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=100)
plt.close()
print(f"\nプロファイルグラフ: {OUT_PATH}")
