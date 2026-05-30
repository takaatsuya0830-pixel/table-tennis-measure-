"""
定規の目盛りが明瞭に出ているy行を自動選択して px_per_cm を精密計測する。
"""
import cv2
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\calib_frame.png"
)

img = cv2.imdecode(
    np.fromfile(str(IMG_PATH), dtype=np.uint8), cv2.IMREAD_COLOR
)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 定規付近のy範囲を総当たりして、最も周期的な目盛りが出ている行を探す
X0, X1 = 700, 1350   # 目盛りが存在しそうな範囲に絞る
best_y, best_px_per_cm, best_n_peaks = 0, 0.0, 0

results = []
for y in range(700, 800):
    row = gray[y, X0:X1].astype(float)
    inv = 255.0 - row
    peaks, _ = find_peaks(inv, height=40, distance=5, prominence=20)
    if len(peaks) < 5:
        continue
    intervals = np.diff(peaks)
    # 等間隔性チェック（標準偏差が小さいほど等間隔）
    if np.mean(intervals) < 3:
        continue
    cv = np.std(intervals) / np.mean(intervals)  # 変動係数
    results.append((y, len(peaks), float(np.median(intervals)), cv))

# 最も峰が多くて等間隔なy行を選ぶ
results.sort(key=lambda r: (-r[1], r[3]))
if not results:
    print("目盛り検出に失敗しました。threshold を下げて再試行してください。")
    exit(1)

best = results[0]
best_y, n_peaks, median_interval, cv = best
print(f"最適な行: y={best_y}")
print(f"  検出峰数    : {n_peaks}")
print(f"  峰間隔中央値: {median_interval:.2f} px")
print(f"  変動係数    : {cv:.3f} (小さいほど等間隔)")

# 間隔の解釈（定規の1mm/2mm/5mm/10mmを判定）
# 1920px幅, 30cm定規 → 30px/cm が目安
# 検出間隔が 2-5px → 1mm目盛り
# 検出間隔が 10-20px → 5mm目盛り
# 検出間隔が 25-40px → 10mm(1cm)目盛り
for mm_per_tick in [1, 2, 5, 10]:
    px_per_mm = median_interval / mm_per_tick
    px_per_cm = px_per_mm * 10
    if 15 < px_per_cm < 70:  # 現実的な範囲
        print(f"\n  仮定: {mm_per_tick}mm目盛り → px_per_cm = {px_per_cm:.2f}")

# 実際のプロファイルと検出結果を保存
row = gray[best_y, X0:X1].astype(float)
inv = 255.0 - row
peaks, _ = find_peaks(inv, height=40, distance=5, prominence=20)

# 定規の範囲を定量化（最初と最後のピーク）
if len(peaks) >= 2:
    p_first = int(peaks[0]) + X0
    p_last = int(peaks[-1]) + X0
    span_px = p_last - p_first
    n_intervals = len(peaks) - 1
    print(f"\n  最初のピーク: x={p_first}  最後のピーク: x={p_last}")
    print(f"  スパン: {span_px}px / {n_intervals} 間隔")

OUT = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\ruler_profile2.png")
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(np.arange(len(row)) + X0, row, linewidth=0.5, label="brightness")
for p in peaks:
    ax.axvline(p + X0, color="red", linewidth=0.5, alpha=0.7)
ax.set_xlabel("x [px]")
ax.set_ylabel("brightness")
ax.set_title(f"Ruler profile y={best_y}  peaks={n_peaks}  interval={median_interval:.1f}px")
ax.grid(True, alpha=0.3)
plt.tight_layout()
buf, _ = None, plt.savefig(str(OUT), dpi=120)
plt.close()
print(f"\nプロファイル保存: {OUT}")
