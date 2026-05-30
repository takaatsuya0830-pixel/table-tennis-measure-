"""検出済みCSVから速度が最も大きいフレームを探す"""
import csv
from pathlib import Path

p = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detections"
    r"\detections_20260522_000152.csv"
)
rows = []
with open(p, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["hough_x"]:
            rows.append((
                int(r["frame_number"]),
                float(r["hough_x"]),
                float(r["hough_y"]),
            ))

# 5フレームウィンドウで速度を計算
speeds = []
for i in range(2, len(rows) - 2):
    fn0, x0, y0 = rows[i - 2]
    fn1, x1, y1 = rows[i + 2]
    dt = (fn1 - fn0) / 233.0
    if dt > 0:
        dx = x1 - x0
        dy = y1 - y0
        spd = (dx**2 + dy**2) ** 0.5 / dt
        speeds.append((rows[i][0], spd, dx / dt, dy / dt,
                       rows[i][1], rows[i][2]))

speeds.sort(key=lambda s: -s[1])
print("速度上位30フレーム:")
print(f"{'frame':>6}  {'spd px/s':>10}  {'vx':>8}  {'vy':>8}"
      f"  {'x':>6}  {'y':>6}")
for fn, sp, vx, vy, x, y in speeds[:30]:
    print(f"{fn:6d}  {sp:10.1f}  {vx:8.1f}  {vy:8.1f}"
          f"  {x:6.0f}  {y:6.0f}")
