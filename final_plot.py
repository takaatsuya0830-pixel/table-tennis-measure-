"""
転がり区間を切り出して卒論用の最終速度グラフを作成する。
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

VELOCITY_CSV = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detections"
    r"\velocity_20260522_103205.csv"
)
OUT_PNG = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\final_velocity_plot.png"
)

T_START = 0.75
T_END = 1.77
MAX_SPD = 10.0

rows = []
with open(VELOCITY_CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if not r["speed_km_h"]:
            continue
        t = float(r["time_sec"])
        spd = float(r["speed_km_h"])
        is_interp = r["interpolated"] == "1"
        if T_START <= t <= T_END and spd <= MAX_SPD:
            rows.append((t, spd, is_interp))

times = np.array([r[0] for r in rows])
speeds = np.array([r[1] for r in rows])
interp = np.array([r[2] for r in rows])

t0 = times[0]
times_rel = times - t0


def linear_decel(t, v0, a):
    return np.maximum(v0 - a * t, 0)


try:
    popt, _ = curve_fit(linear_decel, times_rel, speeds, p0=[3.0, 1.5])
    v0_fit, a_fit = popt
    t_fit = np.linspace(0, times_rel[-1], 200)
    v_fit = linear_decel(t_fit, *popt)
    has_fit = True
    print(f"v0={v0_fit:.2f} km/h  a={a_fit:.2f} km/h/s  ({a_fit/3.6:.3f} m/s^2)")
except Exception as e:
    has_fit = False
    print(f"fit failed: {e}")

fig, ax = plt.subplots(figsize=(9, 5))

real = ~interp
ax.scatter(
    times_rel[real], speeds[real],
    s=15, color="steelblue", zorder=3, label="Detected",
)
if interp.any():
    ax.scatter(
        times_rel[interp], speeds[interp],
        s=10, color="cyan", zorder=2, label="Interpolated",
    )

if has_fit:
    ax.plot(
        t_fit, v_fit,
        color="tomato", linewidth=2, linestyle="--",
        label=(
            f"Linear fit: $v_0$={v0_fit:.2f} km/h,"
            f" $a$={a_fit:.2f} km/h/s"
        ),
    )

ax.set_xlabel("Time after release [s]", fontsize=12)
ax.set_ylabel("Speed [km/h]", fontsize=12)
ax.set_title("Table Tennis Ball Rolling Speed", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

vmax = speeds.max()
vavg = speeds.mean()
ax.text(
    0.98, 0.95,
    f"Max: {vmax:.2f} km/h\nAvg: {vavg:.2f} km/h\nn={len(speeds)} frames",
    transform=ax.transAxes,
    ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round", fc="white", alpha=0.7),
)

plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=150)
plt.close()
print(f"saved: {OUT_PNG}")
