"""
俯瞰アングル動画の複数トライアルを比較する最終プロット。
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

DETECT_DIR = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detections"
)
OUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames")

# 各トライアルの設定: (CSVファイル, T_START, T_END, MAX_SPD_KMH, ラベル)
TRIALS = [
    ("velocity_20260522_113229.csv",  2.75, 3.50,  8.0, "Trial 1"),
    ("velocity_20260522_114319.csv",  1.75, 3.00,  6.0, "Trial 2"),
    ("velocity_20260522_114646.csv",  1.55, 2.38,  7.0, "Trial 3"),
    ("velocity_20260522_114502.csv",  2.50, 3.15, 10.0, "Trial 4"),
    ("velocity_20260522_114802.csv",  2.40, 2.85, 10.0, "Trial 5"),
]

COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "purple"]
G = 9.81  # m/s^2


def linear_decel(t, v0, a):
    return np.maximum(v0 - a * t, 0)


def load_trial(csv_name, t_start, t_end, max_spd):
    rows = []
    with open(DETECT_DIR / csv_name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["speed_km_h"]:
                continue
            t = float(r["time_sec"])
            spd = float(r["speed_km_h"])
            is_interp = r["interpolated"] == "1"
            if t_start <= t <= t_end and spd <= max_spd:
                rows.append((t, spd, is_interp))
    if not rows:
        return None
    times = np.array([r[0] for r in rows])
    speeds = np.array([r[1] for r in rows])
    interp = np.array([r[2] for r in rows])
    times_rel = times - times[0]
    return times_rel, speeds, interp


# ── Figure 1: 各トライアルの個別プロット ──────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes_flat = axes.flatten()

results = []

for idx, (csv_name, t_start, t_end, max_spd, label) in enumerate(TRIALS):
    data = load_trial(csv_name, t_start, t_end, max_spd)
    if data is None:
        print(f"{label}: no data")
        continue

    times_rel, speeds, interp = data
    ax = axes_flat[idx]
    real = ~interp

    ax.scatter(times_rel[real], speeds[real],
               s=15, color=COLORS[idx], zorder=3, label="Detected")
    if interp.any():
        ax.scatter(times_rel[interp], speeds[interp],
                   s=10, color="cyan", zorder=2, alpha=0.7, label="Interpolated")

    try:
        popt, _ = curve_fit(linear_decel, times_rel, speeds, p0=[5.0, 2.0])
        v0, a = popt
        mu = a / 3.6 / G
        t_fit = np.linspace(0, times_rel[-1], 200)
        v_fit = linear_decel(t_fit, *popt)
        ax.plot(t_fit, v_fit, color="black", linewidth=1.5, linestyle="--",
                label=f"Fit: v0={v0:.2f}, a={a:.2f} km/h/s")
        results.append((label, v0, a, mu, len(speeds)))
        print(f"{label}: v0={v0:.2f} km/h  a={a:.2f} km/h/s  mu={mu:.4f}  n={len(speeds)}")
    except Exception as e:
        print(f"{label}: fit failed: {e}")
        results.append((label, np.nan, np.nan, np.nan, len(speeds)))

    ax.set_title(f"{label} (n={len(speeds)})", fontsize=11)
    ax.set_xlabel("Time after release [s]", fontsize=9)
    ax.set_ylabel("Speed [km/h]", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

# 最後のパネルに統計サマリー
ax_sum = axes_flat[5]
ax_sum.axis("off")
if results:
    header = ["Trial", "v0 [km/h]", "a [km/h/s]", "μ"]
    rows_txt = [header] + [
        [r[0], f"{r[1]:.2f}", f"{r[2]:.2f}", f"{r[3]:.4f}"]
        for r in results if not np.isnan(r[1])
    ]
    mus = [r[3] for r in results if not np.isnan(r[3])]
    rows_txt.append(["Mean", "", "", f"{np.mean(mus):.4f}"])
    rows_txt.append(["±SD", "", "", f"±{np.std(mus):.4f}"])

    table = ax_sum.table(cellText=rows_txt[1:], colLabels=rows_txt[0],
                         loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax_sum.set_title("Summary", fontsize=11)

plt.suptitle("Table Tennis Ball Rolling Deceleration — Overhead View", fontsize=13)
plt.tight_layout()
out1 = OUT_DIR / "final_overhead_all_trials.png"
plt.savefig(str(out1), dpi=150)
plt.close()
print(f"\nSaved: {out1}")

# ── Figure 2: 全トライアルを1枚に重ね書き ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))

for idx, (csv_name, t_start, t_end, max_spd, label) in enumerate(TRIALS):
    data = load_trial(csv_name, t_start, t_end, max_spd)
    if data is None:
        continue
    times_rel, speeds, interp = data
    ax2.scatter(times_rel[~interp], speeds[~interp],
                s=12, color=COLORS[idx], zorder=3, alpha=0.7, label=label)
    try:
        popt, _ = curve_fit(linear_decel, times_rel, speeds, p0=[5.0, 2.0])
        t_fit = np.linspace(0, times_rel[-1], 200)
        v_fit = linear_decel(t_fit, *popt)
        ax2.plot(t_fit, v_fit, color=COLORS[idx], linewidth=1.5, linestyle="--")
    except Exception:
        pass

if results:
    mus = [r[3] for r in results if not np.isnan(r[3])]
    mu_mean = np.mean(mus)
    mu_std = np.std(mus)
    ax2.text(0.98, 0.95,
             f"Rolling friction μ\nMean: {mu_mean:.4f}\n±SD: {mu_std:.4f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=11,
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

ax2.set_xlabel("Time after release [s]", fontsize=12)
ax2.set_ylabel("Speed [km/h]", fontsize=12)
ax2.set_title("Table Tennis Ball Rolling Speed — All Trials (Overhead View)", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)
plt.tight_layout()
out2 = OUT_DIR / "final_overhead_comparison.png"
plt.savefig(str(out2), dpi=150)
plt.close()
print(f"Saved: {out2}")

# ── Figure 3: 最良1トライアルの単独プロット ─────────────────────────
# Trial 3 (020406465) が最もクリーン
best_idx = 2
csv_name, t_start, t_end, max_spd, label = TRIALS[best_idx]
data = load_trial(csv_name, t_start, t_end, max_spd)
if data:
    times_rel, speeds, interp = data
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    real = ~interp
    ax3.scatter(times_rel[real], speeds[real],
                s=15, color="steelblue", zorder=3, label="Detected")
    if interp.any():
        ax3.scatter(times_rel[interp], speeds[interp],
                    s=10, color="cyan", zorder=2, label="Interpolated")
    try:
        popt, _ = curve_fit(linear_decel, times_rel, speeds, p0=[5.0, 2.0])
        v0, a = popt
        mu = a / 3.6 / G
        t_fit = np.linspace(0, times_rel[-1], 200)
        v_fit = linear_decel(t_fit, *popt)
        ax3.plot(t_fit, v_fit, color="tomato", linewidth=2, linestyle="--",
                 label=f"Linear fit: $v_0$={v0:.2f} km/h, $a$={a:.2f} km/h/s")
        ax3.text(0.98, 0.95,
                 f"Max: {speeds.max():.2f} km/h\nAvg: {speeds.mean():.2f} km/h\n"
                 f"n={len(speeds)} frames\n$\\mu_r$={mu:.4f}",
                 transform=ax3.transAxes, ha="right", va="top", fontsize=10,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    except Exception as e:
        print(f"best trial fit failed: {e}")

    ax3.set_xlabel("Time after release [s]", fontsize=12)
    ax3.set_ylabel("Speed [km/h]", fontsize=12)
    ax3.set_title("Table Tennis Ball Rolling Speed (Overhead View)", fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    plt.tight_layout()
    out3 = OUT_DIR / "final_overhead_best.png"
    plt.savefig(str(out3), dpi=150)
    plt.close()
    print(f"Saved: {out3}")
