"""6/1セッション全データ集計: v0, a, mu, n, Hough検出率, 判定"""
import os
import glob
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

BASE = r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detections"
G = 9.81  # m/s^2

def model(t, v0, a):
    return np.maximum(v0 - a * t, 0)

def analyze(vel_csv, det_csv):
    vdf = pd.read_csv(vel_csv)
    # 速度が有効な行（speed_km_h が非NaN かつ > 0）
    vdf = vdf.dropna(subset=["speed_km_h"])
    vdf = vdf[vdf["speed_km_h"] > 0].reset_index(drop=True)

    if len(vdf) < 5:
        return None

    t = vdf["time_sec"].values - vdf["time_sec"].values[0]
    v = vdf["speed_km_h"].values

    try:
        popt, _ = curve_fit(model, t, v, p0=[v[0], 1.0],
                            bounds=([0, 0], [np.inf, np.inf]))
        v0, a_kph_s = popt
        mu = (a_kph_s / 3.6) / G
        n = len(vdf)
    except Exception:
        return None

    # Hough検出率
    try:
        ddf = pd.read_csv(det_csv)
        total = len(ddf)
        detected = ddf["hough_x"].notna().sum()
        rate = detected / total * 100 if total > 0 else 0.0
    except Exception:
        rate = float("nan")

    return v0, a_kph_s, mu, n, rate

# 全velocity CSVを取得
vel_files = sorted(glob.glob(os.path.join(BASE, "velocity_20260601_*.csv")))

rows = []
for vf in vel_files:
    stem = os.path.basename(vf).replace("velocity_", "detections_")
    df_path = os.path.join(BASE, stem)
    res = analyze(vf, df_path)
    tag = os.path.basename(vf).replace("velocity_20260601_", "").replace(".csv", "")
    if res is None:
        rows.append({"tag": tag, "v0": None, "a": None, "mu": None, "n": 0, "hough_pct": None, "judge": "データ不足"})
    else:
        v0, a, mu, n, rate = res
        if mu < 0.03:
            judge = "× (低すぎ)"
        elif mu > 0.25:
            judge = "× (高すぎ)"
        elif n < 30:
            judge = "△ (点数少)"
        elif rate < 20:
            judge = "△ (検出率低)"
        else:
            judge = "○"
        rows.append({"tag": tag, "v0": v0, "a": a, "mu": mu, "n": n, "hough_pct": rate, "judge": judge})

df = pd.DataFrame(rows)

print("="*75)
print(f"{'時刻':>10}  {'v0(km/h)':>9}  {'a(km/h/s)':>10}  {'mu':>7}  {'n':>5}  {'Hough%':>7}  判定")
print("-"*75)
for _, r in df.iterrows():
    if r["v0"] is None:
        print(f"{r['tag']:>10}  {'--':>9}  {'--':>10}  {'--':>7}  {r['n']:>5}  {'--':>7}  {r['judge']}")
    else:
        print(f"{r['tag']:>10}  {r['v0']:>9.2f}  {r['a']:>10.3f}  {r['mu']:>7.4f}  {r['n']:>5}  {r['hough_pct']:>6.1f}%  {r['judge']}")
print("="*75)

ok = df[df["judge"] == "○"]
if len(ok) > 0:
    print(f"\n採用試行: {len(ok)} 本")
    print(f"  mu 平均: {ok['mu'].mean():.4f}")
    print(f"  mu 標準偏差: {ok['mu'].std():.4f}")
    print(f"  v0 範囲: {ok['v0'].min():.2f} ~ {ok['v0'].max():.2f} km/h")
else:
    print("\n○判定の試行なし")
