"""フィット時間窓内のフレーム内訳を詳細集計する"""
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

BASE = r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detections"
G = 9.81

def model(t, v0, a):
    return np.maximum(v0 - a * t, 0)

TARGETS = [
    ("191559", "091944934 (俯瞰)"),
    ("193351", "092040464 (俯瞰)"),
    ("194625", "092052984 (俯瞰)"),
    ("173305", "173305 (横)"),
    ("175319", "175319 (横)"),
    ("190049", "091918182 (俯瞰)"),
    ("195521", "092102746 (俯瞰)"),
]

def analyze_window(tag, label):
    vf = os.path.join(BASE, f"velocity_20260601_{tag}.csv")
    df = pd.read_csv(vf)

    total_video_frames = len(df)

    # 速度が有効な行でフィット窓を決める
    has_speed = df["speed_km_h"].notna() & (df["speed_km_h"] > 0)
    if has_speed.sum() < 5:
        print(f"  {label}: データ不足 ({has_speed.sum()} pts)")
        return

    idx_first = df[has_speed].index[0]
    idx_last  = df[has_speed].index[-1]

    # フィット窓: 最初の有効速度から最後まで
    win = df.loc[idx_first:idx_last].copy()
    win_frames = len(win)
    t0 = win["time_sec"].iloc[0]
    t1 = win["time_sec"].iloc[-1]

    # フレーム分類
    detected     = (win["interpolated"] == 0) & win["x_px"].notna()
    interpolated = win["interpolated"] == 1
    missing      = (win["interpolated"] == 0) & win["x_px"].isna()

    n_det  = detected.sum()
    n_int  = interpolated.sum()
    n_mis  = missing.sum()

    # 速度が計算できたフレーム(フィットに使ったデータ点)
    n_speed = has_speed.sum()

    # μフィット
    fit_data = df[has_speed].copy()
    t = fit_data["time_sec"].values - fit_data["time_sec"].values[0]
    v = fit_data["speed_km_h"].values
    try:
        popt, _ = curve_fit(model, t, v, p0=[v[0], 1.0],
                            bounds=([0,0],[np.inf,np.inf]))
        v0, a = popt
        mu = (a / 3.6) / G
    except Exception:
        v0, a, mu = float("nan"), float("nan"), float("nan")

    # 旧スクリプトでの「Hough検出率」= 動画全体を分母にしていた
    # 本来: 窓内分母 vs 動画全体分母
    rate_whole  = n_det / total_video_frames * 100
    rate_window = n_det / win_frames * 100

    print(f"\n{'='*60}")
    print(f"  動画: {label}  (tag={tag})")
    print(f"  フィット時間窓: {t0:.2f}s 〜 {t1:.2f}s  ({t1-t0:.2f}s間)")
    print(f"  {'項目':<20} {'フレーム数':>8} {'割合':>8}")
    print(f"  {'-'*40}")
    print(f"  {'窓内総フレーム':<20} {win_frames:>8}")
    print(f"  {'  実検出 (interp=0,座標あり)':<20} {n_det:>8}  {n_det/win_frames*100:>6.1f}%")
    print(f"  {'  補間 (interp=1)':<20} {n_int:>8}  {n_int/win_frames*100:>6.1f}%")
    print(f"  {'  欠損 (NaN)':<20} {n_mis:>8}  {n_mis/win_frames*100:>6.1f}%")
    print(f"  {'  速度計算済':<20} {n_speed:>8}  {n_speed/win_frames*100:>6.1f}%  (←フィット点数)")
    print(f"  {'-'*40}")
    print(f"  v0={v0:.2f} km/h  a={a:.3f} km/h/s  mu={mu:.4f}")
    print(f"  [参考] 動画全体フレーム数: {total_video_frames}")
    print(f"  実検出率 (窓内分母): {rate_window:.1f}%   (動画全体分母): {rate_whole:.1f}%")

for tag, label in TARGETS:
    analyze_window(tag, label)

print(f"\n{'='*60}")
print("【補足】先のスクリプト(summarize_0601.py)での「Hough検出率」は")
print("  → detections_*.csv の総行数(動画全体フレーム)を分母にしていた")
print("  → ボールが映っている区間(フィット窓)を分母にすると検出率は上がる")
