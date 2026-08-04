"""
ボール速度計算スクリプト

入力: detect_ball.py が出力した detections_*.csv
出力:
  velocity.csv          ← フレームごとの速度
  velocity_plot.png     ← 速度の時間変化グラフ

使い方:
  # キャリブレーションなし（ピクセル速度のみ）
  python calc_velocity.py --csv detections_20260522_120000.csv

  # キャリブレーションあり（km/h変換）
  python calc_velocity.py --csv detections_20260522_120000.csv --px-per-cm 12.5

  # 補間ギャップ上限を変更（デフォルト3フレーム）
  python calc_velocity.py --csv detections_20260522_120000.csv --max-gap 5

キャリブレーション値の求め方:
  定規を映した参照画像で「30cmが何ピクセルか」を測定し、
  --px-per-cm = (ピクセル数 / 30) を渡す

速度計算方式:
  中心差分法  v[i] = (pos[i+1] - pos[i-1]) / (t[i+1] - t[i-1])
  両端フレームは前進/後退差分で補完。
  検出ギャップが --max-gap を超えるフレームは速度を NaN とし、
  CSV では空欄・グラフでは描画なし。

座標ソース選択:
  --source hough  : Hough検出を優先（デフォルト）
  --source color  : 色検出を優先
  --source auto   : 両手法が一致するフレームのみ使用（最厳格）
  いずれも優先手法が空欄のフレームは他手法にフォールバック。
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EFFECTIVE_FPS = 240  # 撮影メタデータ com.android.capture.fps=240 より確定(旧233は誤り)

CSV_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\detections")


def load_detections(csv_path: Path, source: str) -> tuple[np.ndarray, np.ndarray]:
    """
    CSVを読み込み、フレーム番号と座標(x, y)の配列を返す。
    検出なしのフレームは NaN。
    source: "hough" | "color" | "auto"
    """
    frames, xs, ys = [], [], []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = int(row["frame_number"])
            hx = row["hough_x"].strip()
            hy = row["hough_y"].strip()
            cx = row["color_x"].strip()
            cy = row["color_y"].strip()

            if source == "hough":
                x = float(hx) if hx else (float(cx) if cx else math.nan)
                y = float(hy) if hy else (float(cy) if cy else math.nan)
            elif source == "color":
                x = float(cx) if cx else (float(hx) if hx else math.nan)
                y = float(cy) if cy else (float(hy) if hy else math.nan)
            else:  # auto: 両手法が一致するフレームのみ
                if hx and cx:
                    x = (float(hx) + float(cx)) / 2
                    y = (float(hy) + float(cy)) / 2
                else:
                    x, y = math.nan, math.nan

            frames.append(fn)
            xs.append(x)
            ys.append(y)

    return np.array(frames, dtype=int), np.array(
        list(zip(xs, ys)), dtype=float
    )


def interpolate_gaps(
    frames: np.ndarray,
    positions: np.ndarray,
    max_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    NaN のフレームを線形補間する。
    ギャップが max_gap フレーム以内のみ補間。
    それ以上は NaN のまま残す。

    返り値:
      positions_filled: 補間後の座標配列
      is_interpolated:  補間されたフレームのboolマスク
    """
    pos = positions.copy()
    is_interp = np.zeros(len(frames), dtype=bool)

    nan_mask = np.isnan(pos[:, 0])
    if not nan_mask.any():
        return pos, is_interp

    # NaN区間を特定して補間
    i = 0
    n = len(frames)
    while i < n:
        if not nan_mask[i]:
            i += 1
            continue
        # NaN区間の開始
        gap_start = i
        while i < n and nan_mask[i]:
            i += 1
        gap_end = i  # gap_end は NaN区間の次のインデックス（または n）

        gap_len = gap_end - gap_start
        has_before = gap_start > 0 and not nan_mask[gap_start - 1]
        has_after = gap_end < n and not nan_mask[gap_end]

        if gap_len <= max_gap and has_before and has_after:
            t0 = frames[gap_start - 1]
            t1 = frames[gap_end]
            p0 = pos[gap_start - 1]
            p1 = pos[gap_end]
            for j in range(gap_start, gap_end):
                alpha = (frames[j] - t0) / (t1 - t0)
                pos[j] = p0 + alpha * (p1 - p0)
                is_interp[j] = True

    return pos, is_interp


def calc_velocity(
    frames: np.ndarray,
    positions: np.ndarray,
    fps: float,
) -> np.ndarray:
    """
    中心差分法で速度[px/s]を計算。
    NaN が含まれる差分は NaN になる。
    返り値: shape (n, 2) の [vx, vy] 配列
    """
    n = len(frames)
    vel = np.full((n, 2), math.nan)

    for i in range(n):
        if i == 0 or i == n - 1:
            # 端点: 前進 or 後退差分
            if i == 0:
                dt = (frames[1] - frames[0]) / fps
                dp = positions[1] - positions[0]
            else:
                dt = (frames[-1] - frames[-2]) / fps
                dp = positions[-1] - positions[-2]
        else:
            dt = (frames[i + 1] - frames[i - 1]) / fps
            dp = positions[i + 1] - positions[i - 1]

        if dt > 0 and not np.isnan(dp).any():
            vel[i] = dp / dt

    return vel


def write_velocity_csv(
    out_path: Path,
    frames: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    is_interp: np.ndarray,
    fps: float,
    px_per_cm: float | None,
) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "frame_number", "time_sec",
            "x_px", "y_px",
            "vx_px_s", "vy_px_s", "speed_px_s",
            "interpolated",
        ]
        if px_per_cm is not None:
            header += ["vx_cm_s", "vy_cm_s", "speed_cm_s", "speed_km_h"]
        writer.writerow(header)

        for i, fn in enumerate(frames):
            x, y = positions[i]
            vx, vy = velocities[i]
            spd = math.sqrt(vx**2 + vy**2) if not math.isnan(vx) else math.nan

            def fmt(v):
                return f"{v:.4f}" if not math.isnan(v) else ""

            row = [
                fn,
                f"{fn / fps:.6f}",
                fmt(x), fmt(y),
                fmt(vx), fmt(vy), fmt(spd),
                "1" if is_interp[i] else "0",
            ]
            if px_per_cm is not None:
                vx_cm = vx / px_per_cm if not math.isnan(vx) else math.nan
                vy_cm = vy / px_per_cm if not math.isnan(vy) else math.nan
                spd_cm = spd / px_per_cm if not math.isnan(spd) else math.nan
                spd_kmh = spd_cm * 3600 / 100000 if not math.isnan(spd_cm) else math.nan
                row += [fmt(vx_cm), fmt(vy_cm), fmt(spd_cm), fmt(spd_kmh)]
            writer.writerow(row)


def plot_velocity(
    out_path: Path,
    frames: np.ndarray,
    velocities: np.ndarray,
    is_interp: np.ndarray,
    fps: float,
    px_per_cm: float | None,
) -> None:
    times = frames / fps
    spd_px = np.array([
        math.sqrt(vx**2 + vy**2) if not math.isnan(vx) else math.nan
        for vx, vy in velocities
    ])

    fig, axes = plt.subplots(
        2 if px_per_cm else 1, 1,
        figsize=(12, 8 if px_per_cm else 5),
        sharex=True,
    )
    if px_per_cm is None:
        axes = [axes]

    # ピクセル速度
    ax = axes[0]
    # 補間フレームと実検出フレームを色分け
    real_mask = ~is_interp & ~np.isnan(spd_px)
    interp_mask = is_interp & ~np.isnan(spd_px)
    if real_mask.any():
        ax.plot(
            times[real_mask], spd_px[real_mask],
            "b.", ms=3, label="検出",
        )
    if interp_mask.any():
        ax.plot(
            times[interp_mask], spd_px[interp_mask],
            "c.", ms=3, alpha=0.5, label="補間",
        )
    ax.set_ylabel("速度 [px/s]")
    ax.set_title("ボール速度 (ピクセル基準)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # km/h 速度
    if px_per_cm is not None:
        spd_kmh = np.where(
            np.isnan(spd_px), math.nan, spd_px / px_per_cm * 3600 / 100000
        )
        ax2 = axes[1]
        if real_mask.any():
            ax2.plot(
                times[real_mask], spd_kmh[real_mask],
                "r.", ms=3, label="検出",
            )
        if interp_mask.any():
            ax2.plot(
                times[interp_mask], spd_kmh[interp_mask],
                "m.", ms=3, alpha=0.5, label="補間",
            )
        ax2.set_ylabel("速度 [km/h]")
        ax2.set_xlabel("時間 [秒]")
        ax2.set_title("ボール速度 (km/h換算)")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel("時間 [秒]")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    print(f"グラフ保存: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", required=True,
        help="detections_*.csv のファイル名（CSV_DIR 以下 or フルパス）",
    )
    parser.add_argument(
        "--px-per-cm", type=float, default=None,
        help="1cmあたりのピクセル数（定規参照画像から計測）",
    )
    parser.add_argument(
        "--max-gap", type=int, default=3,
        help="線形補間するギャップの上限フレーム数（デフォルト: 3）",
    )
    parser.add_argument(
        "--source", choices=["hough", "color", "auto"], default="hough",
        help="座標ソース（デフォルト: hough）",
    )
    parser.add_argument(
        "--fps", type=float, default=float(EFFECTIVE_FPS),
        help=f"実効fps（デフォルト: {EFFECTIVE_FPS}）",
    )
    parser.add_argument(
        "--max-speed-kmh", type=float, default=None,
        help="この値(km/h)を超える速度フレームをNaNに除去（偽検出スパイク除去）",
    )
    args = parser.parse_args()

    csv_input = Path(args.csv)
    if not csv_input.is_absolute():
        csv_input = CSV_DIR / csv_input
    if not csv_input.exists():
        print(f"エラー: {csv_input} が見つかりません。")
        sys.exit(1)

    out_dir = csv_input.parent
    stem = csv_input.stem.replace("detections", "velocity")
    csv_out = out_dir / f"{stem}.csv"
    plot_out = out_dir / f"{stem}_plot.png"

    print(f"入力CSV    : {csv_input.name}")
    print(f"座標ソース : {args.source}")
    print(f"実効fps    : {args.fps}")
    print(f"補間上限   : {args.max_gap} フレーム")
    if args.px_per_cm:
        print(f"px/cm      : {args.px_per_cm}")
    print()

    frames, positions = load_detections(csv_input, args.source)

    detected = ~np.isnan(positions[:, 0])
    print(f"全フレーム数    : {len(frames)}")
    print(f"検出あり        : {detected.sum()} フレーム")
    print(f"検出なし        : {(~detected).sum()} フレーム")

    positions_filled, is_interp = interpolate_gaps(
        frames, positions, args.max_gap
    )
    print(f"補間で埋めた数  : {is_interp.sum()} フレーム")

    velocities = calc_velocity(frames, positions_filled, args.fps)

    if args.max_speed_kmh is not None and args.px_per_cm is not None:
        max_spd_px_s = args.max_speed_kmh * 100000 / 3600 * args.px_per_cm
        clipped = 0
        for i, (vx, vy) in enumerate(velocities):
            if not math.isnan(vx):
                spd = math.sqrt(vx**2 + vy**2)
                if spd > max_spd_px_s:
                    velocities[i] = [math.nan, math.nan]
                    clipped += 1
        if clipped:
            print(f"速度上限({args.max_speed_kmh}km/h)超えをNaN化: {clipped}フレーム")

    valid_spd = [
        math.sqrt(vx**2 + vy**2)
        for vx, vy in velocities
        if not math.isnan(vx)
    ]
    if valid_spd:
        print(f"\n速度統計 (ピクセル/秒):")
        print(f"  最大: {max(valid_spd):.1f}")
        print(f"  平均: {sum(valid_spd)/len(valid_spd):.1f}")
        if args.px_per_cm:
            kmh = [v / args.px_per_cm * 3600 / 100000 for v in valid_spd]
            print(f"\n速度統計 (km/h):")
            print(f"  最大: {max(kmh):.2f}")
            print(f"  平均: {sum(kmh)/len(kmh):.2f}")

    write_velocity_csv(
        csv_out, frames, positions_filled, velocities,
        is_interp, args.fps, args.px_per_cm,
    )
    print(f"\nCSV出力: {csv_out}")

    plot_velocity(
        plot_out, frames, velocities, is_interp, args.fps, args.px_per_cm
    )


if __name__ == "__main__":
    main()
