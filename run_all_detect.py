"""
全動画に対して detect_ball.py を一括実行するスクリプト。

使い方:
  python run_all_detect.py                        # videos/ 内の全 _h264.mp4 を処理
  python run_all_detect.py --param2 20 --min-r 5  # パラメータ上書き
  python run_all_detect.py --roi 200 400 1900 950 # ROI指定
  python run_all_detect.py --dry-run              # 実行せずコマンド一覧だけ表示
"""

import argparse
import subprocess
import sys
from pathlib import Path

VIDEOS_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos")
DETECT_SCRIPT = Path(r"c:\Users\bi23043\Documents\4年前期\detect_ball.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param2", type=int, default=20)
    parser.add_argument("--min-r", type=int, default=5)
    parser.add_argument("--max-r", type=int, default=60)
    parser.add_argument("--fps", type=float, default=233.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-jump", type=float, default=150.0)
    parser.add_argument(
        "--roi", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="コマンドを表示するだけで実行しない"
    )
    args = parser.parse_args()

    videos = sorted(VIDEOS_DIR.glob("*_h264.mp4"))
    if not videos:
        print(f"エラー: {VIDEOS_DIR} に *_h264.mp4 が見つかりません。")
        sys.exit(1)

    print(f"対象動画: {len(videos)} 本")
    print()

    results = []
    for i, video in enumerate(videos, 1):
        cmd = [
            sys.executable, str(DETECT_SCRIPT),
            "--video", str(video),
            "--param2", str(args.param2),
            "--min-r", str(args.min_r),
            "--max-r", str(args.max_r),
            "--fps", str(args.fps),
            "--save-every", str(args.save_every),
            "--max-jump", str(args.max_jump),
        ]
        if args.roi:
            cmd += ["--roi"] + [str(v) for v in args.roi]

        print(f"[{i}/{len(videos)}] {video.name}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            print()
            continue

        result = subprocess.run(cmd, capture_output=False, text=True)
        status = "OK" if result.returncode == 0 else f"ERROR(code={result.returncode})"
        results.append((video.name, status))
        print(f"  → {status}")
        print()

    if not args.dry_run:
        print("=" * 50)
        print("一括処理 完了")
        for name, status in results:
            print(f"  {status:20s}  {name}")


if __name__ == "__main__":
    main()
