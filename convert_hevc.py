"""
HEVC(H.265)動画をH.264に変換するスクリプト
Pixel 9aの高速度動画をOpenCVで読めるようにする

使い方:
  python convert_hevc.py                        # デフォルト動画を変換
  python convert_hevc.py PXL_20260521_141007907.mp4
  python convert_hevc.py file1.mp4 file2.mp4    # 複数一括変換

回転フラグを自動検出して正しく補正する。
"""

import json
import subprocess
import sys
from pathlib import Path

VIDEO_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論")
DEFAULT_INPUT = VIDEO_DIR / "PXL_20260514_154725948.mp4"


def get_rotation(video_path: Path) -> int:
    """ffprobeで動画の回転角度を取得する。メタデータがなければ 0。"""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", str(video_path)],
        capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        for sd in stream.get("side_data_list", []):
            if "rotation" in sd:
                return int(sd["rotation"])
    return 0


def rotation_to_vf(rotation: int) -> str | None:
    """
    回転角度 → ffmpeg -vf フィルタ文字列。
    補正不要なら None。
    """
    mapping = {
        -90: "transpose=2",   # 反時計回り90度
        90:  "transpose=1",   # 時計回り90度
        180: "vflip,hflip",
        -180: "vflip,hflip",
    }
    return mapping.get(rotation)


def convert(input_path: Path) -> bool:
    """1ファイルを変換する。成功したら True。"""
    output_path = input_path.parent / (input_path.stem + "_h264.mp4")

    if output_path.exists():
        print(f"  スキップ（変換済み）: {output_path.name}")
        return True

    rotation = get_rotation(input_path)
    vf = rotation_to_vf(rotation)

    cmd = ["ffmpeg", "-i", str(input_path)]
    if vf:
        cmd += ["-vf", vf]
        print(f"  回転補正: {rotation}度 ({vf})")
    else:
        print(f"  回転補正: なし")
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an", "-y",
            str(output_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  変換失敗: {input_path.name}")
        print(result.stderr[-1500:])
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  変換完了: {output_path.name} ({size_mb:.1f} MB)")
    return True


def main():
    if len(sys.argv) > 1:
        targets = []
        for arg in sys.argv[1:]:
            p = Path(arg)
            if not p.is_absolute():
                p = VIDEO_DIR / p
            targets.append(p)
    else:
        targets = [DEFAULT_INPUT]

    ok = 0
    for p in targets:
        print(f"\n[{p.name}]")
        if not p.exists():
            print(f"  エラー: ファイルが見つかりません: {p}")
            continue
        if convert(p):
            ok += 1

    print(f"\n変換完了: {ok}/{len(targets)} ファイル")


if __name__ == "__main__":
    main()
