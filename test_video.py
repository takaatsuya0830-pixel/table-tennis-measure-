"""
変換済みH.264動画のフレーム読み込みテスト
convert_hevc.py で変換後に実行する

確認内容:
- OpenCVで開けるか
- フレーム情報（解像度、fps、フレーム数）
- 連続100フレームのデコード
- サンプルフレームの保存
"""

import argparse
import cv2
import numpy as np
import sys
from pathlib import Path


def imwrite(path: Path, img: np.ndarray) -> None:
    """cv2.imwrite の日本語パス対応版。"""
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        buf.tofile(str(path))


VIDEO_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論")
DEFAULT_VIDEO = VIDEO_DIR / "PXL_20260521_141102911_h264.mp4"
EFFECTIVE_FPS = 240  # 実効fps（30fps格納 × 8倍スロー = 240fps、capture.fpsメタデータと一致）
OUTPUT_DIR = VIDEO_DIR / "frames"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=str, default=str(DEFAULT_VIDEO),
        help="テストする動画ファイルパス",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = VIDEO_DIR / video_path

    if not video_path.exists():
        print(f"エラー: {video_path} が見つかりません。")
        print("先に convert_hevc.py を実行してH.264変換を完了してください。")
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("エラー: 動画ファイルを開けませんでした。")
        sys.exit(1)

    file_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_real = total_frames / EFFECTIVE_FPS

    print("=" * 55)
    print("動画ファイル情報")
    print("=" * 55)
    print(f"ファイル     : {video_path.name}")
    print(f"解像度       : {width} x {height}")
    print(f"格納fps      : {file_fps:.2f} fps")
    print(f"総フレーム数 : {total_frames}")
    print(f"実効fps      : {EFFECTIVE_FPS} fps")
    print(f"実時間換算   : {duration_real:.3f} 秒")
    print(f"フレーム間隔 : {1000/EFFECTIVE_FPS:.2f} ms/frame")
    print("=" * 55)

    # 1フレーム目
    ret, frame = cap.read()
    if not ret:
        print("エラー: 1フレーム目を読み込めませんでした。")
        sys.exit(1)
    print(f"OK  1フレーム目読み込み成功  shape={frame.shape}  dtype={frame.dtype}")

    # 連続100フレーム
    success = 1
    last_frame = frame
    for i in range(99):
        ret, f = cap.read()
        if ret:
            success += 1
            last_frame = f
        else:
            print(f"  フレーム {i+2} でデコード失敗")
            break
    print(f"OK  連続デコード: {success}/100 フレーム成功")

    # サンプルフレーム保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "sample_frame001.png"
    imwrite(out_path, frame)
    print(f"OK  サンプル保存: {out_path}")

    out_path2 = OUTPUT_DIR / "sample_frame100.png"
    imwrite(out_path2, last_frame)
    print(f"OK  サンプル保存: {out_path2}")

    cap.release()
    print()
    print("フレーム読み込みテスト完了。")
    print("次のステップ: detect_ball.py でボール検出を試してください。")
    print("  python detect_ball.py")


if __name__ == "__main__":
    main()
