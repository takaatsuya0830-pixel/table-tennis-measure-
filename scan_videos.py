"""
複数動画をまとめてスキャンしてイベントを検出する。
"""
import cv2
import numpy as np
from pathlib import Path

VIDEO_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論")
VIDEOS = [
    "PXL_20260522_010519552_h264.mp4",
    "PXL_20260522_010535801_h264.mp4",
    "PXL_20260522_010548402_h264.mp4",
    "PXL_20260522_010558346_h264.mp4",
]

ROI_Y0, ROI_Y1 = 600, 900
ROI_X0, ROI_X1 = 200, 1900
STEP = 15
THRESHOLD = 3000


def scan(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{'='*60}")
    print(f"{video_path.name}  総フレーム: {total} ({total/233:.1f}s)")

    bg_frames = []
    for _ in range(20):
        ret, f = cap.read()
        if ret:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            bg_frames.append(g[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1])
    bg = np.median(bg_frames, axis=0).astype(np.uint8)

    changes = []
    for i in range(0, total, STEP):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
        diff = cv2.absdiff(roi_gray, bg)
        score = int(np.sum(diff > 25))
        changes.append((i, score))
    cap.release()

    active = [(fn, sc) for fn, sc in changes if sc > THRESHOLD]
    groups = []
    if active:
        gs, ge, gm = active[0][0], active[0][0], active[0][1]
        for fn, sc in active[1:]:
            if fn - ge <= STEP * 3:
                ge = fn
                gm = max(gm, sc)
            else:
                groups.append((gs, ge, gm))
                gs, ge, gm = fn, fn, sc
        groups.append((gs, ge, gm))

    print(f"検出イベント: {len(groups)} 件")
    print(f"{'#':>3}  {'開始':>7}  {'終了':>7}  {'時間(s)':>18}  {'最大変化px':>10}")
    for i, (s, e, mx) in enumerate(groups):
        print(f"{i+1:>3}  {s:>7}  {e:>7}  {s/233:.2f}〜{e/233:.2f}s  {mx:>10}")
    return groups


for v in VIDEOS:
    p = VIDEO_DIR / v
    if p.exists():
        scan(p)
    else:
        print(f"\n{v}: ファイルなし")
