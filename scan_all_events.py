"""
動画全体をスキャンして「ボールだけが映っているフレーム群」を探す。
ROI内（手を除いた台上）の変化量を30フレームごとにサンプリングし、
変化があるセグメントをリストアップする。
"""
import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論"
    r"\PXL_20260521_141102911_h264.mp4"
)
OUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames")

# ボールが転がる台上エリア（手は画面上半分に出るので y>600 に限定）
ROI_Y0, ROI_Y1 = 600, 900
ROI_X0, ROI_X1 = 200, 1900

STEP = 15  # サンプリング間隔（フレーム）
THRESHOLD = 3000  # ROI内の変化ピクセル数の閾値

cap = cv2.VideoCapture(str(VIDEO_PATH))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"総フレーム数: {total}")

# 最初の20フレームで背景作成
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

# 閾値を超えた区間を特定してグループ化
active = [(fn, sc) for fn, sc in changes if sc > THRESHOLD]
print(f"\nROI内で変化あり（>{THRESHOLD}px）: {len(active)} サンプル")

# 連続フレームをグループ化
groups = []
if active:
    grp_start, grp_end, grp_max = active[0][0], active[0][0], active[0][1]
    for fn, sc in active[1:]:
        if fn - grp_end <= STEP * 3:  # 3ステップ以内は同グループ
            grp_end = fn
            grp_max = max(grp_max, sc)
        else:
            groups.append((grp_start, grp_end, grp_max))
            grp_start, grp_end, grp_max = fn, fn, sc
    groups.append((grp_start, grp_end, grp_max))

print(f"\n検出されたイベント: {len(groups)} 件")
print(f"{'#':>3}  {'開始フレーム':>10}  {'終了フレーム':>10}  {'実時間(秒)':>12}  {'最大変化px':>10}")
for i, (s, e, mx) in enumerate(groups):
    print(
        f"{i+1:>3}  {s:>10}  {e:>10}"
        f"  {s/233:.2f}〜{e/233:.2f}s  {mx:>10}"
    )

# 各グループの中央フレームを画像として保存
cap2 = cv2.VideoCapture(str(VIDEO_PATH))
for i, (s, e, mx) in enumerate(groups):
    mid = (s + e) // 2
    cap2.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap2.read()
    if not ret:
        continue
    out_path = OUT_DIR / f"event_{i+1:02d}_frame{mid:05d}.png"
    ok, buf = cv2.imencode(".png", frame)
    if ok:
        buf.tofile(str(out_path))
cap2.release()
print(f"\n各イベントの中央フレームを {OUT_DIR} に保存しました (event_NN_*.png)")
