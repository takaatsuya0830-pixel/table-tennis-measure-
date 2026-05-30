"""
ボールが最初に出現するフレームを素早く探す。
フレーム差分（背景との変化量）が大きくなった最初のフレームを検出する。
"""
import cv2
import numpy as np
from pathlib import Path

VIDEO_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論"
    r"\PXL_20260521_141102911_h264.mp4"
)
OUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames")
STEP = 30   # 何フレームおきに調べるか

cap = cv2.VideoCapture(str(VIDEO_PATH))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"総フレーム数: {total}  サンプリング間隔: {STEP}フレーム")

# 最初の10フレームで背景を作成（メディアン）
bg_frames = []
for _ in range(10):
    ret, f = cap.read()
    if ret:
        bg_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
bg = np.median(bg_frames, axis=0).astype(np.uint8)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

changes = []
for i in range(0, total, STEP):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, bg)
    score = float(np.sum(diff > 30))  # 変化画素数
    changes.append((i, score))
    if i % 300 == 0:
        print(f"  frame {i:5d}: change={score:.0f}", end="\r")

cap.release()
print()

# 変化が大きいフレームを上位10個表示
changes.sort(key=lambda x: -x[1])
print("\n変化量トップ10フレーム:")
for fn, sc in changes[:10]:
    print(f"  frame {fn:5d}  変化px={sc:.0f}")

# 最初に閾値を超えたフレームを探す（ボール登場フレーム推定）
changes_sorted_by_frame = sorted(changes, key=lambda x: x[0])
threshold = max(c[1] for c in changes) * 0.3
first_change = None
for fn, sc in changes_sorted_by_frame:
    if sc > threshold:
        first_change = fn
        break

print(f"\n推定ボール登場フレーム: {first_change}")
print(f"  → {first_change / 233:.3f} 秒")

# 代表フレームを保存
if first_change is not None:
    cap2 = cv2.VideoCapture(str(VIDEO_PATH))
    for sample_fn in [first_change, first_change + 30, first_change + 60,
                      first_change + 120, first_change + 240]:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, sample_fn)
        ret, frame = cap2.read()
        if not ret:
            break
        out_path = OUT_DIR / f"ball_search_{sample_fn:05d}.png"
        ok, buf = cv2.imencode(".png", frame)
        if ok:
            buf.tofile(str(out_path))
    cap2.release()
    print(f"\nサンプル画像を {OUT_DIR} に保存しました（ball_search_*.png）")
