"""キャリブレーション画像に推定端点を描画して確認用に保存する"""
import cv2
import numpy as np
from pathlib import Path

IMG_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\calib_frame.png"
)
OUT_PATH = Path(
    r"c:\Users\bi23043\Documents\4年前期\卒論\frames\calib_annotated.png"
)

# analyze_calib.py の結果
Y = 749
X0 = 343   # 推定左端（0cm候補）
X1 = 1283  # 推定右端（30cm候補）
PX_PER_CM = (X1 - X0) / 30.0

img = cv2.imdecode(
    np.fromfile(str(IMG_PATH), dtype=np.uint8), cv2.IMREAD_COLOR
)

# 端点に赤丸と緑線を描画
cv2.circle(img, (X0, Y), 12, (0, 0, 255), 3)
cv2.circle(img, (X1, Y), 12, (0, 0, 255), 3)
cv2.line(img, (X0, Y), (X1, Y), (0, 255, 0), 2)
cv2.putText(img, f"0cm? ({X0},{Y})", (X0 - 20, Y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(img, f"30cm? ({X1},{Y})", (X1 - 80, Y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.putText(
    img,
    f"{X1-X0}px = 30cm  -> px_per_cm={PX_PER_CM:.2f}",
    (20, img.shape[0] - 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
)

ok, buf = cv2.imencode(".png", img)
if ok:
    buf.tofile(str(OUT_PATH))
    print(f"保存: {OUT_PATH}")
    print(f"推定 px_per_cm = {PX_PER_CM:.3f}")
    print("calib_annotated.png を開いて赤丸が定規の0cmと30cmに一致しているか確認してください。")
