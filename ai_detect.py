"""
YOLOv8 ONNX によるボール検出（cv2.dnn・torch不要）  ── 2026-06-24

Roboflowで学習・書き出した YOLOv8(ONNX) を cv2.dnn で推論し、ボール枠を返す共通モジュール。
計測スクリプト(mu_0609 / spin_sideview / hit_speed)から `--detector ai` で利用する。

使い方:
  from ai_detect import BallDetector
  det = BallDetector("best.onnx", conf=0.25)
  boxes = det.detect(frame)          # [(cx, cy, w, h, conf), ...] 原画素座標
  best  = det.detect_best(frame)     # 最尤1個 or None
  単体テスト:
    python ai_detect.py --onnx best.onnx --image frames/roboflow_test/test01_hit_f573.jpg
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_ONNX = Path(r"c:\Users\bi23043\Documents\4年前期\models\ball_yolo8.onnx")

# AI枠→実径の補正係数。
# 学習ラベルを minEnclosingCircle(球に外接する円)から生成したため、AI枠は
# 球の実径より系統的に大きい。μr斜め動画5本・96サンプルで実測した
#   AI枠短辺 / 実径(面積等価径) = 1.098 ± 0.072 (中央値 1.111)
# より k=0.90。位置(cx,cy)は正確なので径のみ補正する。
# ※スケール(px/cm)に直結するため、μr(減速度)には敏感に効く。
BOX_TO_DIAMETER = 0.90


def letterbox(img, new=640, color=(114, 114, 114)):
    """アスペクト比を保って new×new にパディング。(画像, スケール, 左上pad)を返す。"""
    h, w = img.shape[:2]
    s = min(new / h, new / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new - nh) // 2
    left = (new - nw) // 2
    out = np.full((new, new, 3), color, np.uint8)
    out[top:top + nh, left:left + nw] = resized
    return out, s, left, top


class BallDetector:
    def __init__(self, onnx_path=DEFAULT_ONNX, conf=0.25, iou=0.45, imgsz=640,
                 ball_class=0, box_scale=BOX_TO_DIAMETER):
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNXモデルが見つかりません: {onnx_path}\n"
                "Roboflowで学習→ONNX書き出し→このパスに置いてください。")
        # 日本語パス対策: cv2.dnn はASCII外のパスを開けないため、バイト列から読み込む
        try:
            self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        except cv2.error:
            buf = np.fromfile(str(onnx_path), dtype=np.uint8)
            self.net = cv2.dnn.readNetFromONNX(buf)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.ball_class = ball_class
        self.box_scale = box_scale     # 枠→実径の補正(位置は補正しない)

    def detect(self, frame):
        """frame からボール枠リスト [(cx,cy,w,h,conf)...] を原画素座標で返す。"""
        lb, s, left, top = letterbox(frame, self.imgsz)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, (self.imgsz, self.imgsz),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()              # YOLOv8: (1, 4+nc, 8400)
        out = np.squeeze(out)                 # (4+nc, 8400)
        if out.shape[0] < out.shape[1]:       # (4+nc, N) → (N, 4+nc) に転置
            out = out.T
        # 列: [cx, cy, w, h, class0_score, class1_score, ...]
        boxes_xywh = out[:, :4]
        scores_all = out[:, 4:]
        cls = np.argmax(scores_all, axis=1)
        conf = scores_all[np.arange(len(scores_all)), cls]
        keep = (conf >= self.conf) & (cls == self.ball_class)
        if not keep.any():
            return []
        bx = boxes_xywh[keep]
        cf = conf[keep]
        # letterbox座標 → 原画素
        rects = []
        for (cx, cy, w, h) in bx:
            x = (cx - w / 2 - left) / s
            y = (cy - h / 2 - top) / s
            rects.append([float(x), float(y), float(w / s), float(h / s)])
        idxs = cv2.dnn.NMSBoxes(rects, cf.tolist(), self.conf, self.iou)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()
        result = []
        for i in idxs:
            x, y, w, h = rects[i]
            # 中心は補正せず、径のみ box_scale で実径に補正
            result.append((x + w / 2, y + h / 2,
                           w * self.box_scale, h * self.box_scale, float(cf[i])))
        return result

    def detect_best(self, frame):
        """最も信頼度の高いボール1個 (cx,cy,w,h,conf) or None。"""
        dets = self.detect(frame)
        return max(dets, key=lambda d: d[4]) if dets else None


def _imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser(description="YOLOv8 ONNX ボール検出 単体テスト")
    ap.add_argument("--onnx", default=str(DEFAULT_ONNX))
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    det = BallDetector(args.onnx, conf=args.conf)
    img = _imread(Path(args.image))
    if img is None:
        print("画像を読めません"); return
    dets = det.detect(img)
    print(f"検出 {len(dets)} 個 (conf>={args.conf})")
    for cx, cy, w, h, cf in dets:
        print(f"  ball @({cx:.0f},{cy:.0f}) {w:.0f}x{h:.0f}px  conf={cf:.3f}")
    vis = img.copy()
    for cx, cy, w, h, cf in dets:
        cv2.rectangle(vis, (int(cx - w / 2), int(cy - h / 2)),
                      (int(cx + w / 2), int(cy + h / 2)), (0, 255, 0), 2)
        cv2.putText(vis, f"{cf:.2f}", (int(cx - w / 2), int(cy - h / 2) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    out = Path(args.image).with_name(Path(args.image).stem + "_aidet.jpg")
    ok, buf = cv2.imencode(".jpg", vis)
    if ok:
        open(str(out), "wb").write(buf.tobytes())
        print(f"可視化: {out}")


if __name__ == "__main__":
    main()
