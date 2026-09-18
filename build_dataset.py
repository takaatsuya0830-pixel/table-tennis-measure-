"""
YOLO学習用データセットを古典CV検出で自動生成する  ── 2026-06-24

近接計測動画は古典CVでボールが確実に取れる。それを使ってボール枠を自動ラベル化し、
YOLOv8学習用データセット(images/ labels/)を作る。手作業のラベル付けを最小化する狙い。
（生成後は Roboflow にアップロードしてレビュー/修正 → 学習する）

検出方式は動画の種類で切替（計測スクリプトと同一パラメータ）:
  bright  : μr斜め動画(1348〜1351) — 背景差分+輝度+円形度（mu_0609.detect_track相当）
  framed  : 横アングル回転(1437〜1438) — 連続フレーム差分（spin_sideview.track相当）
  white   : 打球(0645xx) — HSV白球（hit_speed.detect_white_ball相当）

出力(OUT_DIR):
  images/<name>.jpg   labels/<name>.txt(YOLO: "0 cx cy w h"正規化, 負例は空ファイル)
  qc/<name>.jpg       枠を描いた目視確認用
  data.yaml           ローカル学習用(任意)
"""
import io
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

VDIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos")
OUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\dataset")  # frames/はgit無視

# (ファイル名glob, 検出モード, ラベル間引き, 負例間引き, 正例を付けるか, bright用params)
# 横アングル(1437/1438)はフレーム差分が床テクスチャを誤検出するため正例は付けず、
# 「球なし」の負例だけ採用する(床の誤検出抑制に有効)。
# 多様性のため 5/22(机+黒マット,良質4本のみ) と 6/1(高俯瞰・小球) も追加。
# bright params = (min_area, r_min, r_max)
SOURCES = [
    ("PXL_20260609_1348*.mp4", "bright", 3, 120, True, (3000, 40, 130)),
    ("PXL_20260609_1349*.mp4", "bright", 3, 120, True, (3000, 40, 130)),
    ("PXL_20260609_1350*.mp4", "bright", 3, 120, True, (3000, 40, 130)),
    ("PXL_20260609_1351*.mp4", "bright", 3, 120, True, (3000, 40, 130)),
    ("PXL_20260609_1437*.mp4", "framed", 4, 120, False, None),
    ("PXL_20260609_1438*.mp4", "framed", 4, 120, False, None),
    ("PXL_20260623_0645*.mp4", "white", 1, 200, True, None),
    # 5/22 μr採用4本(検出良好と検証済み): 机+黒マット背景・r≈30-45px
    ("PXL_20260522_010535801_h264.mp4", "bright", 4, 150, True, (1800, 25, 90)),
    ("PXL_20260522_010548402_h264.mp4", "bright", 4, 150, True, (1800, 25, 90)),
    ("PXL_20260522_020323568_h264.mp4", "bright", 4, 150, True, (1800, 25, 90)),
    ("PXL_20260522_020337655_h264.mp4", "bright", 4, 150, True, (1800, 25, 90)),
    # 6/1 高俯瞰(小球 r≈25-35px)・フローリング背景
    ("PXL_20260601_0920*_h264.mp4", "bright", 4, 150, True, (1000, 18, 60)),
]


def imwrite(path, img):
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        open(str(path), "wb").write(buf.tobytes())


# ── 検出（計測スクリプトと同一パラメータ） ───────────────────────

def detect_bright(video_path, min_area=3000, r_min=40, r_max=130):
    cap = cv2.VideoCapture(str(video_path)); n = int(cap.get(7))
    samp = []
    for f in range(0, n, 50):
        cap.set(1, f); ok, fr = cap.read()
        if ok: samp.append(fr)
    bg = np.median(np.array(samp), axis=0).astype(np.uint8)
    cap.set(1, 0); fno = 0; det = {}
    while True:
        ok, fr = cap.read()
        if not ok: break
        dg = cv2.cvtColor(cv2.absdiff(fr, bg), cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        m = cv2.bitwise_and((dg > 30).astype(np.uint8), (g > 130).astype(np.uint8)) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_area: continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r_min < r < r_max and a / (np.pi * r * r) > 0.6:
                if best is None or a > best[0]: best = (a, x, y, r)
        if best: det[fno] = (best[1], best[2], best[3])
        fno += 1
    cap.release()
    return det, n


def detect_framed(video_path):
    cap = cv2.VideoCapture(str(video_path)); n = int(cap.get(7))
    prev = None; fno = 0; det = {}
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if prev is None: prev = g; fno += 1; continue
        d = cv2.absdiff(g, prev); prev = g
        m = (d > 28).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) >= 4000:
                (x, y), r = cv2.minEnclosingCircle(c)
                if 50 < r < 140: det[fno] = (x, y, r)
        fno += 1
    cap.release()
    return det, n


def detect_white(video_path):
    cap = cv2.VideoCapture(str(video_path)); n = int(cap.get(7)); fno = 0; det = {}
    while True:
        ok, fr = cap.read()
        if not ok: break
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (0, 0, 160), (180, 55, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 100: continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if 8 < r < 80 and a / (np.pi * r * r) > 0.55:
                if best is None or a > best[0]: best = (a, x, y, r)
        if best: det[fno] = (best[1], best[2], best[3])
        fno += 1
    cap.release()
    return det, n


DETECTORS = {"bright": detect_bright, "framed": detect_framed, "white": detect_white}


def longest_run(frames, max_gap=6):
    if not frames: return []
    frames = sorted(frames)
    runs = [[frames[0]]]
    for f in frames[1:]:
        if f - runs[-1][-1] <= max_gap + 1: runs[-1].append(f)
        else: runs.append([f])
    return max(runs, key=len)


def main():
    # 古い(偽)ラベルを一掃して作り直す（各サブフォルダの中身を削除、ロックは無視）
    for sub in ("images", "labels", "qc"):
        d = OUT_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            try:
                p.unlink()
            except OSError:
                pass

    n_pos = n_neg = 0
    for glob_pat, mode, pos_stride, neg_stride, want_pos, bparams in SOURCES:
        for vp in sorted(VDIR.glob(glob_pat)):
            if mode == "bright" and bparams:
                det, nframes = detect_bright(vp, *bparams)
            else:
                det, nframes = DETECTORS[mode](vp)
            if not det:
                print(f"  {vp.name}: 検出0 → skip"); continue
            run = longest_run(list(det.keys()))
            if len(run) < 8:
                print(f"  {vp.name}: 連続窓{len(run)} → skip"); continue
            f0, f1 = run[0], run[-1]
            # ラベル品質: 窓内の半径中央値から外れる枠を除外
            rs = np.array([det[f][2] for f in run])
            rmed = np.median(rs)
            cap = cv2.VideoCapture(str(vp))
            stem = vp.stem.replace("PXL_2026", "")
            # 正例（窓内・間引き・半径が妥当）。want_pos=False の動画は付けない
            for f in (range(f0, f1 + 1, pos_stride) if want_pos else []):
                if f not in det: continue
                x, y, r = det[f]
                if not (0.6 * rmed < r < 1.6 * rmed): continue
                cap.set(1, f); ok, fr = cap.read()
                if not ok: continue
                H, W = fr.shape[:2]
                bw, bh = 2 * r, 2 * r
                cx, cy = x / W, y / H
                nw, nh = bw / W, bh / H
                if not (0 < cx < 1 and 0 < cy < 1 and 0 < nw < 1 and 0 < nh < 1):
                    continue
                name = f"{stem}_{f:05d}"
                imwrite(OUT_DIR / "images" / f"{name}.jpg", fr)
                (OUT_DIR / "labels" / f"{name}.txt").write_text(
                    f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                vis = fr.copy()
                cv2.rectangle(vis, (int(x - r), int(y - r)), (int(x + r), int(y + r)), (0, 255, 0), 2)
                imwrite(OUT_DIR / "qc" / f"{name}.jpg", vis)
                n_pos += 1
            # 負例（窓外・検出なし＝球が映っていない手/背景のみ）
            for f in range(0, nframes, neg_stride):
                if f0 <= f <= f1 or f in det: continue
                cap.set(1, f); ok, fr = cap.read()
                if not ok: continue
                name = f"{stem}_{f:05d}_neg"
                imwrite(OUT_DIR / "images" / f"{name}.jpg", fr)
                (OUT_DIR / "labels" / f"{name}.txt").write_text("")   # 空=背景
                n_neg += 1
            cap.release()
            print(f"  {vp.name}: 窓 {f0}-{f1}, r中央 {rmed:.0f}px")

    (OUT_DIR / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnc: 1\nnames: ['ball']\n", encoding="utf-8")
    print(f"\n完了: 正例 {n_pos} 枚, 負例 {n_neg} 枚 → {OUT_DIR}")
    print("次: qc/ を目視確認 → Roboflowにimages/+labels/をアップロード → 学習")


if __name__ == "__main__":
    main()
