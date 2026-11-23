"""
試合映像データセット(BlurBall)をYOLO形式に変換する  ── 2026-08-04

外部データセット `blurball_dataset.zip`(プロ試合のラリー動画464本+ボール位置CSV)を
YOLO学習用に変換する。既存の近接データセット(build_dataset.py製)と合わせて学習すれば、
近接映像も試合の引き映像も検出できる汎用モデルになる。

元データ:
  blurball_dataset/<match>/{rallies_videos,videos_rallies}/<NNN>.mp4   1280x720, 約60fps
  all_csv_midpoint_annotations.zip → all_csv_annotations/<match>_csv_<NNN>.csv
      列: Frame, Visibility, X, Y, theta, l   （X,Y はボール中心。Visibility=1 が可視）

変換の考え方:
  - CSVは中心座標のみで枠サイズが無い → 固定サイズの枠を与える(BOX_PX)。
    試合映像のボールは約8〜12px。YOLOは枠中心の回帰が主なので、サイズ一定でも
    「小さく速い球を見つける」学習には十分機能する。
  - Visibility=0 は「ボールが見えない」フレーム → 空ラベル(負例)として一部採用。
  - 連続フレームは冗長なのでストライドで間引く。

出力: 卒論/frames/match_dataset/{images,labels}/ + data.yaml （frames/ はgit無視）
"""
import argparse
import csv as csvmod
import io
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ZIP_PATH = Path(r"C:\Users\bi23043\Downloads\blurball_dataset.zip")
OUT_DIR = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\frames\match_dataset")
BOX_PX = 16          # 試合映像のボール枠サイズ(実測8〜12pxに余裕を持たせる)
POS_STRIDE = 12      # 可視フレームの間引き(連続フレームは冗長)
NEG_PER_VIDEO = 2    # 1動画あたりの負例(ボール不可視フレーム)数


def imwrite_jpg(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        open(str(path), "wb").write(buf.tobytes())


def load_annotations(z):
    """CSV zip を読み、(match, video) -> [(frame, vis, x, y), ...] を返す。"""
    inner = z.read("blurball_dataset/all_csv_midpoint_annotations.zip")
    iz = zipfile.ZipFile(io.BytesIO(inner))
    ann = {}
    for n in iz.namelist():
        if not n.endswith(".csv"):
            continue
        base = n.split("/")[-1].replace(".csv", "")
        parts = base.split("_")
        key = (parts[0], parts[-1])            # (match, video)
        rows = []
        for r in csvmod.DictReader(io.StringIO(iz.read(n).decode("utf-8", "replace"))):
            try:
                rows.append((int(r["Frame"]), int(r["Visibility"]),
                             float(r["X"]), float(r["Y"])))
            except (ValueError, KeyError):
                continue
        ann[key] = rows
    return ann


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=120,
                    help="使用する動画本数(全464本は過多なので既定は一部)")
    ap.add_argument("--stride", type=int, default=POS_STRIDE)
    ap.add_argument("--box", type=int, default=BOX_PX)
    args = ap.parse_args()

    if not ZIP_PATH.exists():
        print(f"エラー: {ZIP_PATH} が見つかりません")
        sys.exit(1)

    for sub in ("images", "labels"):
        d = OUT_DIR / sub
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            try:
                p.unlink()
            except OSError:
                pass

    z = zipfile.ZipFile(ZIP_PATH)
    ann = load_annotations(z)
    vids = [n for n in z.namelist() if n.endswith(".mp4")]

    # マッチが偏らないよう match ごとに散らして選ぶ
    random.seed(0)
    random.shuffle(vids)
    tmp = Path(r"C:\Users\bi23043\AppData\Local\Temp\bb_work")
    tmp.mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = n_vid = 0
    for vpath in vids:
        if n_vid >= args.max_videos:
            break
        parts = vpath.split("/")
        key = (parts[1], parts[-1].replace(".mp4", ""))
        rows = ann.get(key)
        if not rows:
            continue

        local = tmp / "cur.mp4"
        local.write_bytes(z.read(vpath))
        cap = cv2.VideoCapture(str(local))
        if not cap.isOpened():
            continue

        by_frame = {f: (vis, x, y) for f, vis, x, y in rows}
        visible = [f for f, vis, x, y in rows if vis == 1]
        invisible = [f for f, vis, x, y in rows if vis == 0]
        want_pos = set(visible[::args.stride])
        want_neg = set(invisible[::max(1, len(invisible) // max(1, NEG_PER_VIDEO))][:NEG_PER_VIDEO])
        want = want_pos | want_neg
        if not want:
            cap.release(); continue

        for f in sorted(want):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            H, W = fr.shape[:2]
            name = f"m{key[0]}_{key[1]}_{f:05d}"
            imwrite_jpg(OUT_DIR / "images" / f"{name}.jpg", fr)
            if f in want_pos:
                vis, x, y = by_frame[f]
                cx, cy = x / W, y / H
                bw, bh = args.box / W, args.box / H
                if 0 < cx < 1 and 0 < cy < 1:
                    (OUT_DIR / "labels" / f"{name}.txt").write_text(
                        f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    n_pos += 1
                else:
                    (OUT_DIR / "labels" / f"{name}.txt").write_text("")
                    n_neg += 1
            else:
                (OUT_DIR / "labels" / f"{name}.txt").write_text("")
                n_neg += 1
        cap.release()
        n_vid += 1
        if n_vid % 20 == 0:
            print(f"  {n_vid} 本処理  正例 {n_pos} / 負例 {n_neg}")

    (OUT_DIR / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnc: 1\nnames: ['ball']\n", encoding="utf-8")
    print(f"\n完了: 動画 {n_vid} 本 → 正例 {n_pos} 枚, 負例 {n_neg} 枚 → {OUT_DIR}")
    print("次: 近接データセットと合わせてzip化 → Colabで学習")


if __name__ == "__main__":
    main()
