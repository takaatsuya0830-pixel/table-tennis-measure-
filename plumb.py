"""
下げ振り(おもり付きの糸)から重力の向きを求める  ── 2026-11-09

画面に写った糸は真下(重力の向き)を向いている。ピンホールカメラでは、
3次元の直線を写した画像上の線は「カメラの中心とその直線を含む平面」に対応するので、
糸の線から「重力の向き d はこの平面の中にある」(n·d = 0) という条件が1つ得られる。
  ・糸が1本 → 条件が1つ。カメラのロールとピッチの組み合わせが1つに絞られる(角度が1つ残る)
  ・糸が2本以上(左端と右端など離れた位置) → 平面どうしの交線として d が完全に決まる
糸は動画から取り出した複数のフレームで自動で探す(揺れていても平均で真下になる)。
見つからない/誤る場合は手で座標を与える。

使い方(単体で確認):
  python plumb.py --video PXL_20261020_041918779.mp4
  python plumb.py --video <動画> --save 糸の検出結果.png
"""

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np


def ridge_map(gray, w=4):
    """細い明るい線を強調する: 両隣(±w px)のどちらよりも明るい分。
    カーテンのひだ(幅が広い明暗)や壁の縁(片側だけ明るい段差)は弱くなる。"""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    left = np.roll(g, w, axis=1); right = np.roll(g, -w, axis=1)
    r = np.minimum(g - left, g - right)
    r[:, :w] = 0; r[:, -w:] = 0
    return np.clip(r, 0, None)


def lines_in_image(bg, max_tilt_deg=15.0, min_len_frac=0.2):
    """1枚の画像から、縦に近い細い明るい線の候補を探す。

    Returns [(x1, y1, x2, y2, 平均コントラスト, 残差), ...] (y1 < y2)
    線の位置は各行の明るさの山をサブピクセルで求め、x = a + b·y を外れ値に強く当てはめ直したもの。
    """
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY) if bg.ndim == 3 else bg
    H, W = gray.shape
    R = ridge_map(gray)
    thr = max(6.0, np.percentile(R, 99.5))
    m = (R > thr).astype(np.uint8) * 255
    segs = cv2.HoughLinesP(m, 1, np.pi / 360, threshold=int(0.08 * H),
                           minLineLength=int(min_len_frac * H), maxLineGap=int(0.05 * H))
    if segs is None:
        return []
    cands = []
    for x1, y1, x2, y2 in segs[:, 0, :]:
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        tilt = np.degrees(np.arctan2(abs(x2 - x1), max(y2 - y1, 1)))
        if tilt > max_tilt_deg:
            continue
        cands.append((x1, y1, x2, y2))
    lines = []
    for x1, y1, x2, y2 in cands:
        # 各行で、予測位置の近く(±6px)の明るさの山を放物線補間で求める
        ys, xs, ws = [], [], []
        for y in range(int(y1), int(y2) + 1):
            xp = x1 + (x2 - x1) * (y - y1) / max(y2 - y1, 1)
            a, b = int(xp) - 6, int(xp) + 7
            if a < 1 or b > W - 1:
                continue
            row = R[y, a:b]
            k = int(np.argmax(row))
            if row[k] < 0.5 * thr or k == 0 or k == len(row) - 1:
                continue
            l, c, r = row[k - 1], row[k], row[k + 1]
            den = l - 2 * c + r
            off = 0.5 * (l - r) / den if den != 0 else 0.0
            ys.append(y); xs.append(a + k + off); ws.append(c)
        if len(ys) < 0.15 * H:
            continue
        ys = np.array(ys, float); xs = np.array(xs, float)
        keep = np.ones(len(ys), bool)
        for _ in range(3):                                    # 外れ値に強い直線当てはめ
            co = np.polyfit(ys[keep], xs[keep], 1)
            res = xs - np.polyval(co, ys)
            s = 1.4826 * np.median(np.abs(res[keep])) + 0.1
            nk = np.abs(res) < 3 * s
            if (nk == keep).all():
                break
            keep = nk
        if keep.sum() < 0.15 * H:
            continue
        ya, yb = ys[keep].min(), ys[keep].max()
        contrast = float(np.median(np.array(ws)[keep]))
        lines.append((float(np.polyval(co, ya)), ya, float(np.polyval(co, yb)), yb, contrast,
                      float(np.std(res[keep]))))
    return lines


def find_plumb_lines(frames, max_lines=2, min_presence=0.6, min_contrast=50.0, rel_contrast=0.5):
    """複数のフレームから糸を探す。

    糸は振り子のように揺れることがあり、動画の中央値(背景)では消えてしまう(10/20動画で確認)。
    そこで各フレームで線を探し、同じ位置(中央の高さでの x が近い)のものをまとめて、
    ・多くのフレーム(min_presence 以上)に写っている
    ・周りとの明るさの差(コントラスト)が大きい(暗い背景の前の明るい糸)
    ものを糸とする。傾きはフレームごとの値の平均(揺れの平均=真下)。
    frames に画像1枚を渡してもよい。
    Returns [(x1, y1, x2, y2, コントラスト), ...] (コントラストの強い順、横に離れたもの最大 max_lines 本)
    """
    if isinstance(frames, np.ndarray):
        frames = [frames]
    H, W = frames[0].shape[:2]
    obs = []                                                  # (フレーム番号, x中央, 傾きdx/dy, ytop, ybot, コントラスト)
    for i, fr in enumerate(frames):
        for x1, y1, x2, y2, c, _ in lines_in_image(fr):
            slope = (x2 - x1) / max(y2 - y1, 1)
            xmid = x1 + slope * (H / 2 - y1)
            obs.append((i, xmid, slope, y1, y2, c))
    if not obs:
        return []
    O = np.array(obs)
    clusters = []
    for o in O[np.argsort(-O[:, 5])]:
        for cl in clusters:
            if abs(o[1] - np.median([q[1] for q in cl])) < 40:
                cl.append(o); break
        else:
            clusters.append([o])
    res = []
    for cl in clusters:
        C = np.array(cl)
        n_frames = len(set(C[:, 0].astype(int)))
        if n_frames < min_presence * len(frames):
            continue
        slope = float(np.mean(C[:, 2])); xmid = float(np.median(C[:, 1]))
        ya, yb = float(np.median(C[:, 3])), float(np.median(C[:, 4]))
        x_at = lambda y: xmid + slope * (y - H / 2)
        res.append((x_at(ya), ya, x_at(yb), yb, float(np.median(C[:, 5]))))
    res.sort(key=lambda t: -t[4])
    # 糸は暗い背景の前で明るさの差が大きい(10/20 で約120)。扇風機の縁(約47)や
    # カーテンのひだ(約7〜19)を糸と取り違えないよう、差の小さい線は捨てる
    if res:
        top = res[0][4]
        res = [L for L in res if L[4] >= min_contrast and L[4] >= rel_contrast * top]
    out = []
    for L in res:
        xm = (L[0] + L[2]) / 2
        if all(abs(xm - (o[0] + o[2]) / 2) > 0.05 * W for o in out):
            out.append(L)
        if len(out) >= max_lines:
            break
    return out


def plane_normal(line, f, cx, cy):
    """画像上の線(x1,y1,x2,y2)に対応する、カメラ中心を通る平面の法線(単位ベクトル)。"""
    x1, y1, x2, y2 = line[:4]
    r1 = np.array([x1 - cx, y1 - cy, f]); r2 = np.array([x2 - cx, y2 - cy, f])
    n = np.cross(r1, r2)
    return n / np.linalg.norm(n)


def gravity_constraint(lines, f, cx, cy):
    """糸の線から重力の向きの条件を作る。

    Returns None / ("plane", n) / ("fixed", d)
      plane: 重力の向き d は法線 n の平面の中(n·d=0)。糸1本、または2本がほぼ平行で交線が不安定なとき
      fixed: 重力の向き d(単位ベクトル、画像の下向きが正)が決まった
    """
    if not lines:
        return None
    ns = [plane_normal(L, f, cx, cy) for L in lines]
    if len(ns) >= 2:
        d = np.cross(ns[0], ns[1])
        if np.linalg.norm(d) > 0.05:                         # 2枚の平面が十分に交わる
            d = d / np.linalg.norm(d)
            if d[1] < 0:
                d = -d
            return ("fixed", d)
    return ("plane", ns[0])


def gravity_angles(d):
    """重力の向き d をカメラのピッチ・ロール[度]に直す(hit_speed の gdir と同じ定義)。"""
    th = np.arcsin(np.clip(d[2], -1, 1))
    ph = np.arctan2(d[0], d[1])
    return float(np.degrees(th)), float(np.degrees(ph))


def parse_manual(spec):
    """'x1,y1,x2,y2;x1,y1,x2,y2' → 線のリスト"""
    out = []
    for part in spec.split(";"):
        v = [float(t) for t in part.split(",")]
        if len(v) != 4:
            raise ValueError(f"糸の座標は x1,y1,x2,y2 の4つ: {part}")
        x1, y1, x2, y2 = v
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        out.append((x1, y1, x2, y2, 0.0))
    return out


def sample_frames(path, n_samples=30):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    samp = []
    for f in np.linspace(0, n - 1, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if ok:
            samp.append(fr)
    cap.release()
    return samp


def main():
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description="下げ振り(糸)の検出と重力の向き")
    ap.add_argument("--video", required=True)
    ap.add_argument("--focal-px", type=float, default=1250.0)
    ap.add_argument("--save", default=None, help="検出結果を描いた画像の保存先")
    args = ap.parse_args()
    path = Path(args.video)
    if not path.is_absolute():
        path = Path(r"c:\Users\bi23043\Documents\4年前期\卒論\videos") / path
    frames = sample_frames(path)
    bg = frames[len(frames) // 2]
    H, W = bg.shape[:2]
    lines = find_plumb_lines(frames)
    if not lines:
        print("糸が見つかりませんでした。明るい色の糸を、暗い背景の前に、上下に長く写してください。")
        return
    for i, L in enumerate(lines):
        tilt = np.degrees(np.arctan2(L[2] - L[0], L[3] - L[1]))
        print(f"糸{i+1}: ({L[0]:.1f},{L[1]:.0f})→({L[2]:.1f},{L[3]:.0f})  画面上の傾き {tilt:+.2f}°  長さ {L[3]-L[1]:.0f}px  コントラスト {L[4]:.0f}")
    c = gravity_constraint(lines, args.focal_px, W / 2, H / 2)
    if c[0] == "fixed":
        th, ph = gravity_angles(c[1])
        print(f"→ 重力の向きが決まりました: カメラのピッチ {th:+.2f}°  ロール {ph:+.2f}°")
    else:
        print("→ 糸が1本なので、重力の向きは「この平面の中」という条件のみ(角度が1つ残る)。"
              "左端と右端に1本ずつ吊るすと完全に決まります")
    if args.save:
        im = bg.copy()
        for L in lines:
            cv2.line(im, (int(L[0]), int(L[1])), (int(L[2]), int(L[3])), (0, 0, 255), 3)
        ok, buf = cv2.imencode(".png", im)
        open(args.save, "wb").write(buf.tobytes())


if __name__ == "__main__":
    main()
