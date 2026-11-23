"""
静止球によるカメラ較正  ── 2026-10-20

既知の距離に置いた静止ボールの見かけの直径から、
  焦点距離 f[px] と 直径の測り方の偏り b[px] を同時に求める。
求めた値は hit_speed.py --detector bgsub の --focal-px / --d-bias にそのまま渡す。

モデル:
  ピンホールカメラでは、球の見かけの直径(放射方向に直交する短い方の径)は
      d = f·D/Z + b        (D=4cm, Z=光軸方向の奥行き)
  床に敷いた巻尺で正面方向の距離を測れば、それがそのまま Z(--dist forward、既定)。
  レンズ→ボール中心の直線距離 r を測った場合(--dist straight)は、画面の端に置いた球では Z と違う。
  画像中心からの距離 ρ[px] を使うと Z = r·f/√(f²+ρ²) なので
      d = D·√(f²+ρ²)/r + b
  を最小二乗でフィットする(f, b が未知数)。
  直径は打球の追跡と同じ定義(hit_speed.bgsub_candidates: 差分マスクの minAreaRect 短辺)。
  定義をそろえることで、b に「白飛び・ぼけ・二値化のしきい値」による偏りがまとめて入る。

撮影の仕方(横長 240fps スロー、打球と同じ照明・同じカメラ位置・ズーム1倍):
  1本の動画につき1か所。最初の2〜3秒はボールなし(背景)→ボールを置いて手を離し3秒。
  距離はレンズからボール中心まで(前面までを測った場合は --add-cm 2)。
  50/75/100/150/200cm など、近い所から遠い所まで広く。画面の端に置いたものも混ぜると良い。

使い方:
  python calib_static.py --pair PXL_a.mp4:50 --pair PXL_b.mp4:75 --pair PXL_c.mp4:100 ...
"""

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

import hit_speed as hs

if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BG_SECONDS = 0.5          # 背景に使う冒頭/末尾の長さ
STEP = 4                  # 何フレームおきに測るか
MIN_D = 18                # これより小さいものは球としない(200cmで約26px)


def iter_frames(path, step, start=0, stop=None):
    """先頭から順に読み、step おきのフレームだけ復号して返す(シークより速い)。"""
    cap = cv2.VideoCapture(str(path))
    i = 0
    while stop is None or i < stop:
        if i < start or (i - start) % step:
            if not cap.grab():
                break
        else:
            ok, fr = cap.read()
            if not ok:
                break
            yield i, fr
        i += 1
    cap.release()


def static_ball_candidates(fr, bg, open_px=9):
    """静止球むけの候補検出。しきい値は hit_speed.bgsub_candidates と同じ(差分>30)。

    違いは2点:
      ・オープニングを大きく(9px)して、球に触れて垂れている糸などの細い線を切り離す
        (円盤の大きさはオープニングでは変わらない)
      ・直径は輪郭の「上半分」への円当てはめ。台のくぼみで球の下側が隠れたり、
        台への映り込みが下にくっついたりしても、上半分(暗いカーテンが背景)は崩れない
    打球(遮るものがない円盤)では、円当てはめの直径と minAreaRect 短辺は一致する。
    Returns [(cx, cy, 直径, 短辺直径, 端で欠けているか), ...]
    """
    H, W = fr.shape[:2]
    dg = cv2.cvtColor(cv2.absdiff(fr, bg), cv2.COLOR_BGR2GRAY)
    m = (dg > 30).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 200:
            continue
        P = c[:, 0, :].astype(float)
        (_, _), (s1, s2), _ = cv2.minAreaRect(c)
        r = min(s1, s2) / 2
        if r < 10:
            continue
        top = P[:, 1].min()
        sel = P[P[:, 1] <= top + r]                      # まず上半球
        for _ in range(3):
            if len(sel) < 12:
                break
            # 円の当てはめ(代数的): x²+y² + Dx + Ey + F = 0
            A = np.column_stack([sel[:, 0], sel[:, 1], np.ones(len(sel))])
            bb = -(sel[:, 0] ** 2 + sel[:, 1] ** 2)
            (Dc, Ec, Fc), *_ = np.linalg.lstsq(A, bb, rcond=None)
            cx, cy = -Dc / 2, -Ec / 2
            rr = np.sqrt(max(cx ** 2 + cy ** 2 - Fc, 1e-9))
            sel = P[P[:, 1] <= cy + 0.2 * rr]             # 赤道の少し下まで使う
        else:
            # 当てはめた円に輪郭の上半分が沿っているか(丸さの確認)
            up = P[P[:, 1] <= cy]
            if len(up) < 12:
                continue
            dev = np.abs(np.hypot(up[:, 0] - cx, up[:, 1] - cy) - rr)
            if np.median(dev) > 0.06 * rr or not (10 < rr < 200):
                continue
            edge = cx - rr < 2 or cy - rr < 2 or cx + rr > W - 2
            out.append((cx, cy, 2 * rr + 1, min(s1, s2), edge))   # +1: 輪郭は画素中心、幅は画素の端まで
    return out


def measure_static(path, fps):
    """動画の中で止まっているボールを見つけ、直径(px)と位置を返す。

    背景は冒頭(ボールを置く前)の中央値。だめなら末尾(ボールを取った後)を試す。
    各フレームの候補のうち、同じ場所に最も長く留まっている丸いものを静止球とする
    (手や体は動くので留まらない)。
    """
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    nb = max(5, int(BG_SECONDS * fps))
    best = None
    step = max(STEP, n // 300)                         # 長い動画でも測るのは300フレーム程度
    for name, (b0, b1) in [("冒頭", (0, min(nb, n))), ("末尾", (max(0, n - nb), n))]:
        bg = np.median(np.array([fr for _, fr in iter_frames(path, 3, b0, b1)]), axis=0).astype(np.uint8)
        obs = []                                       # (frame, cx, cy, d, 短辺)
        for f, fr in iter_frames(path, step):
            for cx, cy, d, d_rect, edge in static_ball_candidates(fr, bg):
                if not edge and d >= MIN_D:
                    obs.append((f, cx, cy, d, d_rect))
        if not obs:
            continue
        O = np.array(obs)
        # 位置でまとめる: 各観測の近く(半径の半分以内)に何フレーム分あるか
        counts = [np.sum(np.hypot(O[:, 1] - o[1], O[:, 2] - o[2]) < 0.25 * o[3]) for o in O]
        c0 = O[int(np.argmax(counts))]
        near = O[np.hypot(O[:, 1] - c0[1], O[:, 2] - c0[2]) < 0.25 * c0[3]]
        # 直径が安定している部分だけ(置く瞬間・手が触れている瞬間を除く)
        med = np.median(near[:, 3])
        near = near[np.abs(near[:, 3] - med) < 0.05 * med]
        if best is None or len(near) > len(best[1]):
            best = (name, near)
        if len(near) >= 20:                            # 冒頭の背景で十分取れたら末尾は試さない
            break
    if best is None or len(best[1]) < 5:
        return None
    name, near = best
    return {"bg": name, "n": len(near), "cx": float(np.median(near[:, 1])), "cy": float(np.median(near[:, 2])),
            "d": float(np.median(near[:, 3])), "d_sd": float(np.std(near[:, 3])),
            "d_rect": float(np.median(near[:, 4])),
            "t": (near[0, 0] / fps, near[-1, 0] / fps)}


def fit_f_b(r_cm, rho_px, d_px, d_sd, r_err_cm=0.5, fix_b=False, forward=True):
    """直径のモデルを最小二乗で解く。誤差は直径のばらつきと巻尺の誤差(±0.5cm)を合成。
    forward=True : r は正面方向(光軸方向)の距離 = Z  → d = f·D/Z + b
    forward=False: r はレンズからの直線距離         → d = D·√(f²+ρ²)/r + b
    """
    D = hs.BALL_DIAM_CM

    def model(p):
        f, b = p[0], (0.0 if fix_b else p[1])
        if forward:
            return f * D / r_cm + b
        return D * np.sqrt(f ** 2 + rho_px ** 2) / r_cm + b

    def sigma(p):
        dd = model(p) - (0.0 if fix_b else p[1])            # 巻尺誤差による直径の誤差 = d·δr/r
        return np.sqrt(np.maximum(d_sd, 0.3) ** 2 + (dd * r_err_cm / r_cm) ** 2)

    p0 = [np.median(d_px * r_cm / D)] + ([] if fix_b else [0.0])
    for _ in range(2):                                     # σ はモデル値に依存するので2回
        s = sigma(p0)
        res = least_squares(lambda p: (d_px - model(p)) / s, x0=p0)
        p0 = res.x
    J = res.jac
    chi2 = float(np.sum(res.fun ** 2)); dof = max(1, len(d_px) - len(res.x))
    cov = np.linalg.pinv(J.T @ J) * max(chi2 / dof, 1.0)
    err = np.sqrt(np.diag(cov))
    return {"f": float(res.x[0]), "f_err": float(err[0]),
            "b": 0.0 if fix_b else float(res.x[1]), "b_err": 0.0 if fix_b else float(err[1]),
            "resid": d_px - model(res.x), "chi2_dof": chi2 / dof}


def main():
    ap = argparse.ArgumentParser(description="静止球によるカメラ較正(焦点距離 f と直径の偏り b)")
    ap.add_argument("--pair", action="append", required=True,
                    help="動画:距離cm (例 PXL_20261025_100000000.mp4:75)。3本以上、近〜遠を広く")
    ap.add_argument("--add-cm", type=float, default=0.0,
                    help="全距離に足す量[cm]。ボールの前面までを測ったなら 2(=半径)")
    ap.add_argument("--fps", type=float, default=hs.EFFECTIVE_FPS)
    ap.add_argument("--tape-err-cm", type=float, default=0.5, help="巻尺で測った距離の誤差[cm]")
    ap.add_argument("--dist", choices=["forward", "straight"], default="forward",
                    help="距離の意味。forward=床に敷いた巻尺で測った正面方向の距離(=奥行きZ。"
                         "球の高さや左右のずれに依らない。カメラが水平であることが前提)。"
                         "straight=レンズからボール中心までの直線距離")
    args = ap.parse_args()

    rows = []
    for item in args.pair:
        name, _, dist = item.rpartition(":")
        path = Path(name)
        if not path.is_absolute():
            path = hs.VIDEO_DIR / path
        if not path.exists():
            print(f"エラー: {path} が見つかりません"); sys.exit(1)
        cap = cv2.VideoCapture(str(path))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        m = measure_static(path, args.fps)
        r = float(dist) + args.add_cm
        if m is None:
            print(f"  {path.name} ({r:.0f}cm): 静止球が見つかりません(最初か最後にボールなしの時間が必要)")
            continue
        m.update(name=path.name, r=r, rho=float(np.hypot(m["cx"] - W / 2, m["cy"] - H / 2)), W=W, H=H)
        rows.append(m)

    if len(rows) < 2:
        print("較正には2本以上(推奨5本以上)が必要です。"); sys.exit(1)
    if len({(x["W"], x["H"]) for x in rows}) > 1:
        print("⚠ 解像度の違う動画が混ざっています。同じ設定で撮った動画だけにしてください。")

    r_cm = np.array([x["r"] for x in rows]); rho = np.array([x["rho"] for x in rows])
    d = np.array([x["d"] for x in rows]); dsd = np.array([x["d_sd"] for x in rows])
    fw = args.dist == "forward"
    fb = fit_f_b(r_cm, rho, d, dsd, args.tape_err_cm, forward=fw)
    f0 = fit_f_b(r_cm, rho, d, dsd, args.tape_err_cm, fix_b=True, forward=fw)

    print("動画                              距離    画面中心から  直径[px](短辺)        背景   使ったフレーム  残差[px]")
    for x, res in zip(rows, fb["resid"]):
        print(f"{x['name']:<32} {x['r']:5.0f}cm  {x['rho']:7.0f}px   {x['d']:6.1f}±{x['d_sd']:.1f} ({x['d_rect']:5.1f})   {x['bg']}   "
              f"{x['n']:3d} ({x['t'][0]:.1f}-{x['t'][1]:.1f}s)  {res:+5.1f}")
    W = rows[0]["W"]
    print()
    print(f"★ 焦点距離 f = {fb['f']:.0f} ± {fb['f_err']:.0f} px   (水平画角 {2*np.degrees(np.arctan(W/2/fb['f'])):.1f}°)")
    print(f"★ 直径の偏り b = {fb['b']:+.1f} ± {fb['b_err']:.1f} px"
          f"   (直径25pxの球なら {fb['b']/25*100:+.0f}%、100pxなら {fb['b']/100*100:+.0f}% の影響)")
    print(f"  当てはまり: χ²/自由度 = {fb['chi2_dof']:.2f}  (1前後なら誤差の見積もりどおり)")
    print(f"  参考: b=0 と仮定した場合 f = {f0['f']:.0f} ± {f0['f_err']:.0f} px, χ²/自由度 = {f0['chi2_dof']:.2f}")
    if abs(fb["b"]) < 2 * fb["b_err"]:
        print("  → b は誤差の範囲で0と区別できません(直径の測り方に目立つ偏りはない)")
    if fb["chi2_dof"] > 4:
        print("  ⚠ 当てはまりが悪い。距離の測り間違い、別の物を球と誤認、ピンぼけ(近すぎ)を確認してください")
    if r_cm.max() / r_cm.min() < 2.5:
        print("  ⚠ 距離の幅が狭いと f と b を分けにくい。近い所と遠い所(例 50cmと200cm)を両方入れてください")
    print()
    print("打球の解析に使うには:")
    print(f"  python hit_speed.py --video <動画> --detector bgsub --focal-px {fb['f']:.0f} --d-bias {fb['b']:.1f}")


if __name__ == "__main__":
    main()
