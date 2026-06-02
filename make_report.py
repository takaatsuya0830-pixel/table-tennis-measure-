"""ゼミ進捗報告 Word文書 生成スクリプト"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

doc = Document()

# ── ページ設定 (A4) ─────────────────────────────────────
section = doc.sections[0]
section.page_width   = Cm(21.0)
section.page_height  = Cm(29.7)
section.left_margin  = Cm(2.5)
section.right_margin = Cm(2.5)
section.top_margin   = Cm(2.5)
section.bottom_margin = Cm(2.0)

# ── ヘルパー ─────────────────────────────────────────────
def set_font(run, name="游ゴシック", size=11, bold=False, color=None):
    run.font.name = name
    run.font.element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(doc, text, level=1, size=13, color=(0, 70, 127)):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after  = Pt(3)
    run = p.add_run(text)
    set_font(run, size=size, bold=True, color=color)
    return p

def body(doc, text, indent=0, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    run = p.add_run(text)
    set_font(run, size=10.5)
    return p

def bullet(doc, text, indent=0.8):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent  = Cm(indent)
    p.paragraph_format.space_after  = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    run = p.add_run(text)
    set_font(run, size=10.5)
    return p

def hline(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "4472C4")
    pBdr.append(bottom)
    pPr.append(pBdr)

def cell_shading(cell, fill_hex):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tcPr.append(shd)

def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr_row = table.rows[0]
    for j, h in enumerate(headers):
        cell = hdr_row.cells[j]
        cell.text = h
        r = cell.paragraphs[0].runs[0]
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.name = "游ゴシック"
        r.font.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
        r.font.color.rgb = RGBColor(255, 255, 255)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell_shading(cell, "4472C4")
    for i, row_data in enumerate(rows):
        tr = table.rows[i + 1]
        fill = "D9E2F3" if i % 2 == 0 else "FFFFFF"
        for j, val in enumerate(row_data):
            cell = tr.cells[j]
            cell.text = val
            r = cell.paragraphs[0].runs[0]
            r.font.size = Pt(9.5)
            r.font.name = "游ゴシック"
            r.font.element.rPr.rFonts.set(qn("w:eastAsia"), "游ゴシック")
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell_shading(cell, fill)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph()

# ════════════════════════════════════════════════════════
#  タイトル
# ════════════════════════════════════════════════════════
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(0)
p.paragraph_format.space_after  = Pt(4)
set_font(p.add_run("卒業研究 進捗報告"), size=20, bold=True, color=(0, 70, 127))

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
p2.paragraph_format.space_after = Pt(2)
set_font(p2.add_run("卓球ボールの転がり摩擦係数・回転速度の計測システム開発"),
         size=13, bold=True, color=(68, 114, 196))

p3 = doc.add_paragraph()
p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
p3.paragraph_format.space_after = Pt(8)
today = datetime.date.today().strftime("%Y年%m月%d日")
set_font(p3.add_run(f"高橋敦也    {today}"), size=11, color=(80, 80, 80))

hline(doc)

# ════════════════════════════════════════════════════════
#  1. 研究目的
# ════════════════════════════════════════════════════════
heading(doc, "1. 研究目的")
body(doc, ("卓球ボール（白色、直径 40 mm）が床面を転がる際の転がり摩擦係数 "
           "μr を高速度撮影と画像解析によって定量的に計測する。"
           "さらに、ボールに描いた黒点の角度変化から回転速度（rps）を"
           "算出する手法を新たに開発し、転がり運動の詳細な解析を目指す。"))

# ════════════════════════════════════════════════════════
#  2. 計測システム構成
# ════════════════════════════════════════════════════════
heading(doc, "2. 計測システム構成")

heading(doc, "  2-1. 機材", level=2, size=11, color=(68, 114, 196))
add_table(doc,
    ["項目", "内容"],
    [
        ["カメラ",   "Google Pixel 9a（スローモーション 8×）"],
        ["有効 fps", "約 233 fps（動画メタデータから算出）"],
        ["解像度",   "1920 × 1080 px"],
        ["対象",     "白色卓球ボール（直径 40 mm）"],
        ["床面",     "室内床（フローリング系）"],
    ],
    col_widths=[3.5, 12.5]
)

heading(doc, "  2-2. 解析パイプライン", level=2, size=11, color=(68, 114, 196))
for s in [
    "① convert_hevc.py    : HEVC → H.264 変換（ffmpeg、回転補正自動適用）",
    "② calibrate.py       : ボール直径（40 mm）から px/cm を算出",
    "③ detect_ball.py     : Hough 円変換 + 色検出でボール位置を CSV 出力",
    "④ calc_velocity.py   : 位置差分から速度（km/h）を算出・グラフ出力",
    "⑤ final_plot.py      : 転がり区間を切り出し、線形減速フィットで μr を算出",
    "⑥ detect_spin.py     : 黒点の角度変化を追跡して回転数（rps）を算出（新規開発）",
]:
    bullet(doc, s)

# ════════════════════════════════════════════════════════
#  3. キャリブレーション
# ════════════════════════════════════════════════════════
heading(doc, "3. キャリブレーション")
body(doc, "ボール直径の既知値（40 mm = 4 cm）を用いてスケール（px/cm）を算出。")
add_table(doc,
    ["セッション", "撮影日", "アングル", "ボール半径 [px]", "px/cm"],
    [
        ["5/22 セッション", "2026-05-22", "俯瞰（低位置）", "約 94 px", "47.0"],
        ["6/1 セッション",  "2026-06-01", "俯瞰（高位置）", "約 61 px", "30.5"],
    ],
    col_widths=[3.5, 3.0, 3.5, 4.0, 2.0]
)

# ════════════════════════════════════════════════════════
#  4. 転がり摩擦係数の計測結果
# ════════════════════════════════════════════════════════
heading(doc, "4. 転がり摩擦係数 μr の計測結果")
body(doc, "算出式：v(t) = max(v₀ − a·t, 0) を最小二乗フィット → μr = a [m/s²] / g　（g = 9.81 m/s²）")

heading(doc, "  4-1. 5/22 セッション", level=2, size=11, color=(68, 114, 196))
add_table(doc,
    ["試行", "初速 v₀ [km/h]", "減速度 a [km/h/s]", "μr", "採否"],
    [
        ["Trial 1", "4.55", "3.33", "0.094", "採用"],
        ["Trial 2", "3.85", "2.33", "0.066", "採用"],
        ["Trial 3", "4.54", "2.97", "0.084", "採用"],
        ["Trial 4", "7.03", "6.85", "0.194", "除外（初速異常）"],
        ["Trial 5", "7.17", "6.80", "0.193", "除外（初速異常）"],
        ["採用平均", "—", "—", "0.081 ± 0.012", "★ 主結果"],
    ],
    col_widths=[2.5, 3.5, 3.5, 3.5, 3.0]
)

heading(doc, "  4-2. 6/1 セッション（カメラ高位置）", level=2, size=11, color=(68, 114, 196))
add_table(doc,
    ["試行（動画）", "Hough 検出率", "μr", "採否"],
    [
        ["091918182", " 5.0%", "—",             "除外（検出不足）"],
        ["091944934", "20.0%", "≈ 0.050",        "採用"],
        ["092040464", "16.1%", "≈ 0.047",        "採用"],
        ["092052984", "24.6%", "≈ 0.053",        "採用"],
        ["092102746", "25.8%", "0.010",          "除外（停止区間捕捉）"],
        ["採用平均",  "—",     "0.050 ± 0.006",  "参考値"],
    ],
    col_widths=[4.0, 3.0, 3.0, 6.0]
)

heading(doc, "  4-3. セッション比較・考察", level=2, size=11, color=(68, 114, 196))
add_table(doc,
    ["", "5/22 セッション", "6/1 セッション"],
    [
        ["μr（平均）",  "0.081",      "0.050"],
        ["μr（±SD）",  "±0.012",     "±0.006"],
        ["カメラ距離",  "近い",       "遠い"],
        ["球の見かけ半径", "約 94 px","約 61 px"],
        ["Hough 検出品質","良好・密",  "疎・誤検出あり"],
    ],
    col_widths=[4.5, 4.5, 4.5]
)
body(doc, ("2セッションで μr に差異（0.081 vs 0.050）が生じた主因は"
           "カメラ高さの違いによる検出精度の低下と考えられる。"
           "検出品質が高い 5/22 セッションの μr = 0.081 ± 0.012 を"
           "主結果として採用する。文献値（μr ≈ 0.02〜0.10）の範囲内。"))

# ════════════════════════════════════════════════════════
#  5. 回転速度計測システム（新規開発）
# ════════════════════════════════════════════════════════
heading(doc, "5. 回転速度計測システムの開発（新規）")

heading(doc, "  5-1. 手法概要", level=2, size=11, color=(68, 114, 196))
for s in [
    "ボール赤道付近に黒点（マーカー）を描き、233 fps で撮影",
    "各フレームで Hough 円変換によりボール中心・半径を特定",
    "ボール内部（半径の 90% 以内）を輝度閾値で二値化 → 最大暗ブロブの重心 = 黒点",
    "atan2(Δy, Δx) で黒点角度を算出し、フレーム間差分をアンラップして累積角度を計算",
    "累積角度 vs 時間の線形フィット傾きから rps を算出",
    "ナイキスト限界 = 233 / 2 = 116.5 rps（これを超えるとエイリアシング警告）",
]:
    bullet(doc, s)

heading(doc, "  5-2. 初期計測結果（2026-06-01）", level=2, size=11, color=(68, 114, 196))
add_table(doc,
    ["動画", "rps", "rpm", "1回転フレーム数", "黒点検出率", "信頼性"],
    [
        ["142552791", "1.39", " 83.6", "167",  "20.5%", "△ ノイズ多"],
        ["142607743", "0.26", " 15.7", "890",  "26.5%", "△ ノイズ多"],
        ["142707105", "0.92", " 55.0", "254",  "16.1%", "△ ノイズ多"],
        ["142717570", "0.23", " 13.7", "1022", "14.9%", "△ ノイズ多"],
        ["142729147", "0.54", " 32.3", "433",  "18.6%", "△ ノイズ多"],
    ],
    col_widths=[3.5, 2.0, 2.0, 3.5, 3.0, 2.5]
)
body(doc, ("【現状の課題】累積角度が単調増加せず急変警告が多発。"
           "黒点ではなくボールのつなぎ目や影を誤検出している可能性が高い。"
           "フレーム画像の目視確認と dark_thresh パラメータの最適化が次の作業。"))

# ════════════════════════════════════════════════════════
#  6. 今後の課題
# ════════════════════════════════════════════════════════
heading(doc, "6. 今後の課題・スケジュール")
add_table(doc,
    ["優先度", "課題", "具体的内容"],
    [
        ["高", "回転検出精度の向上",
         "dark_thresh 最適化、フレーム画像目視確認、誤検出除去ロジック追加"],
        ["高", "転がり×回転の相関分析",
         "初期回転速度と μr の関係をグラフ化・考察"],
        ["高", "論文執筆開始",
         "序論・手法・結果・考察の各章を順次執筆"],
        ["中", "データ量の増加",
         "各条件で試行数を増やし統計的信頼性を向上（目標：各条件 5 試行以上）"],
        ["中", "床材依存性の検討",
         "カーペット・タイル等の別床材での計測を追加"],
    ],
    col_widths=[1.5, 4.0, 10.5]
)

# ════════════════════════════════════════════════════════
#  フッター
# ════════════════════════════════════════════════════════
footer = section.footer
fp = footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_font(fp.add_run("卓球ボール転がり摩擦係数計測システム開発　進捗報告"), size=9, color=(120, 120, 120))

# 保存
out = r"c:\Users\bi23043\Documents\4年前期\卒論\ゼミ進捗報告.docx"
doc.save(out)
print(f"保存完了: {out}")
