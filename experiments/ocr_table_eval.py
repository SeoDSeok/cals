# batch_table_ocr_eval.py
# 모든 이미지/라벨 쌍에 대해 Tesseract/EasyOCR/PaddleOCR 3모델로 표 OCR 성능을 배치 평가

import os, re, json, math, glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from rapidfuzz import fuzz, process
from bs4 import BeautifulSoup

# ===================== 공통 유틸 =====================

def norm_ws(s: str) -> str:
    if s is None: return ""
    s = str(s).strip()
    s = re.sub(r"[‐-‒–—−]", "-", s)     # 여러 대시 → '-'
    s = re.sub(r"\s+", " ", s)         # 다중 공백 → 1칸
    return s

def norm_numeric(s: str) -> Optional[int]:
    if s is None: return None
    t = str(s).strip()
    if t in ("", "-", "—", "–"): return None
    neg = "△" in t or t.startswith("△")
    t = t.replace("△","").replace(",","").replace(" ","")
    m = re.findall(r"\d+", t)
    if not m: return None
    n = int("".join(m))
    return -n if neg else n

def levenshtein_acc(pred: str, gt: str) -> float:
    pred = pred or ""
    gt   = gt or ""
    if len(gt) == 0: return float('nan')
    la, lb = len(gt), len(pred)
    dp = list(range(lb+1))
    for i, ca in enumerate(gt, 1):
        prev = dp[0]; dp[0] = i
        for j, cb in enumerate(pred, 1):
            cur = dp[j]
            if ca == cb: dp[j] = prev
            else: dp[j] = min(prev+1, dp[j]+1, dp[j-1]+1)
            prev = cur
    dist = dp[lb]
    return max(0.0, 1.0 - dist/max(1,len(gt)))

# ===================== PP-Structure: 표 구조/셀 좌표 =====================

def ppstruct_table(image_path: str, pp_lang="en", use_gpu=False):
    """
    PP-Structure(table+ocr=True)로
      - 표 HTML (구조)
      - 셀 bbox 목록(cell_bbox)  (각 <td>와 같은 순서)
      - PP 자체 OCR 텍스트(백업용)
    을 반환
    """
    from paddleocr import PPStructure
    img = cv2.imread(image_path)
    engine = PPStructure(table=True, ocr=False, show_log=False, use_gpu=use_gpu, lang=pp_lang)
    results = engine(img)
    tables = [r for r in results if r.get("type")=="table" and r.get("res",{}).get("html")]
    if not tables:
        raise RuntimeError("테이블을 찾지 못했습니다.")
    # 가장 큰 테이블
    def area(b): return max(1,b[2]-b[0])*max(1,b[3]-b[1])
    tables.sort(key=lambda r: area(r.get("bbox",[0,0,0,0])), reverse=True)
    tab = tables[0]
    res = tab["res"]
    html = res["html"]
    # 셀 bbox와 PP OCR 텍스트(있으면)
    cell_bbox = res.get("cell_bbox", [])  # [[x1,y1,x2,y2], ...]  (<td> 순서)
    cell_text = res.get("cells", None)    # 버전에 따라 없을 수 있음
    return html, cell_bbox, cell_text, img

def html_to_df(html: str) -> pd.DataFrame:
    dfs = pd.read_html(html)
    if dfs:
        df = max(dfs, key=lambda d:(d.shape[0], d.shape[1]))
    else:
        # 드문 실패 대응: 수동 파싱
        soup = BeautifulSoup(html, "lxml")
        rows=[]
        for tr in soup.select("tr"):
            cells=[norm_ws(td.get_text(" ")) for td in tr.find_all(["td","th"])]
            if cells: rows.append(cells)
        df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows)>1 else pd.DataFrame(rows)
    # 멀티헤더 평탄화
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x)!="nan"]).strip()
                      for tup in df.columns.values]
    df.columns = [norm_ws(c) for c in df.columns]
    df = df.applymap(lambda x: norm_ws(str(x)))
    # 빈 헤더는 col#로
    for i,c in enumerate(list(df.columns)):
        if not c:
            df.rename(columns={c: f"col{i}"}, inplace=True)
    return df

# ===================== OCR 엔진들 =====================

class TesseractOCR:
    name = "tesseract"
    def __init__(self, lang="kor+eng", psm=6, oem=1):
        import shutil, pytesseract
        cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        if not cmd:
            raise RuntimeError("tesseract not found (set TESSERACT_CMD or install tesseract-ocr).")
        pytesseract.pytesseract.tesseract_cmd = cmd
        _ = pytesseract.get_tesseract_version()
        self.t = pytesseract
    def ocr(self, bgr):
        return self.t.image_to_string(bgr, lang="kor+eng", config="--oem 1 --psm 6")

class EasyOCREngine:
    name = "easyocr"
    def __init__(self, langs=None, gpu=True):
        import easyocr
        self.reader = easyocr.Reader(langs or ['ko','en'], gpu=gpu)
    def ocr(self, bgr):
        res = self.reader.readtext(bgr[..., ::-1])  # expects RGB array
        return " ".join([t for (_,t,_) in res]) if res else ""

class PaddleOCREngine:
    name = "paddleocr"
    def __init__(self, lang='korean', use_gpu=False, rec_batch_num=32):
        from paddleocr import PaddleOCR
        self.engine = PaddleOCR(lang=lang, det=False, rec=True, cls=True,
                                use_angle_cls=True, use_gpu=use_gpu,
                                rec_batch_num=rec_batch_num, show_log=False)
    def ocr(self, bgr):
        res = self.engine.ocr(bgr, det=False, rec=True, cls=True)
        if not res: return ""
        # 다양한 반환형 대응
        if isinstance(res[0], list) and len(res[0])==2 and isinstance(res[0][0], str):
            return res[0][0]
        return " ".join([r[0] for r in res])

OCR_FACTORIES = {
    "tesseract": lambda: TesseractOCR(),
    "easyocr":   lambda: EasyOCREngine(['ko','en'], gpu=True),
    "paddleocr": lambda: PaddleOCREngine('korean', use_gpu=False, rec_batch_num=32),
}

# ===================== GT 로딩 & 매칭 =====================

def load_gt_table(json_path: str) -> pd.DataFrame:
    rows = json.load(open(json_path, "r", encoding="utf-8"))
    df = pd.DataFrame(rows)
    df.columns = [norm_ws(c) for c in df.columns]
    df = df.applymap(lambda x: norm_ws(x))
    return df

def map_headers(pred_cols: List[str], gt_cols: List[str]) -> Dict[str,str]:
    mapping, used = {}, set()
    for pc in pred_cols:
        cand = [(gc, fuzz.token_sort_ratio(pc, gc)) for gc in gt_cols if gc not in used]
        if not cand: continue
        gc,_ = max(cand, key=lambda x:x[1])
        mapping[pc] = gc; used.add(gc)
    return mapping

def map_rows(pred_df: pd.DataFrame, gt_df: pd.DataFrame, key_col="공종") -> Dict[int,int]:
    if key_col not in gt_df.columns:
        return {i:i for i in range(min(len(gt_df), len(pred_df)))}
    pred_key = process.extractOne(key_col, pred_df.columns, scorer=fuzz.token_sort_ratio)[0]
    pred_keys = pred_df[pred_key].tolist()
    mapping, used = {}, set()
    for i, gt_val in enumerate(gt_df[key_col].tolist()):
        cands = process.extract(gt_val, pred_keys, scorer=fuzz.token_sort_ratio, limit=5)
        pick = next(((v,sc,j) for v,sc,j in cands if j not in used), None)
        if pick is None: continue
        mapping[i] = pick[2]; used.add(pick[2])
    return mapping

# ===================== 셀 OCR & DF 채우기 =====================

def fill_df_with_cell_ocr(df_struct: pd.DataFrame, cell_bboxes: List[List[int]], img_bgr, ocr_engine, pp_texts=None):
    """
    df_struct와 <td> 순서의 cell_bboxes를 이용해 셀별로 OCR하여 df를 채운다.
    <td> 순서와 DF 셀 수가 다르면 가능한 범위까지만 채우고 나머지는 PP 텍스트로 보완.
    """
    h, w = img_bgr.shape[:2]
    rows, cols = df_struct.shape
    total_cells = rows * cols
    texts = []

    for i, box in enumerate(cell_bboxes[:total_cells]):
        x1,y1,x2,y2 = map(int, box)
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
        crop = img_bgr[y1:y2, x1:x2]
        try:
            txt = ocr_engine.ocr(crop)
        except Exception:
            txt = ""
        texts.append(norm_ws(txt))

    # 부족/과잉 분기
    if len(texts) < total_cells:
        # 부족분은 PP 텍스트로 보완 (가능하면)
        remain = total_cells - len(texts)
        if pp_texts and isinstance(pp_texts, list):
            extra = [norm_ws(t) for t in pp_texts[len(texts):len(texts)+remain]]
            texts.extend(extra)
        else:
            texts.extend([""]*remain)
    else:
        texts = texts[:total_cells]

    # DF 채우기
    filled = df_struct.copy()
    it = iter(texts)
    for r in range(rows):
        vals=[]
        for c in range(cols):
            vals.append(next(it))
        filled.iloc[r,:] = vals
    return filled

# ===================== 평가(문자 정확도/셀 완전일치) =====================

def evaluate_tables(gt_df: pd.DataFrame, pred_df: pd.DataFrame, key_col="공종"):
    hmap = map_headers(list(pred_df.columns), list(gt_df.columns))
    rmap = map_rows(pred_df, gt_df, key_col=key_col)

    char_accs, exact_all, exact_num = [], [], []
    for gi, pj in rmap.items():
        if pj >= len(pred_df): continue
        gt_row, pr_row = gt_df.iloc[gi], pred_df.iloc[pj]
        for pcol, gcol in hmap.items():
            if gcol not in gt_df.columns: continue
            gt_val = norm_ws(gt_row.get(gcol,""))
            pr_val = norm_ws(pr_row.get(pcol,""))

            ca = levenshtein_acc(pr_val, gt_val)
            if not math.isnan(ca): char_accs.append(ca)

            gnum, pnum = norm_numeric(gt_val), norm_numeric(pr_val)
            if gnum is not None:
                ok = (pnum == gnum)
                exact_num.append(int(ok)); exact_all.append(int(ok))
            else:
                exact_all.append(int(pr_val == gt_val))

    summary = {
        "cells_compared": len(char_accs),
        "char_acc_mean": float(np.mean(char_accs)) if char_accs else float('nan'),
        "cell_exact_all": float(np.mean(exact_all)) if exact_all else float('nan'),
        "cell_exact_numeric": float(np.mean(exact_num)) if exact_num else float('nan'),
    }
    return summary

# ===================== 배치 러너 =====================

def pair_json_image(img_dir: str, json_dir: str) -> List[Tuple[str,str]]:
    exts = {".png",".jpg",".jpeg",".tif",".tiff",".bmp"}
    imgs = {Path(p).stem: p for p in glob.glob(os.path.join(img_dir, "*")) if Path(p).suffix.lower() in exts}
    pairs=[]
    for jp in glob.glob(os.path.join(json_dir, "*.json")):
        st = Path(jp).stem
        if st in imgs:
            pairs.append((imgs[st], jp))
        else:
            # stem이 다르면 skip (필요 시 커스텀 규칙 추가)
            pass
    return sorted(pairs, key=lambda x: x[0])

def run_batch(img_dir: str, json_dir: str, out_dir: str, key_col="공종", pp_lang="en", use_gpu=False):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pairs = pair_json_image(img_dir, json_dir)
    if not pairs:
        print("이미지-JSON 쌍을 찾지 못했습니다."); return

    # 미리 구조 추출(이미지마다 1회)
    cache_struct = {}
    for ip, jp in tqdm(pairs, desc="PP-Structure (표 구조 추출)", ncols=100):
        try:
            html, cell_bbox, pp_cells, img_bgr = ppstruct_table(ip, pp_lang=pp_lang, use_gpu=use_gpu)
            df_struct = html_to_df(html)
            cache_struct[ip] = (df_struct, cell_bbox, pp_cells, img_bgr)
        except Exception as e:
            cache_struct[ip] = (None, None, None, None)
            print(f"[warn] 구조 추출 실패: {ip} -> {e}")

    models = ["tesseract","easyocr","paddleocr"]
    all_rows=[]
    per_model_acc = {m:[] for m in models}

    for model_name in models:
        # OCR 엔진 초기화 (실패 시 skip)
        try:
            ocr_engine = OCR_FACTORIES[model_name]()
        except Exception as e:
            print(f"[skip] {model_name} 초기화 실패: {e}")
            continue

        out_csv = Path(out_dir)/f"results_{model_name}.csv"
        rows=[]
        for ip, jp in tqdm(pairs, desc=f"OCR={model_name}", ncols=100):
            gt_df = load_gt_table(jp)
            df_struct, cell_bbox, pp_cells, img_bgr = cache_struct.get(ip, (None,None,None,None))
            if df_struct is None:
                rows.append({"image":Path(ip).name,"model":model_name,
                             "cells":0,"char_acc_mean":np.nan,
                             "cell_exact_all":np.nan,"cell_exact_numeric":np.nan,
                             "error":"structure_failed"})
                continue
            try:
                pred_df = fill_df_with_cell_ocr(df_struct, cell_bbox, img_bgr, ocr_engine, pp_texts=pp_cells)
                summary = evaluate_tables(gt_df, pred_df, key_col=key_col)
                rows.append({
                    "image": Path(ip).name, "model": model_name,
                    "cells": summary["cells_compared"],
                    "char_acc_mean": summary["char_acc_mean"],
                    "cell_exact_all": summary["cell_exact_all"],
                    "cell_exact_numeric": summary["cell_exact_numeric"],
                    "error": ""
                })
                per_model_acc[model_name].append(summary["char_acc_mean"])
            except Exception as e:
                rows.append({"image":Path(ip).name,"model":model_name,
                             "cells":0,"char_acc_mean":np.nan,
                             "cell_exact_all":np.nan,"cell_exact_numeric":np.nan,
                             "error":str(e)})
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    # 요약 저장
    summary=[]
    for m in models:
        accs = [v for v in per_model_acc.get(m,[]) if not (isinstance(v,float) and math.isnan(v))]
        summary.append({
            "model": m,
            "images_evaluated": len(accs),
            "char_acc_mean_over_images": float(np.mean(accs)) if accs else float('nan')
        })
    pd.DataFrame(summary).to_csv(Path(out_dir)/"summary_models.csv", index=False)
    print(f"\n저장 완료: {out_dir}/results_*.csv , {out_dir}/summary_models.csv")

# ===================== CLI =====================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--out_dir", default="runs_table_ocr")
    ap.add_argument("--pp_lang", default="en", help="PP-Structure 레이아웃 언어(en/ch)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--key_col", default="공종")
    args = ap.parse_args()
    run_batch(args.img_dir, args.json_dir, args.out_dir,
              key_col=args.key_col, pp_lang=args.pp_lang, use_gpu=args.gpu)