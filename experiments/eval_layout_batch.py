# eval_layout_batch_px.py
import os, re, json, glob
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------
# 이미지 인덱싱/매칭
# -------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
UUID_PREFIX = re.compile(r"^[0-9a-f]{8,}-", re.I)

def strip_uuid(b: str) -> str:
    return UUID_PREFIX.sub("", b)

def build_img_index(img_dir: str):
    all_imgs = []
    for p in glob.glob(os.path.join(img_dir, "**", "*"), recursive=True):
        if os.path.isfile(p) and p.lower().endswith(IMG_EXTS):
            all_imgs.append(p)
    return all_imgs

def find_page_image_for_json(json_path: str, all_imgs: List[str]) -> str:
    """JSON의 image_file_name에서 토큰을 추출해 이미지 매칭"""
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    tokens = set()
    for it in d.get("training_data_info", {}).get("image_info", []):
        fn = it.get("image_file_name", "") or ""
        m = re.search(r"[A-Za-z]{2,}_\d{3,}_\d{3,}", fn)
        if m: tokens.add(m.group(0))
        m2 = re.search(r"\d{4,}_\d{4,}", fn)
        if m2: tokens.add(m2.group(0))

    base_json = os.path.splitext(strip_uuid(os.path.basename(json_path)))[0]
    tokens.add(base_json)

    cands = []
    for p in all_imgs:
        b = strip_uuid(os.path.basename(p))
        for t in tokens:
            if t and t in b:
                cands.append(p); break

    if cands:
        cands.sort(key=len)
        return cands[0]

    # 백업: JSON과 같은 폴더에서 같은 스템 찾기
    jd = os.path.dirname(json_path)
    for ext in IMG_EXTS:
        p = os.path.join(jd, base_json + ext)
        if os.path.exists(p): return p

    return None

# -------------------------
# GT: (x,y,w,h) [px]  →  (x1,y1,x2,y2) [px]
# -------------------------
def parse_xywh(loc) -> Tuple[float, float, float, float]:
    if isinstance(loc, (list, tuple)) and len(loc) == 4:
        return tuple(map(float, loc))
    if isinstance(loc, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", loc)
        if len(nums) >= 4:
            return tuple(map(float, nums[:4]))
    raise ValueError(f"Bad image_location: {loc!r}")

def loc_xywh_to_xyxy(loc) -> List[float]:
    x, y, w, h = parse_xywh(loc)
    return [x, y, x + w, y + h]

def image_info_to_gt(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    infos = d.get("training_data_info", {}).get("image_info", [])
    gts = []
    for it in infos:
        xyxy = loc_xywh_to_xyxy(it.get("image_location"))
        name = (it.get("image_name") or "") + " " + (it.get("image_caption") or "")
        label = "Table" if ("표" in name) else "Figure"  # 필요시 규칙 조정
        gts.append((xyxy, label))
    return gts

# -------------------------
# IoU / AP / mAP
# -------------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax2-ax1)) * max(0.0, (ay2-ay1))
    area_b = max(0.0, (bx2-bx1)) * max(0.0, (by2-by1))
    union = area_a + area_b - inter
    return inter/union if union > 0 else 0.0

def ap50(preds, gts, cls_name: str) -> float:
    P = [(b,s) for (b,l,s) in preds if l==cls_name]
    G = [b for (b,l) in gts if l==cls_name]
    if len(P) == 0 and len(G) == 0: return np.nan
    if len(P) == 0: return 0.0
    P.sort(key=lambda x:x[1], reverse=True)
    matched = [False]*len(G)
    tp, fp = [], []
    for b, s in P:
        best_iou, best_j = 0.0, -1
        for j,g in enumerate(G):
            if matched[j]: continue
            v = iou_xyxy(b, g)
            if v > best_iou: best_iou, best_j = v, j
        if best_iou >= 0.5 and best_j != -1:
            matched[best_j] = True; tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp = np.cumsum(tp); fp = np.cumsum(fp)
    recalls = tp / max(1, len(G))
    precisions = tp / np.maximum(1, tp+fp)
    ap = 0.0
    for r in np.linspace(0,1,11):
        p = np.max(precisions[recalls>=r]) if np.any(recalls>=r) else 0.0
        ap += p/11.0
    return ap

def map50(preds, gts, classes=("Table","Figure")) -> float:
    vals = [ap50(preds, gts, c) for c in classes]
    return float(np.nanmean(vals))

# -------------------------
# 디텍터 3종
# -------------------------
class BaseDet:
    def detect(self, img: Image.Image):
        raise NotImplementedError

class D2PubLayNet(BaseDet):
    def __init__(self, weight_path: str = None):
        import layoutparser as lp
        cfg = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
        if weight_path and os.path.isfile(weight_path):
            self.model = lp.Detectron2LayoutModel(cfg, weight_path,
                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                         label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"})
        else:
            self.model = lp.Detectron2LayoutModel(
                cfg, "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"}
            )
    def detect(self, img):
        import numpy as np
        layout = self.model.detect(np.array(img))
        out=[]
        for l in layout:
            b = [l.block.x_1, l.block.y_1, l.block.x_2, l.block.y_2]
            sc = float(getattr(l, "score", 1.0) or 1.0)
            out.append((b, l.type, sc))
        return out

class LPPrima(BaseDet):
    def __init__(self, weight_path: str = None):
        import layoutparser as lp
        cfg = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config"
        if weight_path and os.path.isfile(weight_path):
            self.model = lp.Detectron2LayoutModel(cfg, weight_path,
                         extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                         label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"})
        else:
            self.model = lp.Detectron2LayoutModel(
                cfg, "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/model",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
                label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"}
            )
    def detect(self, img):
        import numpy as np
        layout = self.model.detect(np.array(img))
        out=[]
        for l in layout:
            b = [l.block.x_1, l.block.y_1, l.block.x_2, l.block.y_2]
            sc = float(getattr(l, "score", 1.0) or 1.0)
            out.append((b, l.type, sc))
        return out

class PPStructureDet(BaseDet):
    def __init__(self):
        # ko 미지원 → en 고정(레이아웃만 사용)
        from paddleocr import PPStructure
        self.engine = PPStructure(layout=True, ocr=False, show_log=False, lang='en')
    def detect(self, img):
        import cv2, numpy as np
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        res = self.engine(arr)
        out=[]
        for r in res:
            if "bbox" in r and "type" in r:
                x1,y1,x2,y2 = map(float, r["bbox"])
                t = r["type"]
                if t.lower()=="table": lab="Table"
                elif t.lower() in ("image","figure","pic"): lab="Figure"
                else: lab="Text"
                out.append(([x1,y1,x2,y2], lab, 1.0))
        return out

# -------------------------
# 실행/집계
# -------------------------
@dataclass
class OneResult:
    json_file: str
    image_file: str
    model: str
    map50: float
    ap50_table: float
    ap50_figure: float
    num_gt_table: int
    num_gt_figure: int
    num_pred: int

def evaluate_folder(img_dir: str, json_dir: str,
                    pub_pth: str = None, pri_pth: str = None):
    all_imgs = build_img_index(img_dir)
    dets = {
        "d2_publaynet": D2PubLayNet(pub_pth),
        "lp_prima":     LPPrima(pri_pth),
        "pp_structure": PPStructureDet(),
    }
    json_files = sorted(glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True))
    rows: List[OneResult] = []

    for jp in tqdm(json_files, desc="Evaluate (per JSON)"):
        try:
            gts = image_info_to_gt(jp)
            if not gts: 
                continue
            img_path = find_page_image_for_json(jp, all_imgs)
            if not img_path or not os.path.exists(img_path):
                print(f"[WARN] page image not found for {os.path.basename(jp)}")
                continue
            img = Image.open(img_path).convert("RGB")
            gtT = sum(1 for _,l in gts if l=="Table")
            gtF = sum(1 for _,l in gts if l=="Figure")

            for name, det in dets.items():
                preds = det.detect(img)  # [(xyxy,label,score)]
                apT = ap50(preds, gts, "Table")
                apF = ap50(preds, gts, "Figure")
                m = float(np.nanmean([apT, apF]))
                rows.append(OneResult(
                    json_file=os.path.relpath(jp, json_dir),
                    image_file=os.path.relpath(img_path, img_dir),
                    model=name,
                    map50=m,
                    ap50_table=apT,
                    ap50_figure=apF,
                    num_gt_table=gtT,
                    num_gt_figure=gtF,
                    num_pred=len(preds),
                ))
        except Exception as e:
            print(f"[ERROR] {os.path.basename(jp)} -> {e}")
    return rows

def to_dataframe(rows):
    import pandas as pd
    return pd.DataFrame([r.__dict__ for r in rows])

def main():
    import argparse, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--pub_pth", default="src/dectectors/PubLayNet/model_final.pth")
    ap.add_argument("--pri_pth", default="src/dectectors/PrimaLayout/model_final.pth")
    ap.add_argument("--out_csv", default="layout_eval_results_px.csv")
    args = ap.parse_args()

    rows = evaluate_folder(args.img_dir, args.json_dir, args.pub_pth, args.pri_pth)
    df = to_dataframe(rows)
    if len(df):
        df.to_csv(args.out_csv, index=False)
        print("\n== Summary (mean) ==")
        print(df.groupby("model")[["map50","ap50_table","ap50_figure"]]
              .mean(numeric_only=True).to_string())
        print(f"\nSaved: {args.out_csv}")
    else:
        print("No results.")

if __name__ == "__main__":
    main()
