import os, json, glob, math, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from PIL import Image
from PIL import Image
import re, glob
from tqdm import tqdm

IMG_EXTS = (".png",".jpg",".jpeg",".tif",".tiff",".bmp")

def build_img_index(img_dir):
    """
    img_dir 하위 모든 이미지 파일을 인덱싱:
    - by_basename: {basename: [fullpaths]}
    - all_paths: 전체 경로 리스트
    """
    by_base = {}
    all_paths = []
    for p in glob.glob(os.path.join(img_dir, "**", "*"), recursive=True):
        if p.lower().endswith(IMG_EXTS) and os.path.isfile(p):
            b = os.path.basename(p)
            by_base.setdefault(b, []).append(p)
            all_paths.append(p)
    return by_base, all_paths

UUID_PREFIX = re.compile(r"^[0-9a-f]{8,}-", re.I)

def strip_uuid_prefix(basename: str) -> str:
    """'7e1d99b9-ST_0028_0010285.png' -> 'ST_0028_0010285.png'"""
    return UUID_PREFIX.sub("", basename.split("/", 1)[-1])

def pick_by_size(cands, W, H):
    """여러 후보가 있으면 (W,H) 일치하는 것을 우선 선택"""
    good=[]
    for p in cands:
        try:
            with Image.open(p) as im:
                if (im.width, im.height) == (W, H):
                    good.append(p)
        except:
            pass
    return good[0] if good else (cands[0] if cands else None)

def find_image_path(img_key, img_index, all_paths, W, H):
    """
    JSON이 주는 파일명과 실제 파일명이 다를 때 견고하게 찾기:
    1) 완전 동일 basename
    2) UUID 접두사 제거한 basename
    3) 전체 경로에서 '…/<basename without uuid>'로 끝나는 파일
    4) 위 후보들 중 (W,H) 일치하는 것 우선
    """
    base = os.path.basename(img_key)
    by_base, _ = img_index

    # 1) 그대로 찾기
    cands = by_base.get(base, [])

    # 2) UUID 접두사 제거한 이름으로 재시도
    alt = strip_uuid_prefix(base)
    if not cands and alt != base:
        cands = by_base.get(alt, [])

    # 3) suffix 검색 (하위폴더 위치 달라도 매칭)
    if not cands:
        suffix = alt
        cands = [p for p in all_paths if os.path.basename(p) == suffix]

    # 4) 크기로 disambiguation
    return pick_by_size(cands, W, H)

def norm_label(lbl: str) -> str:
    if not lbl:
        return ""
    m = str(lbl).strip().lower()
    mapping = {
        "text":"Text", "title":"Title",
        "table":"Table", "figure":"Figure",
        # 방어적으로 아래도 테이블로 묶어줌(버전에 따라 나올 수 있음)
        "table_caption":"Table", "cell":"Table"
    }
    return mapping.get(m, m.title())

# === OCR 엔진 어댑터들 ===
class BaseOCREngine:
    name = "base"
    def ocr(self, img) -> str:
        raise NotImplementedError

# class TesseractOCR(BaseOCREngine):
#     name = "tesseract"
#     def __init__(self):
#         import pytesseract
#         self.t = pytesseract
#     def ocr(self, img) -> str:
#         return self.t.image_to_string(img)
class TesseractOCR(BaseOCREngine):
    name = "tesseract"
    def __init__(self, lang="kor+eng", psm=6, oem=1):
        import pytesseract, cv2
        self.t = pytesseract
        self.cv2 = cv2
        self.lang = lang
        self.config = f"--oem {oem} --psm {psm}"
        # 바이너리 확인(없으면 예외 발생)
        _ = self.t.get_tesseract_version()

    def ocr(self, img) -> str:
        arr = self.cv2.cvtColor(np.array(img), self.cv2.COLOR_RGB2BGR)
        return self.t.image_to_string(arr, lang=self.lang, config=self.config)

# class EasyOCREngine(BaseOCREngine):
#     name = "easyocr"
#     def __init__(self, lang_list=None):
#         import easyocr
#         self.reader = easyocr.Reader(lang_list or ['ko','en'], gpu=True)
#     def ocr(self, img) -> str:
#         # concat line-wise text
#         res = self.reader.readtext(np.array(img), detail=0, paragraph=True)
#         return "\n".join(res)
class EasyOCREngine(BaseOCREngine):
    name = "easyocr"
    def __init__(self, lang_list=None):
        import easyocr
        self.reader = easyocr.Reader(lang_list or ['ko','en'], gpu=True)
    def ocr(self, img) -> str:
        # detail=0 옵션 대신 기본 반환 형식인 (바운딩 박스, 텍스트, 신뢰도)를 처리합니다.
        res = self.reader.readtext(np.array(img))
        
        # 'res'는 [(bbox, text, conf), ...] 형식의 튜플 리스트입니다.
        # 이 리스트에서 텍스트만 추출하여 새로운 리스트를 만듭니다.
        texts = [text for (bbox, text, conf) in res]
        
        # 추출된 텍스트 리스트를 한 줄씩 합쳐 하나의 문자열로 반환합니다.
        return " ".join(texts)    


class PaddleOCREngine(BaseOCREngine):
    name = "paddleocr"
    def __init__(self, lang='korean'):
        from paddleocr import PaddleOCR
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False, use_gpu=True)
    def ocr(self, img) -> str:
        res = self.ocr_engine.ocr(np.array(img), cls=True)
        lines = []
        for page in res:
            for _, (text, conf) in page:
                lines.append(text)
        return "\n".join(lines)
# class PaddleOCREngine(BaseOCREngine):
#     name = "paddleocr"
#     def __init__(self, lang='korean', use_gpu=False, rec_batch_num=16, cpu_threads=8, use_cls=True):
#         from paddleocr import PaddleOCR
#         # 탐지(det) 비활성화, 인식(rec)만 수행
#         self.ocr_engine = PaddleOCR(
#             lang=lang,
#             det=False,         # ★ 탐지 끔
#             rec=True,
#             cls=use_cls,       # 방향/기울기 분류기 (느리면 False로)
#             use_angle_cls=use_cls,
#             use_gpu=use_gpu,   # GPU 세팅돼 있으면 True
#             rec_batch_num=rec_batch_num,  # 배치 크기(속도 ↑)
#             cpu_threads=cpu_threads,
#             show_log=False
#         )

#     def ocr(self, img) -> str:
#         # 크롭 1장을 인식만 수행
#         arr = np.array(img)
#         res = self.ocr_engine.ocr(arr, det=False, rec=True, cls=True)
#         # det=False일 때 결과는 [[text, score]] 형태 (여러 줄이면 여러 항목)
#         if not res:
#             return ""
#         if isinstance(res[0], list) and len(res[0]) == 2 and isinstance(res[0][0], str):
#             # 한 줄만
#             return res[0][0]
#         # 여러 줄이면 텍스트만 합침
#         return " ".join([r[0] if isinstance(r, list) else str(r) for r in res])

#     # 선택: 여러 크롭을 한 번에 배치 처리 (run_combo에서 사용 가능)
#     def ocr_batch(self, imgs) -> list[str]:
#         arrs = [np.array(im) for im in imgs]
#         results = self.ocr_engine.ocr(arrs, det=False, rec=True, cls=True)
#         outs = []
#         for res in results:
#             if not res:
#                 outs.append("")
#             elif isinstance(res[0], list) and len(res[0]) == 2 and isinstance(res[0][0], str):
#                 outs.append(res[0][0])
#             else:
#                 outs.append(" ".join([r[0] for r in res]))
#         return outs


# === 레이아웃(테이블/피겨) 검출 어댑터들 ===
class BaseLayoutDetector:
    name = "layout"
    # return list of (x1,y1,x2,y2,label,score)
    def detect(self, img: Image.Image) -> List[Tuple[float,float,float,float,str,float]]:
        raise NotImplementedError

def _lp_detectron_model(config_path, model_path, label_map):
    import layoutparser as lp
    return lp.Detectron2LayoutModel(
        config_path, model_path, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5], label_map=label_map
    )

class Detectron2PubLayNet(BaseLayoutDetector):
    name = "d2_publaynet"
    def __init__(self):
        # layoutparser 모델 체크포인트 경로는 내부적으로 자동 다운로드됨
        self.label_map = {0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"}
        self.model = _lp_detectron_model(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            "src/dectectors/PubLayNet/model_final.pth",
            self.label_map
        )
    def detect(self, img):
        layout = self.model.detect(np.array(img))
        out=[]
        for l in layout:
            x1,y1,x2,y2 = l.block.x_1, l.block.y_1, l.block.x_2, l.block.y_2
            out.append((x1,y1,x2,y2,l.type, float(l.score) if hasattr(l,'score') and l.score else 1.0))
        return out

class LayoutParserPRIMA(BaseLayoutDetector):
    name = "lp_prima"
    def __init__(self):
        # PRIMA Layout (문서 레이아웃) - 클래스명이 조금 다름
        self.label_map = {0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"}
        self.model = _lp_detectron_model(
            "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
            "src/dectectors/PrimaLayout/model_final.pth",
            self.label_map
        )
    def detect(self, img):
        layout = self.model.detect(np.array(img))
        out=[]
        for l in layout:
            x1,y1,x2,y2 = l.block.x_1, l.block.y_1, l.block.x_2, l.block.y_2
            out.append((x1,y1,x2,y2,l.type, float(l.score) if hasattr(l,'score') and l.score else 1.0))
        return out

class PPStructureDetector(BaseLayoutDetector):
    name = "pp_structure"
    def __init__(self):
        try:
            from paddleocr import PPStructure
        except ImportError:
            from paddleocr.ppstructure import PPStructure
        self.engine = PPStructure(layout=True, ocr=False, show_log=False, use_gpu=False)

    def detect(self, img):
        import cv2, numpy as np
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = self.engine(arr)  # [{'bbox':[x1,y1,x2,y2], 'type':'table', ...}, ...]
        out=[]
        for r in result:
            x1,y1,x2,y2 = r['bbox']
            label = norm_label(r.get('type','Text'))   # ★ 표준화
            out.append((x1,y1,x2,y2,label, 1.0))
        return out

# === 보조: 좌표 변환/크롭/IoU/편집거리 ===
def pct_box_to_abs(b, W, H):  # value={x,y,width,height} with percent
    x1 = b["x"]/100.0*W
    y1 = b["y"]/100.0*H
    x2 = (b["x"]+b["width"])/100.0*W
    y2 = (b["y"]+b["height"])/100.0*H
    return [x1,y1,x2,y2]

def crop(img: Image.Image, box):
    x1,y1,x2,y2 = map(int, box)
    x1=max(0,x1); y1=max(0,y1)
    x2=min(img.width, x2); y2=min(img.height, y2)
    return img.crop((x1,y1,x2,y2))

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    inter_x1=max(ax1,bx1); inter_y1=max(ay1,by1)
    inter_x2=min(ax2,bx2); inter_y2=min(ay2,by2)
    iw=max(0, inter_x2-inter_x1); ih=max(0, inter_y2-inter_y1)
    inter=iw*ih
    if inter==0: return 0.0
    union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/union if union>0 else 0.0

def levenshtein(a,b):
    la,lb=len(a),len(b)
    dp = list(range(lb+1))
    for i,ca in enumerate(a,1):
        prev=dp[0]; dp[0]=i
        for j,cb in enumerate(b,1):
            cur=dp[j]
            if ca==cb: dp[j]=prev
            else: dp[j]=min(prev+1, dp[j]+1, dp[j-1]+1)
            prev=cur
    return dp[lb]

# === 평가: OCR char-accuracy, detection mAP@0.5 ===
def eval_ocr_char_acc(pairs: List[Tuple[str,str]]) -> float:
    # pairs: list of (pred_text, gt_text)
    if not pairs: return float('nan')
    accs=[]
    for pred, gt in pairs:
        gt = (gt or "").strip()
        pred = (pred or "").strip()
        if len(gt)==0: continue
        dist=levenshtein(pred, gt)
        acc=max(0.0, 1.0 - dist/max(1,len(gt)))
        accs.append(acc)
    return float(np.mean(accs)) if accs else float('nan')

def eval_map50(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, target_classes=("Table","Figure")):
    # very simple AP@0.5 calculation over target classes
    aps=[]
    for cls in target_classes:
        # sort preds of this class by score desc
        P=[(b,s) for b,l,s in zip(pred_boxes,pred_labels,pred_scores) if l==cls]
        P.sort(key=lambda x:x[1], reverse=True)
        G=[b for b,l in zip(gt_boxes,gt_labels) if l==cls]
        matched=[False]*len(G)
        tp=[]; fp=[]
        for b,s in P:
            best_iou=0; best_j=-1
            for j,g in enumerate(G):
                if matched[j]: continue
                v=iou(b,g)
                if v>best_iou:
                    best_iou=v; best_j=j
            if best_iou>=0.5 and best_j!=-1:
                matched[best_j]=True; tp.append(1); fp.append(0)
            else:
                tp.append(0); fp.append(1)
        tp_cum=np.cumsum(tp); fp_cum=np.cumsum(fp)
        if len(P)==0:
            aps.append(float('nan')); continue
        recalls = tp_cum / max(1,len(G))
        precisions = tp_cum / np.maximum(1,(tp_cum+fp_cum))
        # AP: 11-point interpolation
        ap=0.0
        for r in np.linspace(0,1,11):
            p = np.max(precisions[recalls>=r]) if np.any(recalls>=r) else 0.0
            ap += p/11.0
        aps.append(ap)
    return float(np.nanmean(aps)) if aps else float('nan')

# === 데이터 로더 ===
def load_label(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        d=json.load(f)
    anns = d["annotations"][0]["result"]
    W=anns[0]["original_width"]; H=anns[0]["original_height"]
    gt_boxes=[]; gt_labels=[]; gt_texts=[]
    for r in anns:
        v=r["value"]; lbl=r["value"]["rectanglelabels"][0]
        box = pct_box_to_abs(v, W, H)
        text = v.get("contents", None)
        gt_boxes.append(box); gt_labels.append(lbl); gt_texts.append(text)
    # 이미지 경로
    img_key = os.path.basename(d["data"]["image"])[9:]
    return img_key, (gt_boxes, gt_labels, gt_texts)

# === 마스킹: 검출된 Table/Figure를 검은색으로 지우기(텍스트만 남김) ===
def mask_tables_figures(img: Image.Image, dets):
    arr = np.array(img).copy()
    for x1,y1,x2,y2,label,score in dets:
        if label in ("Table","Figure"):
            x1=int(max(0,x1)); y1=int(max(0,y1))
            x2=int(min(arr.shape[1],x2)); y2=int(min(arr.shape[0],y2))
            arr[y1:y2, x1:x2] = 255  # 흰색 마스킹
    return Image.fromarray(arr)

# === 조합 실행 ===
OCR_ENGINES = {
    "tesseract": TesseractOCR,
    "easyocr":  EasyOCREngine,
    "paddleocr": PaddleOCREngine,
}
LAYOUT_DETS = {
    "d2_publaynet": Detectron2PubLayNet,
    "lp_prima":     LayoutParserPRIMA,
    "pp_structure": PPStructureDetector,
}

@dataclass
class Combo:
    ocr: str
    det: str

def run_combo(img_dir, lbl_dir, combo: Combo, img_index=None, all_paths=None):
    # lazy init (모델 로딩)
    ocr = OCR_ENGINES[combo.ocr]()
    det = LAYOUT_DETS[combo.det]()
    rows=[]
    for jp in tqdm(sorted(glob.glob(os.path.join(lbl_dir, "*.json")))):
        try:
            img_key, (gt_boxes, gt_labels, gt_texts) = load_label(jp)
            W = int(gt_boxes and (gt_boxes[0][2] or 0) or 0)  # placeholder… 필요시 load_label에서 W,H를 리턴하세요.
            H = int(gt_boxes and (gt_boxes[0][3] or 0) or 0)
            # 이미지 찾기: file_upload 또는 data/image basename 기준
            candidates = [os.path.join(img_dir, os.path.basename(img_key))]
            if not os.path.exists(candidates[0]):
                # 혹시나 json이 data/upload/... 로 저장된 경우도 대비
                base = os.path.basename(img_key)
                cand2 = glob.glob(os.path.join(img_dir, "**", base), recursive=True)
                candidates = cand2 or candidates
            if not candidates or not os.path.exists(candidates[0]):
                print(f"[warn] image not found for {jp}: {img_key}")
                continue
            img_path = find_image_path(img_key, img_index, all_paths, W, H)
            if not img_path:
                print(f"[warn] image not found for {jp}: {os.path.basename(img_key)}")
                continue
            img = Image.open(img_path).convert("RGB")

            # 레이아웃 검출
            dets = det.detect(img)
            pred_boxes=[(x1,y1,x2,y2) for (x1,y1,x2,y2,lab,sc) in dets]
            pred_labels=[lab for (_,_,_,_,lab,_) in dets]
            pred_scores=[sc for (*_,sc) in dets]

            # Detection 평가용 GT 수집
            det_gt_boxes=[]; det_gt_labels=[]
            for b,l in zip(gt_boxes, gt_labels):
                if l in ("Table","Figure"):
                    det_gt_boxes.append(b); det_gt_labels.append(l)
            map50 = eval_map50(pred_boxes, pred_labels, pred_scores, det_gt_boxes, det_gt_labels)

            # OCR: Table/Figure 마스킹 후, GT의 Text 박스만 크롭하여 평가
            masked = mask_tables_figures(img, dets)
            pairs=[]
            for b,l,t in zip(gt_boxes, gt_labels, gt_texts):
                if ((l=="Text") or (l=="Title")) and (t is not None) and str(t).strip()!="":
                    crop_img = crop(masked, b)
                    pred_text = ocr.ocr(crop_img)
                    pairs.append((pred_text, str(t)))
            char_acc = eval_ocr_char_acc(pairs)

            rows.append({
                "image": os.path.basename(img_path),
                "combo": f"{combo.ocr}+{combo.det}",
                "map50_table_figure": map50,
                "ocr_char_acc": char_acc,
                "num_text_gt": sum(1 for l,t in zip(gt_labels,gt_texts) if l=="Text" and t),
                "num_tf_gt": sum(1 for l in det_gt_labels if l in ("Table","Figure")),
            })
        except Exception as e:
            rows.append({
                "image": os.path.basename(jp),
                "combo": f"{combo.ocr}+{combo.det}",
                "map50_table_figure": np.nan,
                "ocr_char_acc": np.nan,
                "error": str(e)
            })
    df = pd.DataFrame(rows)
    return df

# def main(img_dir, lbl_dir, out_csv="src/eval/results.csv"):
#     os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#     img_index = build_img_index(img_dir) 
#     by_base, all_paths = img_index
#     combos = [Combo(o,d) for o in OCR_ENGINES for d in LAYOUT_DETS]  # 3x3
#     all_rows=[]
#     for c in tqdm(combos):
#         print(f"==> Running {c.ocr}+{c.det}")
#         df = run_combo(img_dir, lbl_dir, c, img_index=img_index, all_paths=all_paths)
#         # 조합별 집계
#         agg = df[["map50_table_figure","ocr_char_acc"]].mean(numeric_only=True).to_dict()
#         agg["combo"] = f"{c.ocr}+{c.det}"
#         agg["images"] = df.shape[0]
#         print(f"   mAP@0.5={agg['map50_table_figure']:.3f}  OCR-char-acc={agg['ocr_char_acc']:.3f}")
#         df.to_csv(out_csv.replace(".csv", f".{c.ocr}+{c.det}.csv"), index=False)
#         all_rows.append(agg)
#     pd.DataFrame(all_rows).to_csv(out_csv, index=False)
#     print(f"\nSaved: {out_csv}")

# if __name__=="__main__":
#     import argparse
#     ap=argparse.ArgumentParser()
#     ap.add_argument("--img_dir", required=True)
#     ap.add_argument("--lbl_dir", required=True)
#     ap.add_argument("--out", default="runs/results.csv")
#     args=ap.parse_args()
#     main(args.img_dir, args.lbl_dir, args.out)


# ... (전체 코드는 동일)

def main(img_dir, lbl_dir, out_csv="src/eval/results.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    img_index = build_img_index(img_dir)
    by_base, all_paths = img_index
    
    # 원하는 조합을 명시적으로 리스트에 추가합니다.
    # 예시: easyocr + d2_publaynet 조합만 실행
    combos = [Combo("tesseract", "d2_publaynet")] 
    
    # 9개 조합 중 다른 특정 조합을 실행하고 싶다면, 이 리스트를 수정하세요.
    # 예시: PaddleOCR + PPStructureDetector 조합만 실행
    # combos = [Combo("paddleocr", "pp_structure")]
    
    all_rows=[]
    for c in combos:
        print(f"==> Running {c.ocr}+{c.det}")
        df = run_combo(img_dir, lbl_dir, c, img_index=img_index, all_paths=all_paths)
        # 조합별 집계
        agg = df[["map50_table_figure","ocr_char_acc"]].mean(numeric_only=True).to_dict()
        agg["combo"] = f"{c.ocr}+{c.det}"
        agg["images"] = df.shape[0]
        print(f"   mAP@0.5={agg['map50_table_figure']:.3f}  OCR-char-acc={agg['ocr_char_acc']:.3f}")
        df.to_csv(out_csv.replace(".csv", f".{c.ocr}+{c.det}.csv"), index=False)
        all_rows.append(agg)
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--lbl_dir", required=True)
    ap.add_argument("--out", default="runs/results.csv")
    args=ap.parse_args()
    main(args.img_dir, args.lbl_dir, args.out)