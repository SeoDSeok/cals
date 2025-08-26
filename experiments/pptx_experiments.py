# -*- coding: utf-8 -*-
import os, json, argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image

# -------------------------
# 유틸: 좌표/IoU/AP 계산
# -------------------------
def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    return [float(x), float(y), float(x + w), float(y + h)]

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def ap_at_05(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]], cls_name="Figure"):
    """
    preds: [{image_id, bbox:[x1,y1,x2,y2], score, label}]
    gts  : [{image_id, bbox:[x1,y1,x2,y2], label}]
    """
    # 해당 클래스만
    P = [p for p in preds if p["label"] == cls_name]
    G = [g for g in gts if g["label"] == cls_name]

    # 이미지별 GT 매칭 플래그
    gt_by_img: Dict[Any, List[Dict[str, Any]]] = {}
    for g in G:
        gt_by_img.setdefault(g["image_id"], []).append({"bbox": g["bbox"], "matched": False})

    if len(G) == 0:
        return np.nan

    # 점수 순 정렬
    P.sort(key=lambda x: x["score"], reverse=True)
    tp, fp = [], []

    for p in P:
        best_iou, best_j = 0.0, -1
        gt_list = gt_by_img.get(p["image_id"], [])
        for j, gt in enumerate(gt_list):
            if gt["matched"]:
                continue
            v = iou_xyxy(p["bbox"], gt["bbox"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j != -1 and best_iou >= 0.5:
            gt_list[best_j]["matched"] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / float(len(G))
    precisions = tp_cum / np.maximum(1, (tp_cum + fp_cum))

    # 11-point interpolation AP
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p_at_r = np.max(precisions[recalls >= r]) if np.any(recalls >= r) else 0.0
        ap += p_at_r / 11.0
    return float(ap)

# -------------------------
# EMU <-> PX 변환 & GT 파서
# -------------------------
import re

EMU_PER_INCH = 914400.0

def _parse_xywh(loc) -> Tuple[float, float, float, float]:
    """loc이 [x,y,w,h] 리스트이거나 '[x,y,w,h]' 문자열이어도 처리"""
    if isinstance(loc, (list, tuple)) and len(loc) == 4:
        x, y, w, h = loc
        return float(x), float(y), float(w), float(h)
    if isinstance(loc, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", loc)
        if len(nums) >= 4:
            x, y, w, h = map(float, nums[:4])
            return x, y, w, h
    raise ValueError(f"Unsupported image_location format: {loc!r}")

def _emu_to_px(v: float, dpi: float) -> float:
    return v * (dpi / EMU_PER_INCH)

def _maybe_convert_xywh_to_px(x: float, y: float, w: float, h: float,
                              coord_unit: str = "auto", dpi: float = 96.0):
    """
    coord_unit:
      - 'emu'  : 무조건 EMU→px 변환
      - 'px'   : 변환하지 않음
      - 'auto' : 값이 매우 크면(>= 50,000) EMU로 간주
    """
    if coord_unit not in ("auto", "emu", "px"):
        coord_unit = "auto"

    is_emu = (coord_unit == "emu")
    if coord_unit == "auto":
        # EMU 값은 보통 수십만~백만 단위. 임계치를 넉넉히 둠.
        is_emu = max(x, y, w, h) >= 50000.0

    if is_emu:
        x, y, w, h = (_emu_to_px(x, dpi),
                      _emu_to_px(y, dpi),
                      _emu_to_px(w, dpi),
                      _emu_to_px(h, dpi))
    return x, y, w, h

def load_gt_from_image_info(json_path: str, image_id: str = "page1",
                            coord_unit: str = "auto", dpi: float = 96.0):
    """
    image_info[].image_location 을 GT(Figure)로 사용.
    - coord_unit: 'emu' | 'px' | 'auto'
    - dpi: PowerPoint PNG export DPI (기본 96)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    infos = d["training_data_info"]["image_info"]
    gts = []
    for it in infos:
        x, y, w, h = _parse_xywh(it["image_location"])
        x, y, w, h = _maybe_convert_xywh_to_px(x, y, w, h, coord_unit=coord_unit, dpi=dpi)
        xyxy = [float(x), float(y), float(x + w), float(y + h)]
        gts.append({"image_id": image_id, "bbox": xyxy, "label": "Figure"})
    return gts

# -------------------------
# 3개 레이아웃 검출기
# -------------------------
class BaseDetector:
    name = "base"
    def detect(self, img: Image.Image) -> List[Tuple[List[float], str, float]]:
        """return list of (bbox[x1,y1,x2,y2], label, score)"""
        raise NotImplementedError

def _lp_model(config_path, model_path, label_map):
    import layoutparser as lp
    return lp.Detectron2LayoutModel(
        config_path, model_path,
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        label_map=label_map
    )

class Detectron2PubLayNet(BaseDetector):
    name = "d2_publaynet"
    def __init__(self, weights_path: str):
        self.label_map = {0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"}
        self.model = _lp_model(
            "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            weights_path,
            self.label_map
        )
    def detect(self, img):
        layout = self.model.detect(np.array(img))
        out = []
        for l in layout:
            bbox = [float(l.block.x_1), float(l.block.y_1),
                    float(l.block.x_2), float(l.block.y_2)]
            lab  = l.type
            score= float(getattr(l, "score", 1.0) or 1.0)
            out.append((bbox, lab, score))
        return out

class LayoutParserPRIMA(BaseDetector):
    name = "lp_prima"
    def __init__(self, weights_path: str):
        self.label_map = {0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"}
        self.model = _lp_model(
            "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config",
            weights_path,
            self.label_map
        )
    def detect(self, img):
        layout = self.model.detect(np.array(img))
        out = []
        for l in layout:
            bbox = [float(l.block.x_1), float(l.block.y_1),
                    float(l.block.x_2), float(l.block.y_2)]
            lab  = l.type
            score= float(getattr(l, "score", 1.0) or 1.0)
            out.append((bbox, lab, score))
        return out

class PPStructureDetector(BaseDetector):
    name = "pp_structure"
    def __init__(self):
        # 언어는 레이아웃에 영향 없음(ocr=False). lang 지정 안 함.
        from paddleocr import PPStructure
        self.engine = PPStructure(layout=True, ocr=False, show_log=False)
    def detect(self, img):
        import cv2, numpy as np
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = self.engine(arr)  # [{'bbox':[x1,y1,x2,y2], 'type':'figure'|...}, ...]
        out=[]
        for r in result:
            x1,y1,x2,y2 = [float(v) for v in r["bbox"]]
            typ = r.get("type","text")
            # paddle은 소문자라서 통일
            label = {"figure":"Figure", "table":"Table", "text":"Text",
                     "title":"Title","list":"List"}.get(typ.lower(), typ.title())
            out.append(([x1,y1,x2,y2], label, 1.0))
        return out

# -------------------------
# 평가 파이프라인
# -------------------------
def evaluate_models(page_img_path: str, ann_json: str,
                    pub_pth: str = None, pri_pth: str = None , coord_unit: str = "auto", dpi: float = 96.0):
    img = Image.open(page_img_path).convert("RGB")
    gts = load_gt_from_image_info(
        ann_json,
        image_id=os.path.basename(page_img_path),
        coord_unit=coord_unit,
        dpi=dpi
    )

    detectors = []
    if pub_pth and os.path.isfile(pub_pth):
        try:
            detectors.append(Detectron2PubLayNet(pub_pth))
        except Exception as e:
            print(f"[warn] PubLayNet init failed: {e}")
    else:
        print("[info] skip PubLayNet (weights not found)")
    if pri_pth and os.path.isfile(pri_pth):
        try:
            detectors.append(LayoutParserPRIMA(pri_pth))
        except Exception as e:
            print(f"[warn] PRIMA init failed: {e}")
    else:
        print("[info] skip PRIMA (weights not found)")
    try:
        detectors.append(PPStructureDetector())
    except Exception as e:
        print(f"[warn] PP-Structure init failed: {e}")

    if not detectors:
        raise RuntimeError("No detector available.")

    print(f"\nGT Figure boxes: {len(gts)}")
    for det in detectors:
        preds = []
        try:
            dets = det.detect(img)
        except Exception as e:
            print(f"[{det.name}] detect failed: {e}")
            continue
        # Figure 클래스만 수집
        for bbox, lab, score in dets:
            if lab == "Figure":
                preds.append({
                    "image_id": os.path.basename(page_img_path),
                    "bbox": bbox, "score": float(score), "label": "Figure"
                })
        ap = ap_at_05(preds, gts, cls_name="Figure")
        num_p = len(preds)
        print(f"[{det.name}] preds(Figure)={num_p}  AP@0.5={ap:.3f}" if not np.isnan(ap)
              else f"[{det.name}] preds(Figure)={num_p}  AP@0.5=NaN (GT=0?)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page_img", required=True, help="페이지 이미지 경로(원본 크기)")
    ap.add_argument("--ann_json", required=True, help="질문에서 준 JSON 라벨 파일")
    ap.add_argument("--pub_pth", default="src/dectectors/PubLayNet/model_final.pth", help="PubLayNet model_final.pth")
    ap.add_argument("--pri_pth", default="src/dectectors/PrimaLayout/model_final.pth", help="PRIMA model_final.pth")
    ap.add_argument("--coord_unit", default="auto", choices=["auto", "emu", "px"],
                help="GT image_location 좌표 단위")
    ap.add_argument("--dpi", type=float, default=96.0,
                help="EMU→px 변환 시 사용할 PNG DPI (기본 96)")
    args = ap.parse_args()
    evaluate_models(args.page_img, args.ann_json, args.pub_pth, args.pri_pth,
                coord_unit=args.coord_unit, dpi=args.dpi)

if __name__ == "__main__":
    main()
