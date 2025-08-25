import cv2
import numpy as np
from collections import Counter
from itertools import groupby
import pandas as pd

# ---------- PaddleOCR ----------
# pip install paddlepaddle-gpu==2.5.2.post120 -f https://www.paddlepaddle.org.cn/whl/mkl/stable.html  # (CUDA 환경일 때 예시)
# pip install "paddleocr>=2.7.0"

def table_improved(path, key=None, url=None, use_gpu=False, ocr_lang='korean', max_pixels=4096, out_csv="results/table_improved.csv", out_debug_png="debug_improved.png"):
    """
    PaddleOCR에 최적화된 개선된 테이블 추출 함수
    
    Args:
        path: 입력 이미지 경로
        key, url: 더이상 사용하지 않음(호환성 위해 남김)
        use_gpu: PaddleOCR GPU 사용 여부
        ocr_lang: PaddleOCR 언어 ('korean' 권장), 예: 'korean','en'
        max_pixels: 긴 변 최대 픽셀(리사이즈 기준)
        out_csv: 출력 CSV 파일 경로
        out_debug_png: 디버그 이미지 출력 경로
    
    Returns:
        pd.DataFrame: 추출된 테이블 데이터
    """
    print("[INFO] table_improved() start - PaddleOCR backend")

    # ---------------------------
    # 이미지 로드 & 리사이즈(좌표계 일치)
    # ---------------------------
    def load_image(image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return img

    def resize_image(image: np.ndarray, max_pixels: int = 4096) -> tuple:
        h, w, _ = image.shape
        if max(h, w) <= max_pixels:
            return image, 1.0
        ratio = max_pixels / float(max(h, w))
        resized = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        return resized, ratio

    image_orig = load_image(path)
    image, resize_ratio = resize_image(image_orig, max_pixels=max_pixels)
    print(f"[INFO] Image resized from {image_orig.shape[:2]} to {image.shape[:2]}, ratio: {resize_ratio:.3f}")

    # ---------------------------
    # 표 그리드(셀) 추출 (morph + CC)
    # ---------------------------
    def table_objection(img_bgr):
        """OpenCV 기반 테이블 셀 검출"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 223, 225, cv2.THRESH_BINARY)
        img_bin = ~img_bin  # invert

        line_min_width = 15  # 데이터에 맞춰 조정 가능
        kernal_h = np.ones((1, line_min_width), np.uint8)
        kernal_v = np.ones((line_min_width, 1), np.uint8)

        img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
        img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)

        img_bin_final = img_bin_h | img_bin_v
        img_bin_final = cv2.dilate(img_bin_final, np.ones((3, 3), np.uint8), iterations=1)

        # 셀 추정: 선이 아닌 영역 연결성 분석
        _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
        return stats

    stats = table_objection(image).tolist()
    print(f"[INFO] Detected {len(stats)-1} potential cells")

    # 디버그: 셀 박스 그려서 저장(리사이즈 기준)
    debug_vis = image.copy()
    for x, y, w, h, area in stats[1:]:  # 0번은 배경이므로 제외
        if area > 100:  # 최소 면적 필터
            cv2.rectangle(debug_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(out_debug_png, debug_vis)
    print(f"[INFO] Debug image saved to {out_debug_png}")

    # ---------------------------
    # PaddleOCR 실행
    # ---------------------------
    from paddleocr import PaddleOCR
    ocr_engine = PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False, use_gpu=use_gpu)
    
    # PaddleOCR에 리사이즈된 이미지 전달
    ocr_res = ocr_engine.ocr(image, cls=True)
    print(f"[INFO] OCR completed")

    # OCR 결과를 더 쉽게 처리할 수 있는 형태로 변환
    ocr_results = []
    if ocr_res and ocr_res[0]:
        for item in ocr_res[0]:
            bbox, (txt, conf) = item
            # bbox는 4점 리스트 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            # 직사각형 형태로 변환 (min_x, min_y, max_x, max_y)
            x_coords = [pt[0] for pt in bbox]
            y_coords = [pt[1] for pt in bbox]
            rect_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            ocr_results.append({
                "text": txt,
                "confidence": float(conf),
                "bbox": rect_bbox,  # (x1, y1, x2, y2)
                "center": ((min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2)
            })
    
    print(f"[INFO] Processed {len(ocr_results)} OCR results")

    # ---------------------------
    # 개선된 텍스트-셀 매칭 함수
    # ---------------------------
    def is_text_in_cell(text_bbox, cell_data, tolerance=3):
        """텍스트가 셀 안에 포함되는지 확인 (여유공간 고려)"""
        tx1, ty1, tx2, ty2 = text_bbox
        
        # cell_data가 리스트인 경우 [x, y, w, h, area] 형태
        if isinstance(cell_data, (list, tuple)) and len(cell_data) >= 4:
            cx, cy, cw, ch = cell_data[0], cell_data[1], cell_data[2], cell_data[3]
        else:
            print(f"[ERROR] Unexpected cell_data format: {cell_data}")
            return False
            
        cx1, cy1, cx2, cy2 = cx, cy, cx + cw, cy + ch
        
        # 텍스트의 중심점이 셀 안에 있는지 확인
        text_center_x = (tx1 + tx2) / 2
        text_center_y = (ty1 + ty2) / 2
        
        return (cx1 - tolerance <= text_center_x <= cx2 + tolerance and 
                cy1 - tolerance <= text_center_y <= cy2 + tolerance)

    def collect_text_in_cell(cell_data):
        """셀 내의 모든 텍스트를 수집하고 결합"""
        texts = []
        for ocr_item in ocr_results:
            if is_text_in_cell(ocr_item["bbox"], cell_data):
                texts.append(ocr_item["text"])
        return ' '.join(texts).strip()

    # ---------------------------
    # 개선된 셀 정렬 및 그룹화
    # ---------------------------
    def analyze_table_structure(cells):
        """테이블 구조를 분석하여 행/열 정보 추출"""
        if len(cells) < 2:
            return []
        
        # y 좌표 기준으로 행 그룹화 (tolerance: 15px)
        cells_sorted_y = sorted(cells, key=lambda x: x[1])  # y 좌표로 정렬
        
        rows = []
        current_row = [cells_sorted_y[0]]
        
        for i in range(1, len(cells_sorted_y)):
            current_y = cells_sorted_y[i][1]
            prev_y = cells_sorted_y[i-1][1]
            
            # y 좌표 차이가 15px 이하면 같은 행으로 간주
            if abs(current_y - prev_y) <= 15:
                current_row.append(cells_sorted_y[i])
            else:
                # 현재 행을 x 좌표로 정렬하여 저장
                current_row.sort(key=lambda x: x[0])
                rows.append(current_row)
                current_row = [cells_sorted_y[i]]
        
        # 마지막 행 추가
        if current_row:
            current_row.sort(key=lambda x: x[0])
            rows.append(current_row)
        
        return rows

    def filter_and_merge_cells(cells, min_area=100, min_width=20, min_height=10):
        """셀 필터링 및 병합"""
        valid_cells = []
        for cell_data in cells[1:]:  # 0번은 배경
            if len(cell_data) >= 5:  # [x, y, w, h, area] 형태 확인
                x, y, w, h, area = cell_data[0], cell_data[1], cell_data[2], cell_data[3], cell_data[4]
                if area >= min_area and w >= min_width and h >= min_height:
                    valid_cells.append(cell_data)
        return valid_cells

    def normalize_cell_data(cell_data):
        """셀 데이터를 표준 형태로 정규화"""
        if isinstance(cell_data, (list, tuple)) and len(cell_data) >= 4:
            return {
                'x': cell_data[0],
                'y': cell_data[1], 
                'w': cell_data[2],
                'h': cell_data[3],
                'area': cell_data[4] if len(cell_data) > 4 else cell_data[2] * cell_data[3]
            }
        return None

    # ---------------------------
    # 메인 테이블 처리 로직
    # ---------------------------
    # 유효한 셀만 필터링
    valid_cells = filter_and_merge_cells(stats)
    
    if len(valid_cells) < 2:
        print("[WARN] Not enough valid cells detected")
        df = pd.DataFrame()
        df.to_csv(out_csv, index=False)
        return df

    print(f"[INFO] Processing {len(valid_cells)} valid cells")
    
    # 테이블 구조 분석
    rows = analyze_table_structure(valid_cells)
    print(f"[INFO] Detected {len(rows)} rows")
    
    if not rows:
        print("[WARN] No rows detected")
        df = pd.DataFrame()
        df.to_csv(out_csv, index=False)
        return df
    
    # 각 셀의 텍스트 수집
    table_data = []
    for row_idx, row_cells in enumerate(rows):
        row_data = []
        for cell in row_cells:
            cell_text = collect_text_in_cell(cell)
            row_data.append(cell_text)
        table_data.append(row_data)
        print(f"[DEBUG] Row {row_idx}: {len(row_data)} cells")
        
        # 처음 몇 행의 데이터 미리보기
        if row_idx < 3:
            preview_data = [text[:20] + "..." if len(text) > 20 else text for text in row_data]
            print(f"[DEBUG] Row {row_idx} content preview: {preview_data}")
    
    # DataFrame 생성
    if not table_data:
        print("[WARN] No table data extracted")
        df = pd.DataFrame()
    else:
        # 컬럼 수 통일 (가장 많은 컬럼을 가진 행 기준)
        max_cols = max(len(row) for row in table_data) if table_data else 0
        
        if max_cols == 0:
            print("[WARN] All rows are empty")
            df = pd.DataFrame()
        else:
            # 모든 행을 동일한 컬럼 수로 맞춤
            normalized_table = []
            for row in table_data:
                normalized_row = row[:max_cols]  # 최대 컬럼 수로 자르기
                while len(normalized_row) < max_cols:  # 부족한 컬럼은 빈 문자열로 채움
                    normalized_row.append("")
                normalized_table.append(normalized_row)
            
            # 첫 번째 행을 헤더로 사용하거나 기본 헤더 생성
            if len(normalized_table) > 1:
                headers = normalized_table[0]
                data_rows = normalized_table[1:]
                
                # 헤더가 모두 비어있으면 기본 헤더 사용
                if all(not h.strip() for h in headers):
                    headers = [f"Column_{i+1}" for i in range(max_cols)]
                else:
                    # 빈 헤더는 기본값으로 대체
                    for i, h in enumerate(headers):
                        if not h.strip():
                            headers[i] = f"Column_{i+1}"
                
                df = pd.DataFrame(data_rows, columns=headers)
            else:
                # 데이터가 한 행뿐인 경우
                headers = [f"Column_{i+1}" for i in range(max_cols)]
                df = pd.DataFrame([normalized_table[0]], columns=headers)
    
    # 결과 저장
    if df.shape[1] == 0:
        print("[WARN] DataFrame is empty. Nothing to save.")
    else:
        df.to_csv(out_csv, index=False)
        print(f"[INFO] Saved table with {df.shape[0]} rows and {df.shape[1]} columns to {out_csv}")
        print(f"[INFO] Headers: {list(df.columns)}")
        
        # 간단한 데이터 미리보기
        if df.shape[0] > 0:
            print("\n[INFO] Data preview:")
            print(df.head(3).to_string(index=False, max_cols=6, max_colwidth=15))

    return df


# 호환성을 위한 래퍼 함수
def table(path, **kwargs):
    """기존 함수명과의 호환성을 위한 래퍼"""
    return table_improved(path, **kwargs)


if __name__ == "__main__":
    # 테스트 실행
    import os
    
    test_image = "../data/table.png"
    if os.path.exists(test_image):
        print("Testing improved table extraction...")
        result = table_improved(
            path=test_image,
            use_gpu=False,
            ocr_lang='korean',
            max_pixels=2048,
            out_csv="/home/skhong/ocr_proj/test_improved_output.csv",
            out_debug_png="/home/skhong/ocr_proj/test_improved_debug.png"
        )
        print("Test completed!")
        print(f"Result shape: {result.shape}")
        if not result.empty:
            print("Sample data:")
            print(result.head())
    else:
        print(f"Test image not found: {test_image}")
        print("Available images:")
        for f in os.listdir("/home/skhong/ocr_proj"):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {f}") 