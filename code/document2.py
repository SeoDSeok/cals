import cv2
import numpy as np
import pandas as pd
from collections import Counter
from itertools import groupby
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
import os
import Utils  
import json

def document(path):
    image_path = path
    # 임시 파일 경로 설정
    output_path = './temp_output.png'
    output_path2 = './temp_output2.png'

    # --- 1. PaddleOCR 엔진 초기화 및 전체 이미지 OCR 수행 (최초 1회) ---
    print("PaddleOCR 엔진을 초기화합니다...")
    ocr_engine = PaddleOCR(use_angle_cls=True, lang='korean')
    print("이미지 전체에 대해 OCR을 수행합니다...")
    # ocr_engine.ocr()은 결과를 이중 리스트로 반환하므로 첫 번째 요소를 사용합니다.
    full_page_ocr_result = ocr_engine.ocr(image_path, cls=True)[0]


    # --- 2. OpenCV를 이용한 테이블 감지 로직 (기존 코드 유지) ---
    def get_image_shape(image_path):
        try:
            with Image.open(image_path) as image:
                return image.size
        except Exception as e:
            print(f"Error getting image shape: {e}")
            return None

    wth, _ = get_image_shape(image_path)

    # 헤더 등 방해 요소를 가리기 위해 상단에 흰 사각형 그리기
    def draw_rectangle_on_top(image_path, output_path, coordinates):
        with Image.open(image_path) as image:
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = coordinates
            draw.rectangle([x1, y1, x2, y2], outline="white", fill="white")
            image.save(output_path)

    draw_rectangle_on_top(image_path, output_path, (0, 0, wth, 60))

    def table_objection(img_path):
        image = cv2.imread(img_path)
        if image is None: return None, []
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray_scale, 223, 225, cv2.THRESH_BINARY)
        img_bin = ~img_bin
        line_min_width = 15
        kernal_h = np.ones((1, line_min_width), np.uint8)
        kernal_v = np.ones((line_min_width, 1), np.uint8)
        img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
        img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
        img_bin_final = img_bin_h | img_bin_v
        final_kernel = np.ones((3, 3), np.uint8)
        img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)
        _, _, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
        return image, stats

    image, stats = table_objection(output_path)
    
    lst = [k for k in stats[2:] if k[4] >= 40 and k[2] >= 15]

    def find_tables_recursively(cell_list):
        tables = []
        remaining_cells = sorted(cell_list, key=lambda item: item[1]) # y좌표로 정렬
        
        while remaining_cells:
            current_table_cells = []
            if not remaining_cells: break
            
            # 첫 셀을 기준으로 테이블 시작
            current_table_cells.append(remaining_cells.pop(0))
            
            i = 0
            while i < len(remaining_cells):
                # 마지막 셀과 다음 셀의 y좌표 차이가 작으면 같은 테이블로 간주
                if abs(remaining_cells[i][1] - current_table_cells[-1][1]) < 150:
                    current_table_cells.append(remaining_cells.pop(i))
                else:
                    i += 1

            x_coords = [c[0] for c in current_table_cells]
            y_coords = [c[1] for c in current_table_cells]
            right_coords = [c[0] + c[2] for c in current_table_cells]
            bottom_coords = [c[1] + c[3] for c in current_table_cells]
            
            if not x_coords: continue

            tables.append((min(x_coords), min(y_coords), max(right_coords), max(bottom_coords)))

        return tables

    table_coordinates = find_tables_recursively(lst)
    detected_types = ['table'] * len(table_coordinates)

    # --- 3. 감지된 테이블을 가리고 이미지(Figure) 감지 ---
    def mask_areas(input_path, output_path, coordinates_to_mask):
        with Image.open(input_path) as image:
            draw = ImageDraw.Draw(image)
            for k in coordinates_to_mask:
                x1, y1, x2, y2 = k
                draw.rectangle([x1-30, y1-30, x2+15, y2+15], outline="white", fill="white")
            image.save(output_path)

    if table_coordinates:
        mask_areas(output_path, output_path2, table_coordinates)
        mrcnn_input_path = output_path2
    else:
        mrcnn_input_path = output_path
    
    print("Mask R-CNN으로 이미지를 분석합니다...")
    model = Utils.load_model()
    _, _, img_coor = Utils.extract_Figures(model, cv2.imread(mrcnn_input_path))

    # 테이블과 겹치지 않는 이미지만 추가
    if img_coor:
        img_bbox = (img_coor[0][0], img_coor[0][1], img_coor[0][2], img_coor[0][3])
        is_overlapping = False
        for tbl_bbox in table_coordinates:
            # AABB 충돌 검사
            if not (tbl_bbox[2] < img_bbox[0] or tbl_bbox[0] > img_bbox[2] or \
                    tbl_bbox[3] < img_bbox[1] or tbl_bbox[1] > img_bbox[3]):
                is_overlapping = True
                break
        if not is_overlapping:
            table_coordinates.append(img_bbox)
            detected_types.append('image')

    # --- 4. 텍스트 단락(Paragraph) 감지 ---
    # 테이블과 이미지를 제외한 영역에서 텍스트 찾기
    all_detected_bboxes = table_coordinates
    text_lines = []
    for line in full_page_ocr_result:
        box, (text, _) = line
        text_bbox = (box[0][0], box[0][1], box[2][0], box[2][1])
        is_part_of_element = False
        for elem_bbox in all_detected_bboxes:
             if not (elem_bbox[2] < text_bbox[0] or elem_bbox[0] > text_bbox[2] or \
                    elem_bbox[3] < text_bbox[1] or elem_bbox[1] > text_bbox[3]):
                is_part_of_element = True
                break
        if not is_part_of_element:
            text_lines.append(line)
            
    # 남은 텍스트 라인들을 단락으로 그룹화 (기존 boxing 로직 단순화 버전)
    text_lines.sort(key=lambda x: (x[0][0][1], x[0][0][0])) # y, x 순으로 정렬
    
    paragraphs = []
    if text_lines:
        current_paragraph_lines = [text_lines[0]]
        for i in range(1, len(text_lines)):
            prev_line_box = text_lines[i-1][0]
            curr_line_box = text_lines[i][0]
            # y좌표 차이가 크면 (예: 40px 이상) 새로운 단락으로 간주
            if (curr_line_box[0][1] - prev_line_box[3][1]) > 40:
                paragraphs.append(current_paragraph_lines)
                current_paragraph_lines = [text_lines[i]]
            else:
                current_paragraph_lines.append(text_lines[i])
        paragraphs.append(current_paragraph_lines)

    for para_lines in paragraphs:
        x_coords = [p[0][0][0] for p in para_lines] + [p[0][1][0] for p in para_lines]
        y_coords = [p[0][0][1] for p in para_lines] + [p[0][2][1] for p in para_lines]
        if not x_coords: continue
        para_bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        table_coordinates.append(para_bbox)
        detected_types.append('text')

    # --- 5. 각 요소(element)의 첫 줄을 목차(index)로 추출 ---
    def get_first_line_in_bbox(bbox, ocr_results):
        lines_in_bbox = []
        for line in ocr_results:
            box, (text, _) = line
            text_center_x = (box[0][0] + box[2][0]) / 2
            text_center_y = (box[0][1] + box[2][1]) / 2
            if bbox[0] <= text_center_x <= bbox[2] and bbox[1] <= text_center_y <= bbox[3]:
                lines_in_bbox.append((box[0][1], text)) # (y좌표, 텍스트)
        
        if not lines_in_bbox:
            return ""
        
        lines_in_bbox.sort() # y좌표 기준으로 정렬
        return lines_in_bbox[0][1] # 가장 위에 있는 텍스트 반환

    print("각 요소의 목차를 추출합니다...")
    indices = [get_first_line_in_bbox(bbox, full_page_ocr_result) for bbox in table_coordinates]

    # --- 6. 최종 JSON 생성 ---
    combined = sorted(zip(table_coordinates, detected_types, indices), key=lambda x: (x[0][1], x[0][0]))

    final_json = []
    id_counter = 1
    page_num = 1
    
    # 정렬된 데이터를 기반으로 JSON 객체 생성
    for bbox, type, index in combined:
        json_obj = {
            "page": page_num,
            "id": id_counter,
            "index": index,
            "type": type,
            "bbox": str(bbox)
        }
        final_json.append(json_obj)
        id_counter += 1
        
    print("임시 파일들을 정리합니다.")
    for temp_file in [output_path, output_path2]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
    return final_json

# --- 코드 실행 예시 ---
# 아래 경로를 실제 설계보고서 이미지 경로로 변경하세요.
image_file_path = '/home/skhong/ocr_proj/data/construct/r_image/r_image_01.png'
result = document(image_file_path)
print(json.dumps(result, indent=4, ensure_ascii=False))