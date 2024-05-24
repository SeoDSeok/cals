import cv2
import numpy as np
import requests
import pandas as pd
import Utils
from PIL import Image, ImageDraw

def document(path, key, url):
  image_path = path
  api_key = key
  url = url
  output_path = '/output.png'

  def get_image_shape(image_path):
    try:
        # 이미지 열기
        image = Image.open(image_path)
        # 이미지의 너비와 높이 추출
        width, height = image.size
        # 이미지 닫기
        image.close()
        return width, height
    except Exception as e:
        print(f"Error: {e}")
        return None


  wth, _ = get_image_shape(image_path)

  def draw_rectangle_first(image_path, output_path, coordinates):
      # 이미지 불러오기
      image = Image.open(image_path)

      # ImageDraw 객체 생성
      draw = ImageDraw.Draw(image)

      # 좌표에서 (x1, y1), (x2, y2) 추출
      x1, y1, x2, y2 = coordinates

      # 하얀색 직사각형 그리기
      draw.rectangle([x1, y1, x2, y2], outline="white", fill="white")

      # 결과 이미지 저장
      image.save(output_path)

  first_coor = (0, 0, wth, 60)
  draw_rectangle_first(image_path, output_path, first_coor)

  #type을 저장할 리스트, bbox 좌표 저장할 리스트
  type = []
  coordinate = []
  def table_objection(out_path):
    image = cv2.imread(output_path)
    gray_scale = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,img_bin = cv2.threshold(gray_scale,223,225,cv2.THRESH_BINARY)
    img_bin = ~img_bin

    line_min_width = 15
    kernal_h = np.ones((1,line_min_width),np.uint8)
    kernal_v = np.ones((line_min_width,1),np.uint8)

    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)

    img_bin_final = img_bin_h|img_bin_v

    final_kernel = np.ones((3,3),np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=1)

    _,labels,stats,_ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8,ltype=cv2.CV_32S)
    return image, stats

  image, stats = table_objection(output_path)

  #stats: cell좌표 2부터 시작
  for x,y,w,h,area in stats[2:]:
      cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
  cv2.imwrite("final_img.png",image)

  # ㅁ, ㅂ과 같은 좌표 제거
  lst = []
  for k in stats[2:]:
      if k[4] < 40:
          continue
      if k[2] < 15:
          continue
      lst.append(k)

  # 표의 bbox 좌표 추출하는 함수 
  def table_coor(lst):
      if len(lst) <= 1:
          return (), []
      x_min = 10000000; y_min = 10000000; x_max = 0; y_max = 0; w = 0; h = 0
      for i, v in enumerate(lst):
          if i < len(lst) - 1:
              if lst[i + 1][1] - lst[i][1] < 150:
                  if lst[i][0] < x_min:
                      x_min = lst[i][0]
                  if lst[i][1] < y_min:
                      y_min = lst[i][1]
                  if lst[i][0] > x_max:
                      x_max = lst[i][0]
                      w = lst[i][2]
                  if lst[i][1] > y_max:
                      y_max = lst[i][1]
                      h = lst[i][3]
                  if len(lst[i+1:]) <= 1:
                      return (x_min, y_min, x_max + w, y_max + h), []
              else:
                  return (x_min, y_min, x_max + w, y_max + h), lst[i+1:]
  #표가 여러개 탐지하기 위한 코드
  coor, new_lst = table_coor(lst)
  coordinate.append(coor)
  while len(new_lst) > 1:
      coor2, new_lst = table_coor(new_lst)
      coordinate.append(coor2)
  # 위에 표 탐지에서 표가 탐지 안됐을 경우 : coordinate에 ()가 append되는 경우에 대해서 제외
  ct = 0
  for k in coordinate:
    if len(k) == 0:
      ct += 1
    elif k[0] == 10000000:
      ct += 1
  for k in range(ct):
    coordinate.pop(0)
  for i in range(len(coordinate)):
    type.append('table')
  coordinate, type
  def draw_rectangle(image_path, output_path, coordinates):
    # 이미지 불러오기
    image = Image.open(image_path)
    # ImageDraw 객체 생성
    draw = ImageDraw.Draw(image)
    # 좌표에서 (x1, y1), (x2, y2) 추출
    for k in coordinates:
        x1, y1, x2, y2 = k

        # 하얀색 직사각형 그리기
        draw.rectangle([x1-30, y1-30, x2+15, y2+15], outline="white", fill="white")

        # 결과 이미지 저장
        image.save(output_path)

  if len(coordinate) == 0:
      out_path = './output.png'
  else:
      out_path = './output2.png'
      draw_rectangle(output_path, out_path, coordinate)

  def mrcnn(out_path):
    image = cv2.imread(out_path)

    def load_image(image_path: str) -> np.ndarray:
        return cv2.imread(image_path)
    image = load_image(out_path)

    model = Utils.load_model()
    extract_imgs,img, coor = Utils.extract_Figures(model, image)
    return extract_imgs, img, coor

  _, img, coor = mrcnn(out_path)

  for k in coordinate:
    if len(coor) == 0:
      break
    if (k[0] > coor[0][0]) &  (k[1] > coor[0][1]) & (k[2] < coor[0][2]) & (k[3] < coor[0][3]) :
      break
    else:
      coordinate.append((coor[0][0], coor[0][1], coor[0][2], coor[0][3]))
      type.append('image')

  n = len(coordinate)

  def load_image(image_path: str) -> np.ndarray:
    return cv2.imread(image_path)

  def resize_image(image: np.ndarray, max_pixels: int = 4096) -> np.ndarray:
      height, width, _ = image.shape
      resize_ratio = 1

      if height > max_pixels or width > max_pixels:
          resize_ratio = max_pixels / max(height, width)
          image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio)

      return image, resize_ratio
  def ocr(image: np.ndarray, url: str, api_key: str) -> dict:
      image = cv2.imencode(".png", image)[1]
      image = image.tobytes()

      headers = {"x-api-key": api_key}

      files = {"image": image}

      response = requests.post(url=url, headers=headers, files=files)

      return response.json()

  image = np.array(img)
  resized_image, resize_ratio = resize_image(image)
  response = ocr(resized_image, url, api_key)

  def boxing(coordinate, type):
    # 결과를 저장할 변수 초기화
    extracted_text = ""
    tmp = []
    tmp2 = []
    # 'results' 키의 값에서 'recognized_word'와 'box'를 가져와서 문장을 만듦
    results = response['responses'][0]['results']
    tmp.append(results[0]['box'])
    for i in range(len(results)):
        result = results[i]
        # 리스트의 첫 번째 문자열을 가져오고, 없으면 빈 문자열 사용
        recognized_word = result['recognized_word'][0] if result['recognized_word'] else ''  
        box = result['box']

        if recognized_word:
            extracted_text += recognized_word + ' '
            
            # 다음 'box'가 존재하고, 현재 'box'의 윗부분의 Y좌표와 다음 'box'의 윗부분의 Y 좌표가 차이가 20이 나면 줄 바꿈
            if i + 1 < len(results) and abs(box[0][1] - results[i + 1]['box'][0][1]) > 20:
                extracted_text += '\n'
                tmp.append(results[i + 1]['box'])
                tmp2.append(box)
    # tmp는 tmp2보다 한개더 많은 쪽수가 나오기 때문에
    if len(tmp) == len(tmp2):
      pass
    else:
      tmp.pop()

    # tmp와 tmp2의 길이가 같은므로 zip으로 묶음
    k = list(zip(tmp, tmp2))
    width = 0

    criteria = min(k)[0][0][0]

    # 가장작은 x 좌표 기준 가장 큰 x좌표 차이를 width로 설정
    for i in range(len(k)):
        diff = k[i][1][1][0] - criteria
        if diff >= width:
            width = diff

    # 결과를 저장할 리스트 f : 들여쓰기가 되었을 경우, f_2 : 내어쓰기가 되었을 경우
    f = []
    f_2 = []

    # 반복문을 통해 조건에 맞는 튜플을 찾음
    for i in range(len(k)-1):
        # 현재 튜플과 다음 튜플의 첫 번째 원소를 비교
        diff_x = k[i+1][0][0][0] - k[i][0][0][0]
        diff_y = k[i+1][1][0][1] - k[i][0][0][1]
        # 첫 번째 원소의 차이가 30 -> 23 이상이면 매핑
        if (diff_x > 0 and diff_y >= 95) or diff_x >= 23:
            mapped_pair = (k[i][0], k[i - 1][1])
            f.append(mapped_pair)
        elif diff_x < -30:
            f_2.append((k[i][0], k[len(k)-1][1]))
            
        if i == len(k)-2 and diff_x < 0:
            mapped_pair = (k[i+1][0], k[i][1])
            f.append(mapped_pair)

    # first : 좌상단 좌표 저장, second : 우상단 좌표 저장
    first = []
    second = []
    for e in f:
        e1, e2 = e
        first.append(e1)
        second.append(e2)
        
    # boxing을 위한 좌상단, 우하단 matching
    s_0 = second[0]
    second = second[1:]
    second.append(s_0)
    second.pop()
    last = tmp2[len(tmp2)-1]
    second.append(last)

    # 문단에 boxing을 할 높이 저장
    t = list(zip(first,second))

    height = []
    for i in range(len(t)):
        diff = abs(t[i][1][2][1]- t[i][0][0][1])
        height.append(diff)

    for i in range(len(height)):
      x = first[i][0][0]
      y = first[i][0][1]
      h = height[i]
      w = width
      
      coordinate.append((x,y,x+w,y+h))
      type.append('text')
    return coordinate, type

  coordinate, type = boxing(coordinate, type)

  def index_extraction(coordinate):
    paths = []
    # 좌표를 사용하여 이미지 crop
    for i, c in enumerate(coordinate):
        cropped_image = img.crop(c)
    # crop된 이미지 저장
        img_path = './cropped_image{}.png'.format(i)
        cropped_image.save(img_path)
        paths.append(img_path)

    #목차를 저장할 리스트
    head = []
    for k in paths:
        image = load_image(k)
        resized_image, resize_ratio = resize_image(image)
        response = ocr(resized_image, url, api_key)
        c = 0
        # 결과를 저장할 변수 초기화
        extracted_text = ""
        
        # 'results' 키의 값에서 'recognized_word'와 'box'를 가져와서 문장을 만듦
        results = response['responses'][0]['results']
        for i in range(len(results)):
            result = results[i]
            # 리스트의 첫 번째 문자열을 가져오고, 없으면 빈 문자열 사용
            recognized_word = result['recognized_word'][0] if result['recognized_word'] else ''  
            box = result['box']

            if recognized_word:
                extracted_text += recognized_word + ' '
                c += 1
                # 다음 'box'가 존재하고, 현재 'box'의 윗부분의 Y좌표와 다음 'box'의 윗부분의 Y 좌표가 차이가 20이 나면 줄 바꿈
                if i + 1 < len(results) and abs(box[0][1] - results[i + 1]['box'][0][1]) > 20:
                    head.append(extracted_text)
                    break
                # 목차 다음 글자가 없는 경우에도 head를 탐지하기 위한 조건
                if len(results) == c:
                    head.append(extracted_text)
    return head

  head = index_extraction(coordinate)
  fin = []
  for k in zip(coordinate, type):
    fin.append(k)
  sorted_data = sorted(fin, key=lambda x: (x[0][1], x[0][0]))

  def make_json(sorted_data):
    fin = []
    h_i = 0
    for i in range(len(coordinate)):
      if (i == 0) & (sorted_data[i][1] != 'text'):
        fin.append((sorted_data[i][0], sorted_data[i][1], '이전 장에 목차가 존재합니다.'))
        h_i -= 1
      elif (i == 0) & (sorted_data[i][1] == 'text'):
        fin.append((sorted_data[i][0], sorted_data[i][1], head[h_i]))
      else:
        if sorted_data[i][1] == 'text':
          h_i += 1
          fin.append((sorted_data[i][0], sorted_data[i][1], head[h_i]))
        else:
          fin.append((sorted_data[i][0], sorted_data[i][1], head[h_i]))
    return fin
  
  fin = make_json(sorted_data)
  
  final = []
  id = 1
  page = 1
  for k in fin:
    f = {}
    f["page"] = page
    f["id"] = id
    f["index"] = k[2]
    f["type"] = k[1]
    f["bbox"] = str(k[0])
    final.append(f)
    id += 1
  return final