import cv2
import numpy as np
import requests
from collections import Counter
from itertools import groupby
import pandas as pd


def table(path,key,url):
    print("import table")
    image_path = path
    api_key = key
    url = url

    def table_objection(image_path):
        image = cv2.imread(image_path)
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

    image, stats = table_objection(image_path)

    #stats: cell좌표 2부터 시작
    for x,y,w,h,area in stats[2:]:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite("final_img.png",image)


    def bycol(lst):
        col = 0
        codition = [item[1] for item in lst]
        con = Counter(codition)
        comb = list(con.items())
        first = comb[0][1]
        header = comb[0][1] + comb[1][1]
        for k in comb:
            if col < k[1]:
                col = k[1]
        return col, header, first

    def header(lst):
        cnt = 1
        for i in range(len(lst) - 1):
            if (lst[i+1][1] - lst[i][1]) < 15:
                cnt += 1
            else:
                break
        if len(lst) % cnt == 0:
            return cnt
        else:
            c, h, first = bycol(lst)
            return c, h, first
        
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

    image = load_image(image_path)
    resized_image, resize_ratio = resize_image(image)
    response = ocr(resized_image, url, api_key)


    def custom_sort_x(item):
        return round(item[0] / 10) * 10
    # y좌표를 10으로 나누고 반올림하여 정렬 기준으로 사용
    def custom_sort_y(item):
        return round(item[1] / 10) * 10
    # stats[2:]를 높이를 이용하여 정렬한 후 열의 숫자만큼 grouping하기 위한 함수
    def group_elements(original_list, group_size):
        grouped_list = [original_list[i::group_size] for i in range(group_size)]
        return grouped_list

    def make_dictionary():
        if type(header(stats[2:])) == tuple:
            c, h, first = header(stats[2:])
            ## step1. 열별로 value 정렬하기
            lst = list(stats[2+h:])
            lst.sort(key=custom_sort_x)
            
            # 좌상단 x좌표가 같은 원소들을 묶음
            result = [list(group) for _, group in groupby(lst, key=lambda x: x[0])]
            r = []
            for cell in lst:
                cell_x, cell_y, cell_width, cell_height = cell[0], cell[1], cell[2], cell[3]
                word = ''
                # 셀과 글자의 좌표 비교
                for text in response['responses'][0]['results']:
                    x, y = text['box'][0]
                    if cell_x - 3 <= x <= cell_x + cell_width and cell_y -1<= y <= cell_y + cell_height:
                        # 현재 글자가 현재 셀에 속한 것으로 판단
                        word += text['recognized_word'][0]
                r.append(word)
            # 리스트 컴프리헨션을 사용하여 2차원 리스트로 변환
            group_size = len(r) // c
            result_list = [r[i:i + group_size] for i in range(0, len(r), group_size)]
            
            ## step2. header 처리
            sub = stats[2:h+2]
            sub2 = stats[2:first+2]
            sub3 = sub[len(sub2):]
            # header가 2줄이 col의 index 추출
            index = []
            for i, a in enumerate(sub2):
                for b in sub3:
                    if a[0] == b[0]:
                        index.append(i)
            # header의 첫번째 줄 text 추출
            head = []
            for cell in sub2:
                cell_x, cell_y, cell_width, cell_height = cell[0], cell[1], cell[2], cell[3]
                word = ''
                # 셀과 글자의 좌표 비교
                for text in response['responses'][0]['results']:
                    x, y = text['box'][0]
                    if cell_x - 3 <= x <= cell_x + cell_width and cell_y -1<= y <= cell_y + cell_height:
                        # 현재 글자가 현재 셀에 속한 것으로 판단
                        word += text['recognized_word'][0]
                head.append(word)
            # header의 두번째 줄 text 추출
            head2 = []
            for cell in sub3:
                cell_x, cell_y, cell_width, cell_height = cell[0], cell[1], cell[2], cell[3]
                word = ''
                # 셀과 글자의 좌표 비교
                for text in response['responses'][0]['results']:
                    x, y = text['box'][0]
                    if cell_x - 3 <= x <= cell_x + cell_width and cell_y -1<= y <= cell_y + cell_height:
                        # 현재 글자가 현재 셀에 속한 것으로 판단
                        word += text['recognized_word'][0]
                head2.append(word)
            # 두번째 줄의 header가 반복되는 횟수
            l = int(len(head2)/len(index))
            # 조건에 맞게 header 생성 및 정렬
            cnt = 0
            for i, v in enumerate(index):
                j = 0
                n = len(head[v])
                if i % 2 == 0:
                    head[v] = head[v] + ' ' + head2[i]
                else:
                    head[v] = head[v] + ' ' + head2[i+1]
                j += 1
                while j != l:
                    if i % 2 == 0:
                        head.insert(v+1, head[v][:n] + ' ' + head2[i+1])
                    else:
                        head.insert(v+1, head[v][:n] + ' ' + head2[i])
                    j += 1
                cnt += l - 1
                if i < len(index) - 1:
                    index[i+1] = index[i+1] + cnt
            # header가 키이고 value가 value이 dictionary 생성
            final = {}
            for val in zip(head, result_list):
                final[val[0]] = val[1]
        # header가 한 줄일 경우
        elif type(header(stats[2:])) == int:
            lst = list(stats[2:])
            lst.sort(key=custom_sort_y)
            i = header(stats[2:])
            sub = lst[:i]
            temp = stats[2:]
            ## step2. header 처리하기
            # header에 해당하는 내용 저장
            h = []
            for cell in sub:
                cell_x, cell_y, cell_width, cell_height = cell[0], cell[1], cell[2], cell[3]
                word = ''
                # 셀과 글자의 좌표 비교
                for text in response['responses'][0]['results']:
                    x, y = text['box'][0]
                    if cell_x - 3 <= x <= cell_x + cell_width and cell_y -1<= y <= cell_y + cell_height:
                        # 현재 글자가 현재 셀에 속한 것으로 판단
                        word += text['recognized_word'][0]
                h.append(word)
            ## step1. value 처리하기
            lst = list(temp[i:])
            lst = sorted(lst, key=lambda x: (x[1], x[3]))  # 2번째 요소로 정렬 후 4번째 요소로 정렬
            lst = group_elements(lst, i)
            flattened_list = [element for sublist in lst for element in sublist]
            # value에 해당하는 내용저장
            r = []
            for cell in flattened_list:
                cell_x, cell_y, cell_width, cell_height = cell[0], cell[1], cell[2], cell[3]
                word = ''
                # 셀과 글자의 좌표 비교
                for text in response['responses'][0]['results']:
                    x, y = text['box'][0]
                    if cell_x - 3 <= x <= cell_x + cell_width and cell_y -1<= y <= cell_y + cell_height:
                        # 현재 글자가 현재 셀에 속한 것으로 판단
                        word += text['recognized_word'][0]
                r.append(word)
            # 리스트 컴프리헨션을 사용하여 2차원 리스트로 변환
            group_size = len(r) // i
            result_list = [r[i:i + group_size] for i in range(0, len(r), group_size)]
            final = {}
            for val in zip(h, result_list):
                final[val[0]] = val[1]
        return final
    final = make_dictionary()
    df = pd.DataFrame(final)
    df.to_csv("/results/table.csv",index=False)