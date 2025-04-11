import cv2
import numpy as np
import os
import random

# 경로 설정
img_dir = "images/"
label_dir = "labels/"
save_img_dir = "aug_images/"
save_label_dir = "aug_labels/"

os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_label_dir, exist_ok=True)

# YOLO -> (x1, y1, x2, y2) 변환 함수
def yolo_to_xyxy(bbox, img_w, img_h):
    cx, cy, w, h = bbox
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2

# (x1, y1, x2, y2) -> YOLO 형식
def xyxy_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return [cx, cy, w, h]

# 이미지 리스트
img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]

# Cut & Paste 2개 합성
for i in range(10):  # 원하는 수 만큼 반복
    img1_file = random.choice(img_files)
    img2_file = random.choice(img_files)

    # 이미지 & 라벨 로딩
    img1 = cv2.imread(os.path.join(img_dir, img1_file))
    img2 = cv2.imread(os.path.join(img_dir, img2_file))
    h, w, _ = img1.shape

    # 라벨 로딩
    def load_yolo_label(img_name):
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            cls, cx, cy, bw, bh = map(float, f.readline().strip().split())
        return int(cls), [cx, cy, bw, bh]

    cls1, bbox1 = load_yolo_label(img1_file)
    cls2, bbox2 = load_yolo_label(img2_file)

    # bbox로 crop
    x1_1, y1_1, x2_1, y2_1 = yolo_to_xyxy(bbox1, w, h)
    x1_2, y1_2, x2_2, y2_2 = yolo_to_xyxy(bbox2, w, h)
    obj1 = img1[y1_1:y2_1, x1_1:x2_1]
    obj2 = img2[y1_2:y2_2, x1_2:x2_2]

    # 빈 캔버스 만들기 (흰 배경 or 랜덤 이미지)
    new_img = np.ones_like(img1) * 255

    # 랜덤 위치 설정 (단, 객체가 이미지 범위를 넘지 않도록)
    def get_random_position(obj, img_shape):
        oh, ow = obj.shape[:2]
        max_x = img_shape[1] - ow
        max_y = img_shape[0] - oh
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return x, y

    x1, y1 = get_random_position(obj1, new_img.shape)
    x2, y2 = get_random_position(obj2, new_img.shape)

    # 붙여넣기
    new_img[y1:y1+obj1.shape[0], x1:x1+obj1.shape[1]] = obj1
    new_img[y2:y2+obj2.shape[0], x2:x2+obj2.shape[1]] = obj2

    # bbox 새로 계산
    new_bbox1 = xyxy_to_yolo([x1, y1, x1 + obj1.shape[1], y1 + obj1.shape[0]], w, h)
    new_bbox2 = xyxy_to_yolo([x2, y2, x2 + obj2.shape[1], y2 + obj2.shape[0]], w, h)

    # 저장
    new_filename = f"aug_{i}.jpg"
    cv2.imwrite(os.path.join(save_img_dir, new_filename), new_img)

    with open(os.path.join(save_label_dir, new_filename.replace('.jpg', '.txt')), 'w') as f:
        f.write(f"{cls1} {' '.join(map(str, new_bbox1))}\n")
        f.write(f"{cls2} {' '.join(map(str, new_bbox2))}\n")
