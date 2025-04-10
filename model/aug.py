import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# transform = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.MotionBlur(p=0.2),
#         A.RandomFog(p=0.1),
#         A.RandomShadow(p=0.1),
#         A.Perspective(scale=(0.05, 0.1), p=0.3),
#         A.OneOf([
#             # A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, p=1.0),
#             A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=1.0)
#         ], p=0.3),
#         A.Resize(640, 640),  # YOLO input size
#         ToTensorV2()
#     ],
#     bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
# )

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=10, p=0.3),
],
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def get_image_path(folder_path):
    paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            paths.append(image_path)

    return paths

def polygon_to_yolo_bounding_box(polygon, image_width, image_height):
    """
    다각형 좌표를 YOLO 형식의 바운딩 박스로 변환하는 함수.
    
    Args:
        polygon (list of tuples): 다각형의 각 좌표들 [(x1, y1), (x2, y2), ...]
        image_width (int): 이미지의 가로 크기
        image_height (int): 이미지의 세로 크기

    Returns:
        tuple: (class_id, x_center, y_center, width, height)
    """
    # 다각형 좌표를 NumPy 배열로 변환
    polygon = np.array(polygon, dtype=np.float32)
    
    # 최소 외접 직사각형 구하기
    rect = cv2.minAreaRect(polygon)
    box = cv2.boxPoints(rect)
    # box = np.int0(box)  # 바운딩 박스의 꼭지점 (int형으로 변환)

    # 바운딩 박스의 중심, 너비, 높이 계산
    center = rect[0]  # 중심 (x_center, y_center)
    width = rect[1][0]  # 직사각형의 너비
    height = rect[1][1]  # 직사각형의 높이
    angle = rect[2]  # 회전 각도 (YOLO에는 사용 안 함)

    # YOLO 형식으로 변환: 좌표를 0~1 범위로 정규화
    x_center = center[0] / image_width
    y_center = center[1] / image_height
    width = width / image_width
    height = height / image_height

    return (x_center, y_center, width, height)

image_images_path = './data/train/images/'
image_label_path = './data/train/labels/'
aug_images_path = './data/train/images/'
aug_label_path = './data/train/labels/'

for filename in os.listdir(image_images_path):
    if not filename.endswith('.jpg'):
        continue

    # transform - image 
    image_path = os.path.join(image_images_path, filename)
    image = cv2.imread(image_path)

    # transform - image 
    # transform - class_labels 
    bboxes = []
    class_labels = []
    
    label_filename = filename.replace('.jpg', '.txt')
    label_path = os.path.join(image_label_path, label_filename)

    # bbox 구하기 위해 polygon 필요 - 이미지 크기 얻기
    image_height, image_width, _ = image.shape

    print('===========')
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])  # 클래스 ID
            label = parts[1:]
            polygon = [(label[i], label[i + 1]) for i in range(0, len(label), 2)]
            # [x_center, y_center, width, height]
            
            bbox = polygon_to_yolo_bounding_box(polygon, image_width, image_height)
            # print(bbox)
            bboxes.append(bbox)
            class_labels.append(class_id)


    print("????????", image)
    print("????????", bboxes)
    print("????????", class_labels)

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']