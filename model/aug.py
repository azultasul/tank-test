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

image_images_dir = './data/train/images/'
image_labels_dir = './data/train/labels/'
aug_images_dir = './data/train/aug_images/'
aug_labels_dir = './data/train/aug_labels/'

for filename in os.listdir(image_images_dir):
    if not filename.endswith('.jpg'):
        continue

    # transform - image 
    image_path = os.path.join(image_images_dir, filename)
    image = cv2.imread(image_path)

    # transform - image 
    # transform - class_labels 
    bboxes = []
    class_labels = []
    
    label_filename = filename.replace('.jpg', '.txt')
    label_path = os.path.join(image_labels_dir, label_filename)

    # bbox 구하기 위해 polygon 필요 - 이미지 크기 얻기
    image_height, image_width, _ = image.shape

    print('===========', image_height, image_width)
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = parts[0]  # 클래스 ID
            bbox = parts[1:] # [x_center, y_center, width, height]
            
            bboxes.append(bbox)
            class_labels.append(class_id)

    # print("????????", image)
    # print("????????", bboxes)
    # print("????????", class_labels)

    # 이미지 증강
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    print(augmented)
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # # 증강 이미지 저장
    # aug_images_path = os.path.join(aug_images_dir, f'{filename}_aug.jpg')
    # aug_labels_path = os.path.join(aug_labels_dir, f'{filename}_aug.txt')

    # cv2.imwrite(aug_images_path, aug_image)
    # with open(label_path, 'w') as f:
    #     for label, bbox in zip(aug_labels, aug_bboxes):
    #         x_center, y_center, w, h = bbox  # 이미 YOLO 형식일 경우
    #         f.write(f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")