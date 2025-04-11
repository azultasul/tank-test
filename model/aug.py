import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.3
        ),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5),
            A.Blur(blur_limit=(3, 5), p=0.5),
        ], p=0.3),
        A.RandomFog(p=0.1),
        A.RandomShadow(p=0.1),
        A.Perspective(scale=(0.05, 0.1), p=0.2),
        A.OneOf([
            # ☁️ 안개
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6, alpha_coef=0.1, p=1.0),

            # 🌧️ 비
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                blur_value=3,
                brightness_coefficient=0.7,
                rain_type=None,
                p=1.0
            ),

            # ❄️ 눈
            A.RandomSnow(
                snow_point_lower=0.1,
                snow_point_upper=0.3,
                brightness_coeff=1.2,
                p=1.0
            ),

            # 🌞 햇빛/플레어
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),  # 상단 50% 영역에만 적용
                angle_lower=0.3,
                src_radius=100,
                src_color=(255, 255, 255),
                p=1.0
            ),

            # 🌑 그림자 (나무, 건물 등)
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                shadow_roi=(0, 0.5, 1, 1),  # 하단 절반에만 그림자
                p=1.0
            ),
        ], p=0.3),
        A.Resize(640, 640),
        # ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

image_images_dir = './data/train/images/'
image_labels_dir = './data/train/labels/'
aug_images_dir = './data/train/aug_images/'
aug_labels_dir = './data/train/aug_labels/'


image_images_dir = './data/train/images/'

os.makedirs(aug_images_dir, exist_ok=True)
os.makedirs(aug_labels_dir, exist_ok=True)

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

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = parts[0]  # 클래스 ID
            bbox = [float(i) for i in parts[1:]] # [x_center, y_center, width, height]
            
            bboxes.append(bbox)
            class_labels.append(int(float(class_id)))

    # print("????????", image)
    # print("????????", bboxes)
    # print("????????", [int(float(i)) for i in class_labels])

    # 이미지 증강
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    # print(augmented)
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # 증강 이미지 저장
    aug_images_path = os.path.join(aug_images_dir, f'{filename}_aug.jpg')
    aug_labels_path = os.path.join(aug_labels_dir, f'{filename}_aug.txt')

    cv2.imwrite(aug_images_path, aug_image)
    with open(aug_labels_path, 'w') as f:
        for label, bbox in zip(aug_labels, aug_bboxes):
            x_center, y_center, w, h = bbox  # 이미 YOLO 형식일 경우
            f.write(f"{label} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")