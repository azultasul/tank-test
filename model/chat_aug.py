import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 변환 정의
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=10, p=0.3),
],
bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.MotionBlur(p=0.2),
    A.RandomFog(p=0.1),
    A.RandomShadow(p=0.1),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.OneOf([
        # A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, p=1.0),
        A.CoarseDropout(max_holes=4, max_height=16, max_width=16, p=1.0)
    ], p=0.3),
    A.Resize(640, 640),  # YOLO input size
    ToTensorV2()
])

# 경로 설정
image_dir = 'data/train/images'
label_dir = 'data/train/labels'
aug_image_dir = 'data/train/images_aug'
aug_label_dir = 'data/train/labels_aug'
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

# 바운딩 박스 그리는 함수
def draw_boxes(image, bboxes, labels):
    h, w, _ = image.shape
    for bbox, cls in zip(bboxes, labels):
        x_center, y_center, bw, bh = bbox
        x1 = int((x_center - bw / 2) * w)
        y1 = int((y_center - bh / 2) * h)
        x2 = int((x_center + bw / 2) * w)
        y2 = int((y_center + bh / 2) * h)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(cls), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

# 증강 루프
for filename in os.listdir(image_dir):
    print(filename)
    if not filename.endswith('.jpg'):
        continue

    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # YOLO 라벨 읽기
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            bbox = list(map(float, parts[1:]))
            bboxes.append(bbox)
            class_labels.append(cls)

    # Albumentations 증강
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    # 시각화
    vis_image = draw_boxes(aug_image.copy(), aug_bboxes, aug_labels)
    cv2.imshow('Augmented Image', vis_image)
    key = cv2.waitKey(0)
    if key == 27:  # ESC 누르면 종료
        break

    # 저장
    cv2.imwrite(os.path.join(aug_image_dir, filename), aug_image)
    with open(os.path.join(aug_label_dir, filename.replace('.jpg', '.txt')), 'w') as f:
        for cls, bbox in zip(aug_labels, aug_bboxes):
            f.write(f"{cls} {' '.join(map(str, bbox))}\n")

cv2.destroyAllWindows()
