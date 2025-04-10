import os 
from PIL import Image
import cv2
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_image_path(folder_path):
    paths = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            paths.append(image_path)

    return paths

test = get_image_path('./data/')
print(test)



def cv_image_read(image_path):
    # print(cv2.imread(image_path))
    return cv2.imread(image_path)

def show_image(cv_image):
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(rgb)
    plt.show()


# # Albumentations 데이터세트 클래스
# class AlbumentationsDataset(Dataset):
#     """__init__ and __len__ functions are the same as in TorchvisionDataset"""
#     def __init__(self, file_paths, labels, transform=None):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         label = self.labels[idx]
#         file_path = self.file_paths[idx]
#         # 이미지 읽기
#         image = cv2.imread(file_path)
#         # BGR opencv 이미지를 RGB 이미지로 변경
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # 이미지 변경 수행
#         if self.transform:
#             augmented = self.transform(image=image)
#             image = augmented['image']
#         return image, label



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

# albumentations_dataset = AlbumentationsDataset(
#     file_paths=paths,
#     labels=list(range(21)),
#     transform=train_transform,
# )

# print(len(albumentations_dataset))

# # 랜덤으로 2번 변형 수행
# for i in range(2):
#   sample, _ = albumentations_dataset[20]

#   plt.figure()
#   plt.imshow(transforms.ToPILImage()(sample))
#   plt.show()