import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import cv2, os

aug = A.Compose([
    A.Resize(256,256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2()
])

class PupilMaskDataset(Dataset):
    def __init__(self,img_dir,mask_dir):
        self.names = os.listdir(img_dir)
        self.img_dir, self.mask_dir = img_dir, mask_dir
    def __len__(self): return len(self.names)
    def __getitem__(self,idx):
        name = self.names[idx]
        img = cv2.imread(os.path.join(self.img_dir,name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_dir,name), cv2.IMREAD_GRAYSCALE)
        augmented = aug(image=img, mask=mask)
        return augmented['image'], augmented['mask'].unsqueeze(0)/255.0