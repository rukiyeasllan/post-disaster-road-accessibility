import os
import numpy as np
from torch.utils.data import Dataset, random_split
from PIL import Image
import torch


class DeepGlobeDataset(Dataset):
    def __init__(self, image_dir, img_size=512, augment=False):
        self.image_dir = image_dir
        self.img_size = img_size
        self.augment = augment
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith("_sat.jpg")])
        print(f"Yuklenen goruntu sayisi: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace("_sat.jpg", "_mask.png")
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.image_dir, mask_name)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        if self.augment:
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = (image - mean) / std
        return image, mask


def get_dataloaders(data_root, img_size=512, batch_size=4):
    from torch.utils.data import DataLoader
    full_dataset = DeepGlobeDataset(image_dir=os.path.join(data_root, "train"), img_size=img_size, augment=False)
    total = len(full_dataset)
    val_size = int(total * 0.15)
    train_size = total - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_dataset.dataset.augment = True
    print(f"Train: {train_size} | Val: {val_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader
