import os
import torch
import torch.optim as optim
from tqdm import tqdm
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch.nn.functional as F
from model import SegFormerRoad, CombinedLoss, compute_metrics


class AugmentedDataset(Dataset):
    def __init__(self, image_dir, img_size=512, augment=False):
        self.image_dir = image_dir
        self.img_size = img_size
        self.augment = augment
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith("_sat.jpg")])
        print(f"Yuklenen goruntu: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace("_sat.jpg", "_mask.png")
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.image_dir, mask_name)).convert("L")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            # Yatay flip
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            # Dikey flip
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            # 90 derece rotasyon
            if np.random.rand() > 0.5:
                k = np.random.randint(1, 4)
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
            # Renk jitter
            if np.random.rand() > 0.5:
                factor = np.random.uniform(0.7, 1.3)
                image = np.clip(image * factor, 0, 1)
            # Gaussian noise
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
                image = np.clip(image + noise, 0, 1)
            # Random brightness
            if np.random.rand() > 0.5:
                delta = np.random.uniform(-0.15, 0.15)
                image = np.clip(image + delta, 0, 1)
            # Cutout - overfit azaltır
            if np.random.rand() > 0.5:
                h, w = image.shape[:2]
                cut_h = np.random.randint(32, 96)
                cut_w = np.random.randint(32, 96)
                y = np.random.randint(0, h - cut_h)
                x = np.random.randint(0, w - cut_w)
                image[y:y+cut_h, x:x+cut_w] = 0

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        image = (image - mean) / std
        return image, mask


def get_loaders(data_root, img_size=512, batch_size=4):
    full = AugmentedDataset(os.path.join(data_root, "train"), img_size, augment=False)
    total = len(full)
    val_size = int(total * 0.15)
    train_size = total - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_ds.dataset.augment = True
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader


def finetune(data_root, model_path, save_dir, epochs=20, batch_size=4, lr=1e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning | lr={lr} | epochs={epochs} | Cihaz: {device}")

    train_loader, val_loader = get_loaders(data_root, img_size=512, batch_size=batch_size)

    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Model yuklendi: {model_path}")

    criterion = CombinedLoss(lambda1=0.4, lambda2=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    best_iou = 0
    history = []
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # TRAIN
        model.train()
        t_loss, t_iou, t_f1 = 0, 0, 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            m = compute_metrics(out.detach(), masks)
            t_iou += m["iou"]; t_f1 += m["f1"]

        n = len(train_loader)
        # VAL
        model.eval()
        v_loss, v_iou, v_f1 = 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val"):
                images, masks = images.to(device), masks.to(device)
                out = model(images)
                v_loss += criterion(out, masks).item()
                m = compute_metrics(out, masks)
                v_iou += m["iou"]; v_f1 += m["f1"]

        nv = len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch}: Train IoU={t_iou/n:.4f} F1={t_f1/n:.4f} | Val IoU={v_iou/nv:.4f} F1={v_f1/nv:.4f}")
        history.append({"epoch": epoch, "train_iou": t_iou/n, "val_iou": v_iou/nv, "val_f1": v_f1/nv})

        if v_iou/nv > best_iou:
            best_iou = v_iou/nv
            torch.save(model.state_dict(), os.path.join(save_dir, "best_segformer_ft.pth"))
            print(f"  Kaydedildi! Val IoU: {best_iou:.4f}")

    with open(os.path.join(save_dir, "history_finetune.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Tamamlandi! En iyi Val IoU: {best_iou:.4f}")
    return best_iou


if __name__ == "__main__":
    DATA_ROOT  = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_PATH = os.path.expanduser("~/road_damage_project/models/best_model.pth")
    SAVE_DIR   = os.path.expanduser("~/road_damage_project/models")
    finetune(DATA_ROOT, MODEL_PATH, SAVE_DIR, epochs=20, lr=1e-5)
