import os
import torch
import torch.optim as optim
from tqdm import tqdm
import json
import segmentation_models_pytorch as smp
from dataset import get_dataloaders
from model import CombinedLoss, compute_metrics
from torch.amp import autocast, GradScaler


def train_model(model_name, data_root, save_dir, epochs=30, batch_size=1, lr=2e-5, img_size=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Model: {model_name} | {img_size}x{img_size} | lr={lr} | Cihaz: {device}")
    print(f"{'='*50}")

    train_loader, val_loader = get_dataloaders(data_root, img_size, batch_size)

    if model_name == "unet":
        model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "enet":
        model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights="imagenet", in_channels=3, classes=1)

    model = model.to(device)
    criterion = CombinedLoss(lambda1=0.4, lambda2=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = GradScaler("cuda")

    best_iou = 0
    history = []
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t_iou, t_f1, n = 0, 0, 0
        for images, masks in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch} Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                out = model(images)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            m = compute_metrics(out.detach().float(), masks)
            t_iou += m["iou"]; t_f1 += m["f1"]; n += 1

        model.eval()
        v_iou, v_f1, nv = 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"[{model_name}] Epoch {epoch} Val"):
                images, masks = images.to(device), masks.to(device)
                with autocast("cuda"):
                    out = model(images)
                m = compute_metrics(out.float(), masks)
                v_iou += m["iou"]; v_f1 += m["f1"]; nv += 1

        scheduler.step()
        print(f"Epoch {epoch}: Train IoU={t_iou/n:.4f} | Val IoU={v_iou/nv:.4f} | Val F1={v_f1/nv:.4f}")
        history.append({"epoch": epoch, "train_iou": t_iou/n, "val_iou": v_iou/nv, "val_f1": v_f1/nv})

        if v_iou/nv > best_iou:
            best_iou = v_iou/nv
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_name}.pth"))
            print(f"  YENİ EN İYİ! Val IoU: {best_iou:.4f}")

    with open(os.path.join(save_dir, f"history_{model_name}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n{model_name} tamamlandi! En iyi Val IoU: {best_iou:.4f}")
    return best_iou


if __name__ == "__main__":
    DATA_ROOT = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_DIR = os.path.expanduser("~/road_damage_project/models")

    results = {}
    for model_name in ["unet", "enet"]:
        best_iou = train_model(model_name, DATA_ROOT, MODEL_DIR, epochs=30, img_size=768)
        results[model_name] = best_iou

    print("\n=== KARSILASTIRMA SONUCLARI ===")
    for name, iou in results.items():
        print(f"{name:10s}: IoU={iou:.4f}")
