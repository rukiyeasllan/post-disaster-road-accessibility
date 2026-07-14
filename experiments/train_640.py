import os
import torch
import torch.optim as optim
from tqdm import tqdm
import json
from dataset import get_dataloaders
from model import SegFormerRoad, CombinedLoss, compute_metrics
from torch.cuda.amp import autocast, GradScaler


def train_640(data_root, model_path, save_dir, epochs=30, batch_size=2, lr=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"640x640 egitim | lr={lr} | batch={batch_size} | Cihaz: {device}")

    train_loader, val_loader = get_dataloaders(data_root, img_size=640, batch_size=batch_size)

    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Model yuklendi: {model_path}")

    criterion = CombinedLoss(lambda1=0.4, lambda2=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = GradScaler()

    best_iou = 0.6647
    history = []
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t_iou, t_f1, n = 0, 0, 0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
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
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val"):
                images, masks = images.to(device), masks.to(device)
                with autocast():
                    out = model(images)
                m = compute_metrics(out.float(), masks)
                v_iou += m["iou"]; v_f1 += m["f1"]; nv += 1

        scheduler.step()
        print(f"Epoch {epoch}: Train IoU={t_iou/n:.4f} | Val IoU={v_iou/nv:.4f} | Val F1={v_f1/nv:.4f}")
        history.append({"epoch": epoch, "train_iou": t_iou/n, "val_iou": v_iou/nv, "val_f1": v_f1/nv})

        if v_iou/nv > best_iou:
            best_iou = v_iou/nv
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  YENİ EN İYİ! Val IoU: {best_iou:.4f}")

    with open(os.path.join(save_dir, "history_640.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Tamamlandi! En iyi Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    DATA_ROOT  = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_PATH = os.path.expanduser("~/road_damage_project/models/best_model.pth")
    SAVE_DIR   = os.path.expanduser("~/road_damage_project/models")
    train_640(DATA_ROOT, MODEL_PATH, SAVE_DIR, epochs=30, batch_size=2, lr=2e-5)
