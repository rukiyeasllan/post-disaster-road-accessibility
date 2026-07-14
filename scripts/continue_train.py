import os
import torch
import torch.optim as optim
from tqdm import tqdm
import json
from dataset import get_dataloaders
from model import SegFormerRoad, CombinedLoss, compute_metrics


def continue_train(data_root, model_path, save_dir, epochs=30, batch_size=4, lr=3e-5, img_size=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Devam egitimi | lr={lr} | epochs={epochs} | Cihaz: {device}")

    train_loader, val_loader = get_dataloaders(data_root, img_size, batch_size)

    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Model yuklendi: {model_path}")

    criterion = CombinedLoss(lambda1=0.4, lambda2=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_iou = 0.6526  # mevcut en iyi
    history = []
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t_iou, t_f1, t_loss = 0, 0, 0
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
        model.eval()
        v_iou, v_f1, v_loss = 0, 0, 0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} Val"):
                images, masks = images.to(device), masks.to(device)
                out = model(images)
                v_loss += criterion(out, masks).item()
                m = compute_metrics(out, masks)
                v_iou += m["iou"]; v_f1 += m["f1"]

        nv = len(val_loader)
        scheduler.step()

        print(f"Epoch {epoch}: Train IoU={t_iou/n:.4f} | Val IoU={v_iou/nv:.4f} | Val F1={v_f1/nv:.4f}")
        history.append({"epoch": epoch, "train_iou": t_iou/n, "val_iou": v_iou/nv, "val_f1": v_f1/nv})

        if v_iou/nv > best_iou:
            best_iou = v_iou/nv
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"  YENİ EN İYİ! Val IoU: {best_iou:.4f}")

    with open(os.path.join(save_dir, "history_continue.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Tamamlandi! En iyi Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    DATA_ROOT  = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_PATH = os.path.expanduser("~/road_damage_project/models/best_model.pth")
    SAVE_DIR   = os.path.expanduser("~/road_damage_project/models")
    continue_train(DATA_ROOT, MODEL_PATH, SAVE_DIR, epochs=30, lr=3e-5)
