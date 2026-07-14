import os
import torch
import torch.optim as optim
from tqdm import tqdm
import json
from dataset import get_dataloaders
from model import SegFormerRoad, CombinedLoss, compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    metrics_sum = {'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0}
    for images, masks in tqdm(loader, desc="Egitim"):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        m = compute_metrics(outputs.detach(), masks)
        for k in metrics_sum:
            metrics_sum[k] += m[k]
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0}
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Dogrulama"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            m = compute_metrics(outputs, masks)
            for k in metrics_sum:
                metrics_sum[k] += m[k]
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in metrics_sum.items()}


def train(data_root, model_save_dir, epochs=30, batch_size=4, lr=6e-5, img_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Cihaz: {device}")
    train_loader, val_loader = get_dataloaders(data_root, img_size, batch_size)
    model = SegFormerRoad(pretrained=True).to(device)
    criterion = CombinedLoss(lambda1=0.4, lambda2=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_iou = 0
    history = []
    os.makedirs(model_save_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_loss, train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_m = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f} | IoU: {train_m['iou']:.4f} | F1: {train_m['f1']:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | IoU: {val_m['iou']:.4f} | F1: {val_m['f1']:.4f}")
        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_iou': train_m['iou'], 'train_f1': train_m['f1'],
            'val_loss': val_loss, 'val_iou': val_m['iou'], 'val_f1': val_m['f1']
        })
        if val_m['iou'] > best_iou:
            best_iou = val_m['iou']
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'best_model.pth'))
            print(f"  En iyi model kaydedildi (IoU: {best_iou:.4f})")
    with open(os.path.join(model_save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nEgitim tamamlandi! En iyi Val IoU: {best_iou:.4f}")
    return history


if __name__ == '__main__':
    DATA_ROOT = os.path.expanduser('~/road_damage_project/data/deepglobe')
    MODEL_DIR = os.path.expanduser('~/road_damage_project/models')
    train(
        data_root=DATA_ROOT,
        model_save_dir=MODEL_DIR,
        epochs=30,
        batch_size=4,
        lr=6e-5,
        img_size=512
    )
