import os
import torch
from tqdm import tqdm
from dataset import get_dataloaders
from model import SegFormerRoad
from torch.amp import autocast


def find_best_threshold(data_root, model_path, img_size=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(data_root, img_size, batch_size=1)
    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    print("Threshold optimizasyonu:")
    results = {}
    thresholds = [0.35, 0.40, 0.42, 0.45, 0.47, 0.50, 0.52, 0.55]

    for thresh in thresholds:
        tp, fp, fn = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                with autocast("cuda"):
                    out = torch.sigmoid(model(images))
                pred = (out > thresh).float()
                tp += (pred * masks).sum().item()
                fp += (pred * (1-masks)).sum().item()
                fn += ((1-pred) * masks).sum().item()
        iou = tp / (tp + fp + fn + 1e-8)
        f1 = 2*tp / (2*tp + fp + fn + 1e-8)
        print(f"Threshold={thresh:.2f}: IoU={iou:.4f} F1={f1:.4f}")
        results[thresh] = iou

    best_thresh = max(results, key=results.get)
    print(f"\nEn iyi threshold: {best_thresh} → IoU: {results[best_thresh]:.4f}")
    return best_thresh, results[best_thresh]


if __name__ == "__main__":
    DATA_ROOT  = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_PATH = os.path.expanduser("~/road_damage_project/models/best_model.pth")
    find_best_threshold(DATA_ROOT, MODEL_PATH)
