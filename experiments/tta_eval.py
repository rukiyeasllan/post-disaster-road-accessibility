import os
import torch
import numpy as np
from tqdm import tqdm
import json
from dataset import get_dataloaders
from model import SegFormerRoad
from torch.amp import autocast


def tta_predict(model, image, device):
    preds = []
    with torch.no_grad():
        with autocast("cuda"):
            # Orijinal
            out = torch.sigmoid(model(image))
            preds.append(out)
            # Yatay flip
            flipped = torch.flip(image, dims=[3])
            out_f = torch.sigmoid(model(flipped))
            preds.append(torch.flip(out_f, dims=[3]))
            # Dikey flip
            flipped2 = torch.flip(image, dims=[2])
            out_f2 = torch.sigmoid(model(flipped2))
            preds.append(torch.flip(out_f2, dims=[2]))
    return torch.stack(preds).mean(dim=0)


def evaluate_with_tta(data_root, model_path, img_size=768, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TTA Degerlendirme | img_size={img_size} | Cihaz: {device}")

    _, val_loader = get_dataloaders(data_root, img_size, batch_size)

    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Model yuklendi: {model_path}")

    print("\n--- Normal Tahmin ---")
    iou_sum, f1_sum, n = 0, 0, 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Normal"):
            images, masks = images.to(device), masks.to(device)
            with autocast("cuda"):
                out = model(images)
            pred = (torch.sigmoid(out) > 0.5).float()
            tp = (pred * masks).sum()
            fp = (pred * (1-masks)).sum()
            fn = ((1-pred) * masks).sum()
            iou = tp / (tp + fp + fn + 1e-8)
            f1 = 2*tp / (2*tp + fp + fn + 1e-8)
            iou_sum += iou.item(); f1_sum += f1.item(); n += 1
    print(f"Normal IoU: {iou_sum/n:.4f} | F1: {f1_sum/n:.4f}")

    print("\n--- TTA Tahmin (flip only) ---")
    iou_tta, f1_tta, n = 0, 0, 0
    for images, masks in tqdm(val_loader, desc="TTA"):
        images, masks = images.to(device), masks.to(device)
        pred_avg = tta_predict(model, images, device)
        pred = (pred_avg > 0.5).float()
        tp = (pred * masks).sum()
        fp = (pred * (1-masks)).sum()
        fn = ((1-pred) * masks).sum()
        iou = tp / (tp + fp + fn + 1e-8)
        f1 = 2*tp / (2*tp + fp + fn + 1e-8)
        iou_tta += iou.item(); f1_tta += f1.item(); n += 1
    print(f"TTA IoU: {iou_tta/n:.4f} | F1: {f1_tta/n:.4f}")
    print(f"Iyilesme: +{(iou_tta-iou_sum)/n:.4f} IoU")

    results = {"normal_iou": iou_sum/n, "tta_iou": iou_tta/n, "tta_f1": f1_tta/n}
    with open(os.path.expanduser("~/road_damage_project/models/tta_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    DATA_ROOT  = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_PATH = os.path.expanduser("~/road_damage_project/models/best_model.pth")
    evaluate_with_tta(DATA_ROOT, MODEL_PATH, img_size=768, batch_size=1)
