import os
import torch
from tqdm import tqdm
from dataset import get_dataloaders
from model import SegFormerRoad
from torch.amp import autocast


def tta_predict(model, image):
    preds = []
    with torch.no_grad():
        with autocast("cuda"):
            preds.append(torch.sigmoid(model(image)))
            flipped = torch.flip(image, dims=[3])
            out_f = torch.sigmoid(model(flipped))
            preds.append(torch.flip(out_f, dims=[3]))
            flipped2 = torch.flip(image, dims=[2])
            out_f2 = torch.sigmoid(model(flipped2))
            preds.append(torch.flip(out_f2, dims=[2]))
    return torch.stack(preds).mean(dim=0)


def evaluate_tta_threshold(data_root, model_path, img_size=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_dataloaders(data_root, img_size, batch_size=1)
    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    for thresh in [0.40, 0.42, 0.45, 0.47, 0.50, 0.55]:
        tp, fp, fn = 0.0, 0.0, 0.0
        for images, masks in tqdm(val_loader, desc=f"TTA thresh={thresh}"):
            images = images.to(device)
            masks = masks.to(device)
            pred_avg = tta_predict(model, images)
            pred = (pred_avg > thresh).float()
            tp += (pred * masks).sum().item()
            fp += (pred * (1-masks)).sum().item()
            fn += ((1-pred) * masks).sum().item()
        iou = tp / (tp + fp + fn + 1e-8)
        f1 = 2*tp / (2*tp + fp + fn + 1e-8)
        print(f"TTA threshold={thresh:.2f}: IoU={iou:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    DATA_ROOT  = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_PATH = os.path.expanduser("~/road_damage_project/models/best_model.pth")
    evaluate_tta_threshold(DATA_ROOT, MODEL_PATH)
