import os
import torch
from tqdm import tqdm
from dataset import get_dataloaders
import segmentation_models_pytorch as smp
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


def evaluate_model(model_name, model_path, data_root, img_size=768):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*50}")
    print(f"Model: {model_name} | TTA + Threshold Optimizasyonu")
    print(f"{'='*50}")

    _, val_loader = get_dataloaders(data_root, img_size, batch_size=1)

    if model_name == "unet":
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    elif model_name == "enet":
        model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Model yuklendi: {model_path}")

    for thresh in [0.35, 0.40, 0.45, 0.47, 0.50, 0.55]:
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
    DATA_ROOT = os.path.expanduser("~/road_damage_project/data/deepglobe")
    MODEL_DIR = os.path.expanduser("~/road_damage_project/models")

    evaluate_model("unet", os.path.join(MODEL_DIR, "best_unet.pth"), DATA_ROOT)
    evaluate_model("enet", os.path.join(MODEL_DIR, "best_enet.pth"), DATA_ROOT)
