import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F


class SegFormerRoad(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b2",
                num_labels=1,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig.from_pretrained("nvidia/mit-b2")
            config.num_labels = 1
            self.model = SegformerForSemanticSegmentation(config)
        
        print("SegFormer-B2 modeli yüklendi.")

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, 1, H/4, W/4)
        
        # Orijinal boyuta upsample
        upsampled = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return upsampled  # (B, 1, H, W)


class CombinedLoss(nn.Module):
    def __init__(self, lambda1=0.4, lambda2=0.6):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target)
        return self.lambda1 * bce + self.lambda2 * dice


def compute_metrics(pred_logits, target):
    pred = (torch.sigmoid(pred_logits) > 0.5).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    tn = ((1 - pred) * (1 - target)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    return {
        'iou': iou.item(),
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'accuracy': accuracy.item()
    }
