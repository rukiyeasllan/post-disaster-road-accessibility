import json, torch, os
from dataset import get_dataloaders
from model import SegFormerRoad, CombinedLoss
import segmentation_models_pytorch as smp

device = torch.device('cuda')
data_root = '/home/rukiye/road_damage_project/data/deepglobe'
_, val_loader = get_dataloaders(data_root, img_size=768, batch_size=2)
criterion = CombinedLoss(lambda1=0.4, lambda2=0.6)

def get_val_losses(model, name):
    model.eval()
    losses = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            out = model(images)
            loss = criterion(out, masks)
            losses.append(loss.item())
    avg = sum(losses)/len(losses)
    print(f'{name} val_loss: {avg:.4f}')
    return avg

segformer = SegFormerRoad(pretrained=False).to(device)
segformer.load_state_dict(torch.load('/home/rukiye/road_damage_project/models/best_model.pth', map_location=device))

unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
unet.load_state_dict(torch.load('/home/rukiye/road_damage_project/models/best_unet.pth', map_location=device))

enet = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1).to(device)
enet.load_state_dict(torch.load('/home/rukiye/road_damage_project/models/best_enet.pth', map_location=device))

results = {
    'segformer': get_val_losses(segformer, 'SegFormer'),
    'unet': get_val_losses(unet, 'U-Net'),
    'enet': get_val_losses(enet, 'ENet')
}

with open('/home/rukiye/road_damage_project/models/val_losses.json', 'w') as f:
    json.dump(results, f)
print('Kaydedildi!')
