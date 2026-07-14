import os, torch, numpy as np, matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from dataset import DeepGlobeDataset
from model import SegFormerRoad

device = torch.device('cuda')

segformer = SegFormerRoad(pretrained=False).to(device)
segformer.load_state_dict(torch.load('/home/rukiye/road_damage_project/models/best_model.pth', map_location=device))
segformer.eval()

unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
unet.load_state_dict(torch.load('/home/rukiye/road_damage_project/models/best_unet.pth', map_location=device))
unet.eval()

enet = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, in_channels=3, classes=1).to(device)
enet.load_state_dict(torch.load('/home/rukiye/road_damage_project/models/best_enet.pth', map_location=device))
enet.eval()

dataset = DeepGlobeDataset(image_dir='/home/rukiye/road_damage_project/data/deepglobe/train', img_size=512, augment=False)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
os.makedirs('/home/rukiye/road_damage_project/outputs/figures', exist_ok=True)

for i in [0, 10, 20]:
    image, mask = dataset[i]
    inp = image.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_sf = (torch.sigmoid(segformer(inp)) > 0.5).float().squeeze().cpu()
        pred_un = (torch.sigmoid(unet(inp)) > 0.5).float().squeeze().cpu()
        pred_en = (torch.sigmoid(enet(inp)) > 0.5).float().squeeze().cpu()

    img_show = np.clip((image * std + mean).permute(1,2,0).numpy(), 0, 1)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    titles = ['Girdi', 'Gerçek Maske', 'ENet', 'U-Net', 'SegFormer-B2']
    imgs = [img_show, mask.squeeze().numpy(), pred_en.numpy(), pred_un.numpy(), pred_sf.numpy()]
    cmaps = [None, 'gray', 'gray', 'gray', 'gray']
    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=13, pad=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'/home/rukiye/road_damage_project/outputs/figures/compare_{i}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'compare_{i}.png kaydedildi')
