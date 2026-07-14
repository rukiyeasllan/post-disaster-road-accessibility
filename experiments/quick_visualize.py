import os, torch, numpy as np, matplotlib.pyplot as plt
from dataset import DeepGlobeDataset
from model import SegFormerRoad

device = torch.device('cuda')
model = SegFormerRoad(pretrained=False).to(device)
model.load_state_dict(torch.load('/home/rukiye/road_damage_project/models/best_model.pth', map_location=device))
model.eval()

dataset = DeepGlobeDataset(image_dir='/home/rukiye/road_damage_project/data/deepglobe/train', img_size=512, augment=False)

mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

os.makedirs('/home/rukiye/road_damage_project/outputs/figures', exist_ok=True)

for i in [0, 10, 20, 30, 40]:
    image, mask = dataset[i]
    with torch.no_grad():
        pred = (torch.sigmoid(model(image.unsqueeze(0).to(device))) > 0.5).float().squeeze().cpu()
    img_show = np.clip((image * std + mean).permute(1,2,0).numpy(), 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_show); axes[0].set_title('Uydu Görüntüsü'); axes[0].axis('off')
    axes[1].imshow(mask.squeeze(), cmap='gray'); axes[1].set_title('Gerçek Maske'); axes[1].axis('off')
    axes[2].imshow(pred.numpy(), cmap='gray'); axes[2].set_title('SegFormer Tahmini'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(f'/home/rukiye/road_damage_project/outputs/figures/sample_{i}.png', dpi=150)
    plt.close()
    print(f'sample_{i}.png kaydedildi')
