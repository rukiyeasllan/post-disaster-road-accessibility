import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import DeepGlobeDataset
from model import SegFormerRoad, compute_metrics
from torch.utils.data import DataLoader


def evaluate_model(data_root, model_path, img_size=512, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model yükle
    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model yüklendi: {model_path}")
    
    # Test verisi
    test_dataset = DeepGlobeDataset(
        image_dir=os.path.join(data_root, 'train'),
        img_size=img_size,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    metrics_sum = {'iou': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            m = compute_metrics(outputs, masks)
            for k in metrics_sum:
                metrics_sum[k] += m[k]
    
    n = len(test_loader)
    results = {k: v / n for k, v in metrics_sum.items()}
    
    print("\n=== TEST SONUÇLARI ===")
    print(f"IoU:       {results['iou']*100:.2f}%")
    print(f"F1 Skoru:  {results['f1']*100:.2f}%")
    print(f"Kesinlik:  {results['precision']*100:.2f}%")
    print(f"Duyarlılık:{results['recall']*100:.2f}%")
    print(f"Doğruluk:  {results['accuracy']*100:.2f}%")
    
    return results


def visualize_predictions(data_root, model_path, output_dir, num_samples=5, img_size=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SegFormerRoad(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    dataset = DeepGlobeDataset(
        image_dir=os.path.join(data_root, 'train'),
        img_size=img_size,
        augment=False
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    
    for i in range(min(num_samples, len(dataset))):
        image, mask = dataset[i]
        
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu()
        
        # Görüntüyü normalize'den geri al
        img_show = (image * std + mean).permute(1,2,0).numpy()
        img_show = np.clip(img_show, 0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_show);        axes[0].set_title('Uydu Görüntüsü');  axes[0].axis('off')
        axes[1].imshow(mask.squeeze(), cmap='gray'); axes[1].set_title('Gerçek Maske'); axes[1].axis('off')
        axes[2].imshow(pred.numpy(), cmap='gray');   axes[2].set_title('SegFormer Tahmini'); axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'), dpi=150)
        plt.close()
        print(f"Kaydedildi: sample_{i+1}.png")


if __name__ == '__main__':
    DATA_ROOT  = os.path.expanduser('~/road_damage_project/data/deepglobe')
    MODEL_PATH = os.path.expanduser('~/road_damage_project/models/best_model.pth')
    OUTPUT_DIR = os.path.expanduser('~/road_damage_project/outputs/figures')
    
    evaluate_model(DATA_ROOT, MODEL_PATH)
    visualize_predictions(DATA_ROOT, MODEL_PATH, OUTPUT_DIR)
