# 🚧 Post-Disaster Road Accessibility Assessment using SegFormer-B2

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Research](https://img.shields.io/badge/Status-Research-orange)
![OpenStreetMap](https://img.shields.io/badge/OpenStreetMap-Integrated-success)

</p>

<p align="center">

A deep learning framework for automatic post-disaster road accessibility assessment using semantic segmentation, graph analysis, and OpenStreetMap comparison.

</p>

---

# 📖 Overview

Natural disasters such as earthquakes, floods, and landslides frequently cause severe disruptions to transportation networks. Rapid identification of inaccessible roads is essential for emergency response, rescue operations, and humanitarian aid delivery.

This repository presents an end-to-end deep learning framework that automatically estimates post-disaster road accessibility from aerial and satellite imagery.

Unlike conventional methods requiring both pre-disaster and post-disaster images, the proposed framework operates using only post-disaster imagery, enabling rapid deployment immediately after a disaster.

The framework combines semantic segmentation, skeleton extraction, graph analysis, and OpenStreetMap (OSM) integration to detect disconnected or damaged road segments.

---

# 🎯 Main Contributions

- Semantic road extraction using **SegFormer-B2**
- Progressive Resolution Training strategy
- Skeleton-based road graph generation
- OpenStreetMap graph integration
- Graph topology comparison
- Automatic accessibility estimation
- Test-Time Augmentation (TTA)
- Threshold optimization
- Validation on multiple real disaster scenarios

---

# 🌍 Real Disaster Case Studies

The framework was evaluated using real-world disaster imagery collected from different disaster events.

| Disaster | Country | Year |
|----------|----------|------|
| Hatay Earthquake | Türkiye | 2023 |
| Derna Flood | Libya | 2023 |
| Noto Peninsula Earthquake | Japan | 2024 |

---

# ✨ Features

- 🚀 SegFormer-B2 semantic segmentation
- 🛰️ UAV and satellite image processing
- 🕸️ Skeleton extraction
- 📍 Road graph generation
- 🗺️ OpenStreetMap comparison
- 📈 Progressive Resolution Training
- 🔍 Test-Time Augmentation
- 🎯 Threshold Optimization
- 📊 Quantitative evaluation
- 🌍 Multiple disaster case studies

---

# 🏗️ Framework Pipeline

The proposed framework consists of six major stages.

```text
Post-disaster Image
        │
        ▼
SegFormer-B2 Road Segmentation
        │
        ▼
Skeleton Extraction
        │
        ▼
Road Graph Generation
        │
        ▼
OpenStreetMap Graph
        │
        ▼
Graph Comparison
        │
        ▼
Road Accessibility Assessment
```

The segmented road network is transformed into a graph representation and compared with OpenStreetMap to identify disconnected road segments caused by disaster-related damage.

---

# 📂 Repository Structure

```text
post-disaster-road-accessibility/
│
├── assets/
├── cache/
├── data/
├── docs/
├── experiments/
├── models/
├── notebooks/
├── outputs/
├── scripts/
│
├── environment.yml
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```
---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/rukiyeasllan/post-disaster-road-accessibility.git

cd post-disaster-road-accessibility
```

---

## Create Conda Environment

```bash
conda env create -f environment.yml

conda activate post-disaster-road-accessibility
```

Alternatively,

```bash
pip install -r requirements.txt
```

---

# 📦 Dependencies

Main libraries used in this project include:

- PyTorch
- Transformers
- Segmentation Models PyTorch
- OpenCV
- Rasterio
- NetworkX
- OSMnx
- Shapely
- PyProj
- NumPy
- Matplotlib
- Pillow

---

# 🛰️ Dataset

The framework was developed using both benchmark datasets and real disaster imagery.

## Benchmark Dataset

- DeepGlobe Road Extraction Dataset

## Real Disaster Images

- Hatay Earthquake (Türkiye)
- Derna Flood (Libya)
- Noto Peninsula Earthquake (Japan)

Sample images are provided inside the **data/** directory.

---

# 🤖 Model

The proposed framework uses **SegFormer-B2** as the backbone semantic segmentation model.

### Training Configuration

| Parameter | Value |
|-----------|------:|
| Backbone | SegFormer-B2 |
| Framework | PyTorch |
| Optimizer | AdamW |
| Loss | BCE + Dice Loss |
| Task | Binary Road Segmentation |

---

# 📈 Progressive Resolution Training

Instead of training at a fixed image resolution, the model is progressively fine-tuned.

| Stage | Resolution | Epochs |
|--------|-----------:|-------:|
| Stage 1 | 512×512 | 30 |
| Stage 2 | 640×640 | 30 |
| Stage 3 | 768×768 | 20 |

This strategy improves segmentation quality while maintaining efficient training.

---

# 🚀 Additional Performance Improvements

Several techniques were incorporated to improve segmentation performance.

- Progressive Resolution Training
- Test-Time Augmentation (TTA)
- Threshold Optimization
- Graph-based Connectivity Analysis
- OpenStreetMap Integration

---

# ▶️ Running the Framework

Example pipeline execution:

```bash
python scripts/pipeline.py \
    --image data/hatay_defne_crop2.jpg \
    --lat <latitude> \
    --lon <longitude> \
    --model models/best_model.pth
```

The pipeline automatically performs:

1. Road Segmentation
2. Skeleton Extraction
3. Graph Generation
4. OpenStreetMap Extraction
5. Graph Comparison
6. Accessibility Analysis
7. Visualization Generation

---

# 📁 Output Files

Example outputs generated by the framework include:

- Binary segmentation mask
- Skeleton image
- Road graph visualization
- Accessibility map
- Damaged road visualization

Example results are available inside the **outputs/** directory.
---

# 📊 Experimental Results

The proposed framework was evaluated using the DeepGlobe Road Extraction dataset and validated on multiple real-world disaster scenarios.

## Quantitative Performance

| Metric | Score |
|---------|-------:|
| IoU | **70.18%** |
| F1-score | **82.47%** |
| Precision | **76.45%** |
| Recall | **84.82%** |

The proposed framework achieved competitive segmentation performance while additionally providing graph-based accessibility analysis, which is not available in conventional segmentation-only methods.

---

# 📈 Training Performance

The training process employed progressive resolution learning and additional optimization strategies.

The repository contains:

- Training history
- Validation loss
- Threshold optimization results
- Test-Time Augmentation (TTA) evaluation

These files are available inside the **models/** directory.

---

# 🌍 Real Disaster Examples

The framework was tested on several real disaster scenarios.

## 🇹🇷 Hatay Earthquake

- Post-earthquake UAV imagery
- Road accessibility estimation
- Graph-based connectivity analysis

## 🇱🇾 Derna Flood

- Flood-damaged transportation network
- Bridge accessibility analysis
- OSM graph comparison

## 🇯🇵 Noto Peninsula Earthquake

- Satellite imagery analysis
- Road disruption detection
- Accessibility estimation

Example visualizations are available in the **outputs/** directory.

---

# 📂 Repository Contents

| Directory | Description |
|------------|-------------|
| `scripts/` | Core implementation |
| `experiments/` | Training and evaluation scripts |
| `outputs/` | Example output visualizations |
| `models/` | Training history and evaluation results |
| `data/` | Sample disaster images |
| `assets/` | Figures used in the documentation |

---

# 🔬 Research Highlights

This framework integrates several computer vision and geospatial analysis techniques into a single end-to-end pipeline.

Main components include:

- SegFormer-B2 semantic segmentation
- Skeleton extraction
- Graph generation
- OpenStreetMap integration
- Topological graph comparison
- Accessibility estimation
- Progressive Resolution Training
- Test-Time Augmentation

---

# 📚 Citation

If you use this repository in your research, please cite the corresponding publication.

```bibtex
Citation information will be updated after publication.
```

Citation metadata is also available in:

```
CITATION.cff
```

---

# 📄 License

This project is released under the **MIT License**.

See the LICENSE file for more information.

---

# 👩‍💻 Author

**Rukiye Aslan**

Software Engineering Student

Fırat University

---

# 🙏 Acknowledgements

The following open-source projects and datasets made this work possible.

- DeepGlobe Road Extraction Dataset
- OpenStreetMap
- Hugging Face Transformers
- SegFormer
- PyTorch
- NetworkX
- Rasterio
- OSMnx

---

⭐ If you find this project useful, consider giving the repository a star.
