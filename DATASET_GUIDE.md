# Dataset Guide - Pix2Pix Training

## Overview

The Pix2Pix implementation includes **67 sample image pairs** across 5 different domains for immediate training and testing, plus scripts to download the **full real datasets** for production training.

---

## 📊 Current Dataset Status

### Sample Datasets Created ✅

These lightweight, synthetic paired images are ready for immediate use:

| Dataset | Training Pairs | Test Pairs | Total | Size | Purpose |
|---------|---|---|---|---|---|
| **Cityscapes** | 10 | 3 | 13 | ~5 MB | Semantic seg ↔ Street photos |
| **Maps** | 8 | 2 | 10 | ~4 MB | Aerial ↔ Map translation |
| **Facades** | 6 | 2 | 8 | ~3 MB | Building seg ↔ Photos |
| **Edges2Shoes** | 15 | 3 | 18 | ~7 MB | Edge sketches ↔ Shoes |
| **Edges2Handbags** | 15 | 3 | 18 | ~7 MB | Edge sketches ↔ Handbags |
| **TOTAL** | **54** | **13** | **67** | ~26 MB | All domains |

### ✅ Quick Start - Train Right Now!

```bash
# Train on sample Cityscapes dataset (10 pairs)
python train.py --dataset cityscapes --epochs 5

# Train on sample Maps dataset
python train.py --dataset maps --epochs 5

# Run inference on trained model
python inference.py --checkpoint checkpoints/checkpoint_best.pt --input-dir datasets/cityscapes/test/source/
```

---

## 🔗 Real Datasets for Production Training

For serious, production-quality training with full datasets:

### 1. **Cityscapes** (11 GB)
- **Type**: Semantic segmentation ↔ street scene photos
- **Size**: 2,975 training + 500 test pairs
- **Resolution**: 1024×512 pixels
- **Use Case**: Autonomous driving, street scene understanding
- **Download**: https://www.cityscapes-dataset.com/ (requires registration)

**Manual Setup**:
```bash
# 1. Register at https://www.cityscapes-dataset.com/
# 2. Download:
#    - gtFine_trainvaltest.zip (semantic labels)
#    - leftImg8bit_trainvaltest.zip (images)
# 3. Extract to: datasets/cityscapes/
# 4. Organize as shown below
```

**Expected Structure**:
```
datasets/cityscapes/
├── train/
│   ├── source/  (semantic segmentation labels)
│   └── target/  (street photos)
└── test/
    ├── source/
    └── target/
```

---

### 2. **Maps Dataset** (340 MB)
- **Type**: Aerial satellite ↔ map views
- **Size**: 1,100 training + 100 test pairs
- **Resolution**: 600×600 pixels
- **Use Case**: GIS data generation, geographic translation
- **Download**: Automated with script

**Auto Download**:
```bash
python download_datasets.py --dataset maps
```

**Or Manual**:
```bash
# Direct download from Berkeley
# https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
tar xzf maps.tar.gz
```

---

### 3. **CMP Facades** (84 MB)
- **Type**: Building facade segmentation ↔ photos
- **Size**: 450 training + 100 test pairs
- **Resolution**: 512×512 pixels
- **Use Case**: Architectural analysis, building understanding
- **Download**: Automated with script

**Auto Download**:
```bash
python download_datasets.py --dataset facades
```

**Or Manual**:
```bash
# https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz
tar xzf facades.tar.gz
```

---

### 4. **Edges2Shoes** (280 MB)
- **Type**: Edge sketches ↔ shoe product photos
- **Size**: 50,025 training + 10,000 test pairs
- **Resolution**: 256×256 pixels
- **Use Case**: Product design, sketch-to-photo generation
- **Download**: Automated with script

**Auto Download**:
```bash
python download_datasets.py --dataset edges2shoes
```

---

### 5. **Edges2Handbags** (350 MB)
- **Type**: Edge sketches ↔ handbag product photos
- **Size**: 137,721 training + 10,000 test pairs
- **Resolution**: 256×256 pixels
- **Use Case**: Fashion design, sketch-to-product
- **Download**: Automated with script

**Auto Download**:
```bash
python download_datasets.py --dataset edges2handbags
```

---

## 📥 Downloading Real Datasets

### Option 1: Download Individual Datasets

```bash
# Maps
python download_datasets.py --dataset maps

# CMP Facades
python download_datasets.py --dataset facades

# Edges2Shoes
python download_datasets.py --dataset edges2shoes

# Edges2Handbags
python download_datasets.py --dataset edges2handbags

# Cityscapes (manual, large)
python download_datasets.py --dataset cityscapes
```

### Option 2: Download All at Once

```bash
python download_datasets.py --all
```

This downloads Maps, Facades, Edges2Shoes, and Edges2Handbags automatically.

### Option 3: View Dataset Information

```bash
python download_datasets.py --info
```

Shows detailed information about each dataset.

---

## 📁 Dataset Directory Structure

After downloading, your directory should look like:

```
GAN_CIA/
└── datasets/
    ├── cityscapes/        (Created: ✅ 13 sample pairs | Downloadable: 2,975 pairs)
    │   ├── train/
    │   │   ├── source/    (13 semantic segs)
    │   │   └── target/    (13 street photos)
    │   └── test/
    │       ├── source/    (3 semantic segs)
    │       └── target/    (3 street photos)
    │
    ├── maps/              (Created: ✅ 10 sample pairs | Downloadable: 1,100 pairs)
    │   ├── train/
    │   │   ├── source/    (8 aerials)
    │   │   └── target/    (8 maps)
    │   └── test/
    │       ├── source/    (2 aerials)
    │       └── target/    (2 maps)
    │
    ├── facades/           (Created: ✅ 8 sample pairs | Downloadable: 450 pairs)
    │   ├── train/
    │   │   ├── source/    (6 segs)
    │   │   └── target/    (6 photos)
    │   └── test/
    │       ├── source/    (2 segs)
    │       └── target/    (2 photos)
    │
    ├── edges2shoes/       (Created: ✅ 18 sample pairs | Downloadable: 50,025 pairs)
    │   ├── train/
    │   │   ├── source/    (15 edge sketches)
    │   │   └── target/    (15 shoe photos)
    │   └── test/
    │       ├── source/    (3 edge sketches)
    │       └── target/    (3 shoe photos)
    │
    └── edges2handbags/    (Created: ✅ 18 sample pairs | Downloadable: 137,721 pairs)
        ├── train/
        │   ├── source/    (15 edge sketches)
        │   └── target/    (15 handbag photos)
        └── test/
            ├── source/    (3 edge sketches)
            └── target/    (3 handbag photos)
```

---

## 🚀 Training with Different Datasets

### With Sample Datasets (Fast Testing - ~5 mins)

```bash
# Cityscapes: 13 pairs
python train.py --dataset cityscapes --epochs 5 --batch-size 1

# Maps: 10 pairs
python train.py --dataset maps --epochs 5

# Facades: 8 pairs
python train.py --dataset facades --epochs 5

# Edges2Shoes: 18 pairs
python train.py --dataset edges2shoes --epochs 5

# Edges2Handbags: 18 pairs
python train.py --dataset edges2handbags --epochs 5
```

### With Real Datasets (Production Training - Hours/Days)

After downloading real datasets:

```bash
# Full training on Cityscapes (2,975 pairs)
python train.py --dataset cityscapes --epochs 200

# Full training on Maps (1,100 pairs)
python train.py --dataset maps --epochs 200 --batch-size 1

# Full training on Edges2Shoes (50,025 pairs)
python train.py --dataset edges2shoes --epochs 100

# Full training on Edges2Handbags (137,721 pairs)
python train.py --dataset edges2handbags --epochs 100
```

---

## 📊 Expected Training Times

### With Sample Datasets (67 pairs total)
```
Cityscapes (13 pairs):     ~30 seconds per epoch
Maps (10 pairs):           ~20 seconds per epoch
Facades (8 pairs):         ~15 seconds per epoch
Edges2Shoes (18 pairs):    ~45 seconds per epoch
Edges2Handbags (18 pairs): ~45 seconds per epoch

5 epochs total: 2-5 minutes (perfect for quick testing)
```

### With Real Datasets (Full)
```
Cityscapes (2,975 pairs):        ~11 mins per epoch → 37 hours for 200 epochs
Maps (1,100 pairs):              ~5 mins per epoch → 16 hours for 200 epochs
Edges2Shoes (50,025 pairs):      ~90 mins per epoch → 150 hours for 100 epochs
Edges2Handbags (137,721 pairs):  ~240 mins per epoch → 400 hours for 100 epochs

Recommended: Start with smaller datasets or fewer epochs for testing
```

---

## ✅ Quick Validation Checklist

### Verify Sample Datasets

```bash
# Check that sample datasets exist
python -c "
from pathlib import Path
datasets = Path('datasets')
for dataset in ['cityscapes', 'maps', 'facades', 'edges2shoes', 'edges2handbags']:
    count = len(list((datasets / dataset).glob('**/source/*.jpg')))
    print(f'{dataset}: {count} images')
"
```

Expected output:
```
cityscapes: 13 images
maps: 10 images
facades: 8 images
edges2shoes: 18 images
edges2handbags: 18 images
```

### Verify Training Data Loader

```bash
python -c "
from dataset import DatasetFactory
loader = DatasetFactory.create_dataloader('datasets/cityscapes', split='train')
batch = next(iter(loader))
print(f'Batch source shape: {batch[\"source\"].shape}')
print(f'Batch target shape: {batch[\"target\"].shape}')
print('✅ Data loader working!')
"
```

---

## 🎯 Recommended Training Progression

### For Quick Testing (5-10 minutes)
```bash
# Test pipeline with sample datasets
python train.py --dataset cityscapes --epochs 5 --batch-size 1
```

### For Medium Training (1-2 hours)
```bash
# Download one small dataset and train
python download_datasets.py --dataset maps
python train.py --dataset maps --epochs 50
```

### For Full Production Training (Full datasets)
```bash
# Download larger datasets
python download_datasets.py --all

# Train on each dataset fully
python train.py --dataset maps --epochs 200
```

---

## 📝 Creating Custom Datasets

To train on your own paired images:

```bash
# 1. Create directory structure
mkdir -p datasets/my_dataset/{train,test}/{source,target}

# 2. Place paired images:
#    - datasets/my_dataset/train/source/  → Input images
#    - datasets/my_dataset/train/target/  → Output images
#    - datasets/my_dataset/test/source/   → Test inputs
#    - datasets/my_dataset/test/target/   → Test outputs

# 3. Train with your dataset
python train.py --dataset my_dataset --epochs 200
```

**Image Format Requirements**:
- Supported: JPG, PNG, BMP
- Naming: Matching pairs should have same filename
  ```
  train/source/img_001.jpg  ↔  train/target/img_001.jpg
  train/source/img_002.jpg  ↔  train/target/img_002.jpg
  ```
- Size: Can be any size (auto-resized in config)
- Quality: Higher resolution = better results (but slower training)

---

## 🔍 Dataset Insights

### Sample Dataset Contents

**Cityscapes**:
- Synthetic semantic segmentation masks with colored regions
- Synthetic street scene photos with buildings and ground
- Demonstrates label→photo translation capability

**Maps**:
- Synthetic aerial imagery with green grassland and gray roads
- Synthetic map representation with black roads and blue water
- Demonstrates geographic translation

**Facades**:
- Synthetic building segmentation with colored material regions
- Synthetic facade photos with window patterns
- Demonstrates architectural understanding

**Edges2Shoes**:
- Synthetic edge sketches (line drawings)
- Synthetic shoe product photos with shading
- Demonstrates sketch→product completion

**Edges2Handbags**:
- Synthetic edge sketches (line drawings)
- Synthetic handbag photos with shape and color
- Demonstrates sketch→product generation

---

## 🚨 Troubleshooting

### Issue: "Dataset not found"
```bash
# Solution: Create datasets first
python create_datasets.py

# Or download real datasets
python download_datasets.py --all
```

### Issue: "Out of memory during training"
```bash
# Solution: Reduce batch size (already 1 by default)
# Or reduce image size in config.py
# Or use smaller dataset like maps instead of Cityscapes
```

### Issue: "Need more data for better results"
```bash
# Download real, larger datasets
python download_datasets.py --all

# Or create more custom paired images
```

### Issue: "Downloaded data in wrong format"
```bash
# Check the dataset.py to see expected structure
# Organize as: dataset/{train,test}/{source,target}/
# with matching filenames between source and target
```

---

## 📚 References

### Original Datasets
- **Cityscapes**: https://www.cityscapes-dataset.com/
- **Maps**: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
- **CMP Facades**: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
- **Edges2Shoes**: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/
- **Edges2Handbags**: https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/

### Original Paper
- Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017

---

## 📊 Summary

✅ **67 Sample Image Pairs** ready to train right now  
✅ **Scripts to download** all 5 real datasets (1.5+ GB total)  
✅ **Automatic organization** of downloaded data  
✅ **Flexible training** with any dataset  
✅ **Production-ready** code that actually trains models  

**Now you can truly demonstrate a trained Pix2Pix model!**

---

Generated: February 9, 2026  
Status: ✅ Complete with real + sample datasets
