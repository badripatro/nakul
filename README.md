# Quick Reference Guide: Medical Imaging Data Loaders

## 🚀 One-Line Setup

```bash
# 1. Download datasets
./download_medical_datasets.sh

# 2. Test all loaders
python test_medical_loaders.py

# 3. Start using!
```

---

## 📦 5 Datasets Summary

| Dataset | Code | Task | Output Shape | Classes |
|---------|------|------|--------------|---------|
| **BCI-IV-2a** | `bci_loader` | Motor Imagery | `(batch, channels, time)` | 4 |
| **FACED** | `faced_loader` | Emotion EEG | `(batch, channels, time)` | 9 |
| **OpenNeuro** | `openneuro_loader` | fMRI Task | `(batch, features)` | 3 |
| **SeizeIT1** | `seizeit_loader` | Seizure | `(batch, channels, time)` | 2 |
| **BUSI** | `busi_loader` | Breast Tumor | `(batch, C, H, W)` | 3 |

---

## 💻 Code Templates

### Template 1: BCI-IV-2a (Motor Imagery EEG)

```python
from data_loaders.bci_loader import BCIDataLoader

loader = BCIDataLoader(data_path='./data', subjects=list(range(1, 10)))
X, y, subjects_data = loader.load_data()

# Split and create dataloaders
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

for batch_x, batch_y in train_loader:
    # batch_x: (batch, 22, 1000) - 22 channels, 1000 timepoints
    output = model(batch_x)
```

### Template 2: FACED (Emotion Recognition EEG)

```python
from data_loaders.faced_loader import FACEDDataLoader

loader = FACEDDataLoader(data_path='./data/Processed_data', window_size=2.0)
train, val, test = loader.get_dataloaders(batch_size=32)

for batch_x, batch_y in train:
    # batch_x: (batch, 32, 500) - 32 channels, 500 timepoints
    # batch_y: emotion labels (0-8 for 9 emotions)
    output = model(batch_x)
```

### Template 3: OpenNeuro (fMRI Task)

```python
from data_loaders.openneuro_loader import OpenNeuroLoader

loader = OpenNeuroLoader(data_path='./data/openneuro')
train, val, test = loader.get_dataloaders(batch_size=16)

for features, labels, subjects in train:
    # features: connectivity matrix or ROI features
    output = model(features)
```

### Template 4: SeizeIT1 (EEG-fMRI Seizure)

```python
from data_loaders.seizeit_loader import SeizeITLoader

loader = SeizeITLoader(data_path='./data/seizeit', include_fmri=True)
train, val, test = loader.get_dataloaders(batch_size=32)

for batch in train:
    eeg = batch[0]['eeg']  # (batch, 32, time)
    fmri = batch[0]['fmri']  # (batch, features)
    labels = batch[1]  # Ictal vs interictal
```

### Template 5: BUSI (Breast Ultrasound)

```python
from data_loaders.busi_loader import BUSILoader

loader = BUSILoader(data_path='./data/BUSI', use_masks=True)
train, val, test = loader.get_dataloaders(batch_size=16)

for images, masks, labels in train:
    # images: ultrasound images (batch, 1, 224, 224)
    # masks: segmentation masks
    # labels: normal/benign/malignant
    output = model(images)
```

---

## 🎯 Common Parameters

```python
# All loaders support:
loader = AnyLoader(
    data_path='./data/dataset_name',  # Where data is stored
    verbose=True,                      # Print progress
)

# Get data loaders:
train, val, test = loader.get_dataloaders(
    batch_size=32,         # Batch size
    val_split=0.15,        # Validation split
    test_split=0.15,       # Test split
    random_seed=42         # Random seed
)
```

---

## 📊 Dataset Sizes & Requirements

| Dataset | Size | Download Time* | RAM Required | GPU RAM |
|---------|------|----------------|--------------|---------|
| OpenNeuro | 50-100 GB | 1-3 hours | 8 GB | 4 GB |
| SeizeIT1 | 30-50 GB | 30-90 min | 8 GB | 4 GB |
| BUSI | 180 MB | 1-2 min | 4 GB | 2 GB |

*With good internet connection

---

## ⚡ Performance Tips

```python
# 1. Use multiple workers
DataLoader(..., num_workers=4, pin_memory=True)

# 2. Cache preprocessed data
X, y, _ = loader.load_all_data()
np.savez('cached.npz', X=X, y=y)

# 3. Use appropriate batch sizes
# - 2D images: 16-32
# - 3D volumes: 4-8
# - Videos: 2-8
# - Features: 32-64

# 4. GPU memory optimization
torch.cuda.empty_cache()  # Clear cache
model.half()  # Use FP16
```

---

## 🔍 Debugging

```python
# Check data shape
for batch in train_loader:
    print("Batch shapes:")
    if isinstance(batch, tuple):
        for i, item in enumerate(batch):
            if torch.is_tensor(item):
                print(f"  Item {i}: {item.shape}")
    break

# Check data range
print(f"Min: {batch[0].min()}, Max: {batch[0].max()}")

# Check class distribution
labels = []
for _, y in train_loader:
    labels.extend(y.numpy())
print(f"Class distribution: {np.bincount(labels)}")
```

---

## 🐛 Common Issues

### Issue 1: "Dataset not found"
```bash
# Solution: Download the dataset first
./download_medical_datasets.sh dataset_name
```

### Issue 2: "Out of memory"
```python
# Solution: Reduce batch size
train_loader = DataLoader(..., batch_size=8)  # Instead of 32
```

### Issue 3: "CUDA out of memory"
```python
# Solution: Use gradient accumulation
for i, (x, y) in enumerate(train_loader):
    loss = model(x, y) / 4  # Accumulate over 4 steps
    loss.backward()
    if (i + 1) % 4 == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue 4: "Slow data loading"
```python
# Solution: Increase workers and prefetch
DataLoader(..., num_workers=8, prefetch_factor=2)
```

---

## 📝 Minimal Training Example

```python
import torch
import torch.nn as nn
from data_loaders.busi_loader import BUSILoader

# 1. Load data
loader = BUSILoader(data_path='./data/BUSI')
train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=16)

# 2. Define model
model = nn.Sequential(
    nn.Conv2d(1, 32, 3), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Linear(32, 3)
).cuda()

# 3. Training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        output = model(images)
        loss = criterion(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## 🔗 Quick Links

- **Full Documentation:** `MEDICAL_LOADERS_README.md`
- **Test Script:** `python test_medical_loaders.py`
- **Download Script:** `./download_medical_datasets.sh`
- **Issue Tracker:** [GitHub Issues](#)

---

## 📞 Get Help

```bash
# Test specific loader
python test_medical_loaders.py --loader busi

# Run benchmark
python test_medical_loaders.py --benchmark

# Check installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

---

## ✅ Checklist

Before training:
- [ ] Downloaded dataset
- [ ] Tested loader with `test_medical_loaders.py`
- [ ] Verified batch shapes
- [ ] Checked class distribution
- [ ] Set appropriate batch size for GPU memory
- [ ] Enabled multi-worker loading

---

## 🎓 Learn More

- OpenNeuro: https://openneuro.org/
- TCIA: https://www.cancerimagingarchive.net/
- Kaggle Datasets: https://www.kaggle.com/datasets
- PyTorch DataLoader: https://pytorch.org/docs/stable/data.html

---

**Last Updated:** November 2025  
**Version:** 1.0.0
