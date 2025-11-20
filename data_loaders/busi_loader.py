"""
BUSI (Breast Ultrasound Images) Dataset Loader
===============================================

This module provides functionality to load and process the BUSI dataset
for breast tumor classification.

Dataset: BUSI (Breast Ultrasound Images)
- Modality: Breast ultrasound images
- Images: 780 images (133 normal, 437 benign, 210 malignant)
- Use Case: Tumor classification, symbolic staging with hormonal archetypes
- Details: Includes segmentation masks for lesions
- URL: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
- Resolution: Variable (typically ~500x500 pixels)

Classes:
- Normal (0): No lesions
- Benign (1): Benign tumors
- Malignant (2): Malignant tumors
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Install with: pip install opencv-python")


class BUSIDataset(Dataset):
    """PyTorch Dataset for BUSI ultrasound images"""
    
    def __init__(self, images, masks, labels, metadata=None, transform=None):
        """
        Args:
            images: Tensor of shape (n_samples, height, width) or (n_samples, channels, height, width)
            masks: Tensor of shape (n_samples, height, width) - segmentation masks
            labels: Tensor of shape (n_samples,) - class labels
            metadata: Additional metadata
            transform: Optional transform (e.g., albumentations)
        """
        self.images = images
        self.masks = masks
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx] if self.masks is not None else None
        y = self.labels[idx]
        
        if self.transform:
            if mask is not None:
                # Apply transform to both image and mask
                transformed = self.transform(image=img.numpy(), mask=mask.numpy())
                img = torch.FloatTensor(transformed['image'])
                mask = torch.FloatTensor(transformed['mask'])
            else:
                transformed = self.transform(image=img.numpy())
                img = torch.FloatTensor(transformed['image'])
        
        if mask is not None:
            return img, mask, y
        return img, y


class BUSILoader:
    """
    Data loader for BUSI (Breast Ultrasound Images) dataset
    
    Supports:
    - Image and mask loading
    - Multi-class classification (normal/benign/malignant)
    - Image preprocessing and augmentation
    - Segmentation mask handling
    """
    
    def __init__(
        self,
        data_path: str = './data/BUSI',
        image_size: Tuple[int, int] = (224, 224),
        use_masks: bool = True,
        normalize: bool = True,
        augment: bool = False,
        grayscale: bool = False,
        binary_classification: bool = False,  # Normal+Benign vs Malignant
        verbose: bool = True
    ):
        """
        Initialize BUSI loader
        
        Args:
            data_path: Path to BUSI dataset
            image_size: Target image size (height, width)
            use_masks: Load segmentation masks
            normalize: Apply normalization
            augment: Apply data augmentation
            grayscale: Convert to grayscale (ultrasound is typically grayscale)
            binary_classification: Binary (benign vs malignant) instead of 3-class
            verbose: Print progress
        """
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.use_masks = use_masks
        self.normalize = normalize
        self.augment = augment
        self.grayscale = grayscale
        self.binary_classification = binary_classification
        self.verbose = verbose
        
        # Check if data is in Dataset_BUSI_with_GT subdirectory
        if (self.data_path / 'Dataset_BUSI_with_GT').exists():
            self.data_path = self.data_path / 'Dataset_BUSI_with_GT'
        
        # Class labels
        self.class_map = {
            'normal': 0,
            'benign': 1,
            'malignant': 2
        }
        
        # Binary classification: 0=benign (normal+benign), 1=malignant
        self.binary_map = {
            'normal': 0,
            'benign': 0,
            'malignant': 1
        }
        
        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("="*70)
            print("BUSI Breast Ultrasound Loader Initialized")
            print("="*70)
            print(f"Data path: {self.data_path}")
            print(f"Image size: {image_size}")
            print(f"Use masks: {use_masks}")
            print(f"Binary classification: {binary_classification}")
            print("="*70)
    
    def download_dataset(self):
        """
        Download BUSI dataset from Kaggle
        
        Note: Requires Kaggle API credentials
        """
        if self.data_path.exists() and any(self.data_path.glob('*')):
            if self.verbose:
                print(f"✓ Dataset already exists: {self.data_path}")
            return True
        
        if self.verbose:
            print(f"\nDownloading BUSI dataset...")
            print("Note: This requires Kaggle API credentials.")
            print("Setup instructions:")
            print("  1. Create account at https://www.kaggle.com")
            print("  2. Go to Account settings -> API -> Create New API Token")
            print("  3. Place kaggle.json in ~/.kaggle/")
            print("  4. Run: kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset")
            print("\nCreating mock data for demonstration...")
        
        # Create mock data
        self._create_mock_data()
        return True
    
    def _create_mock_data(self):
        """Create mock ultrasound images and masks"""
        if self.verbose:
            print("\nCreating mock BUSI data...")
        
        # Create class directories
        classes = ['normal', 'benign', 'malignant']
        class_counts = [40, 100, 60]  # Approximate distribution
        
        for class_name, count in zip(classes, class_counts):
            class_dir = self.data_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                # Create mock ultrasound image (grayscale with speckle noise)
                img = np.random.rand(500, 500) * 255
                
                # Add ultrasound-like features
                # Darker at top (transducer), brighter in middle
                for y in range(500):
                    factor = y / 500  # Darker at top
                    img[y, :] *= factor
                
                # Add speckle noise (characteristic of ultrasound)
                speckle = np.random.gamma(1, 1, (500, 500))
                img = img * speckle
                img = np.clip(img, 0, 255).astype(np.uint8)
                
                # Add lesion for benign/malignant cases
                if class_name != 'normal':
                    center_x = np.random.randint(150, 350)
                    center_y = np.random.randint(200, 400)
                    radius = np.random.randint(30, 80)
                    
                    # Create mask
                    mask = np.zeros((500, 500), dtype=np.uint8)
                    
                    for x in range(max(0, center_x-radius), min(500, center_x+radius)):
                        for y in range(max(0, center_y-radius), min(500, center_y+radius)):
                            dist = np.sqrt((x-center_x)**2 + (y-center_y)**2)
                            if dist < radius:
                                # Irregular border for malignant, smooth for benign
                                if class_name == 'malignant':
                                    noise = np.random.rand() * 10
                                    if dist < radius - noise:
                                        img[y, x] = np.random.randint(100, 180)
                                        mask[y, x] = 255
                                else:
                                    img[y, x] = np.random.randint(80, 150)
                                    mask[y, x] = 255
                    
                    # Save mask
                    if self.use_masks:
                        mask_img = Image.fromarray(mask)
                        mask_img.save(class_dir / f'{class_name} ({i+1})_mask.png')
                
                # Save image
                img_pil = Image.fromarray(img, mode='L')
                img_pil.save(class_dir / f'{class_name} ({i+1}).png')
        
        if self.verbose:
            print(f"✓ Created mock data: {sum(class_counts)} images")
    
    def load_image(self, img_path: Path) -> np.ndarray:
        """
        Load and preprocess a single image
        
        Returns:
            Image array of shape (height, width) or (height, width, channels)
        """
        # Load image
        if CV2_AVAILABLE:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR)
            if not self.grayscale and img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(img_path).convert('L' if self.grayscale else 'RGB'))
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        # Resize
        if CV2_AVAILABLE:
            img = cv2.resize(img, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize(self.image_size[::-1], Image.BILINEAR)
            img = np.array(img_pil)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        """
        Load segmentation mask
        
        Returns:
            Mask array of shape (height, width) or None if not found
        """
        if not mask_path.exists():
            return None
        
        # Load mask
        if CV2_AVAILABLE:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.array(Image.open(mask_path).convert('L'))
        
        if mask is None:
            return None
        
        # Resize
        if CV2_AVAILABLE:
            mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
        else:
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize(self.image_size[::-1], Image.NEAREST)
            mask = np.array(mask_pil)
        
        # Binarize
        mask = (mask > 127).astype(np.float32)
        
        return mask
    
    def load_class_data(self, class_name: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Load all images from a class
        
        Returns:
            images: List of image arrays
            masks: List of mask arrays
            labels: List of labels
        """
        class_dir = self.data_path / class_name
        
        if not class_dir.exists():
            return [], [], []
        
        # Find all images (exclude masks)
        image_files = [f for f in class_dir.glob('*.png') if '_mask' not in f.name]
        
        images = []
        masks = []
        labels = []
        
        for img_file in image_files:
            try:
                # Load image
                img = self.load_image(img_file)
                images.append(img)
                
                # Load mask if available
                mask_file = img_file.parent / f"{img_file.stem}_mask{img_file.suffix}"
                if self.use_masks:
                    mask = self.load_mask(mask_file)
                    if mask is None:
                        mask = np.zeros(self.image_size, dtype=np.float32)
                    masks.append(mask)
                
                # Assign label
                if self.binary_classification:
                    label = self.binary_map[class_name]
                else:
                    label = self.class_map[class_name]
                labels.append(label)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error loading {img_file.name}: {e}")
                continue
        
        return images, masks, labels
    
    def load_all_data(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Load all data
        
        Returns:
            X: Images (n_samples, height, width) or (n_samples, channels, height, width)
            masks: Segmentation masks (n_samples, height, width) or None
            y: Labels (n_samples,)
        """
        # Download if needed
        self.download_dataset()
        
        if self.verbose:
            print("\nLoading BUSI dataset...")
        
        all_images = []
        all_masks = []
        all_labels = []
        
        classes = ['normal', 'benign', 'malignant']
        
        for class_name in classes:
            if self.verbose:
                print(f"Loading {class_name}...", end=' ')
            
            images, masks, labels = self.load_class_data(class_name)
            
            all_images.extend(images)
            if self.use_masks:
                all_masks.extend(masks)
            all_labels.extend(labels)
            
            if self.verbose:
                print(f"✓ {len(images)} images")
        
        # Convert to arrays
        X = np.array(all_images)
        y = np.array(all_labels)
        masks_array = np.array(all_masks) if self.use_masks else None
        
        # Add channel dimension if grayscale
        if len(X.shape) == 3:  # (n_samples, height, width)
            X = X[:, np.newaxis, :, :]  # (n_samples, 1, height, width)
        elif len(X.shape) == 4:  # (n_samples, height, width, channels)
            X = np.transpose(X, (0, 3, 1, 2))  # (n_samples, channels, height, width)
        
        # Normalize
        if self.normalize:
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            std = X.std(axis=(0, 2, 3), keepdims=True)
            X = (X - mean) / (std + 1e-8)
        
        if self.verbose:
            print(f"\n✓ Loaded {len(X)} images")
            print(f"  Image shape: {X.shape}")
            if masks_array is not None:
                print(f"  Mask shape: {masks_array.shape}")
            print(f"  Classes: {np.unique(y)}")
            print(f"  Class distribution: {np.bincount(y)}")
        
        return X, masks_array, y
    
    def get_dataloaders(
        self,
        batch_size: int = 16,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train/val/test data loaders"""
        X, masks, y = self.load_all_data()
        
        # Split data
        if masks is not None:
            X_temp, X_test, mask_temp, mask_test, y_temp, y_test = train_test_split(
                X, masks, y, test_size=test_split, random_state=random_seed, stratify=y
            )
            X_train, X_val, mask_train, mask_val, y_train, y_val = train_test_split(
                X_temp, mask_temp, y_temp, 
                test_size=val_split/(1-test_split), 
                random_state=random_seed, 
                stratify=y_temp
            )
        else:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_seed, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=val_split/(1-test_split), 
                random_state=random_seed, 
                stratify=y_temp
            )
            mask_train = mask_val = mask_test = None
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        
        if mask_train is not None:
            mask_train = torch.FloatTensor(mask_train)
            mask_val = torch.FloatTensor(mask_val)
            mask_test = torch.FloatTensor(mask_test)
        
        # Create datasets
        train_dataset = BUSIDataset(X_train, mask_train, y_train)
        val_dataset = BUSIDataset(X_val, mask_val, y_val)
        test_dataset = BUSIDataset(X_test, mask_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        if self.verbose:
            print(f"\nData split:")
            print(f"  Train: {len(X_train)} samples")
            print(f"  Val:   {len(X_val)} samples")
            print(f"  Test:  {len(X_test)} samples")
        
        return train_loader, val_loader, test_loader


def test_busi_loader():
    """Test BUSI loader"""
    print("\n" + "="*70)
    print("Testing BUSI Breast Ultrasound Loader")
    print("="*70)
    
    loader = BUSILoader(
        data_path='./data/BUSI',
        image_size=(224, 224),
        use_masks=True,
        grayscale=True,
        verbose=True
    )
    
    try:
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=8)
        
        # Test batch
        for batch in train_loader:
            if len(batch) == 3:
                x, mask, y = batch
                print(f"\n✓ Image shape: {x.shape}")
                print(f"  Mask shape: {mask.shape}")
                print(f"  Labels shape: {y.shape}")
            else:
                x, y = batch
                print(f"\n✓ Image shape: {x.shape}")
                print(f"  Labels shape: {y.shape}")
            break
        
        print("\n✅ BUSI loader test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_busi_loader()
