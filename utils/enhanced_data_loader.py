"""
Enhanced Data Loader with Proper Validation and Augmentation
Addresses data leakage and class imbalance issues
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Optional, Dict, List
import warnings

class EEGAugmentation:
    """EEG-specific data augmentation techniques"""
    
    def __init__(self, noise_factor: float = 0.05, time_shift_max: int = 10):
        self.noise_factor = noise_factor
        self.time_shift_max = time_shift_max
        
    def add_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """Add gaussian noise to EEG data"""
        noise = np.random.normal(0, self.noise_factor, data.shape)
        return data + noise
    
    def time_shift(self, data: np.ndarray) -> np.ndarray:
        """Apply random time shift"""
        if self.time_shift_max == 0:
            return data
            
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        if shift == 0:
            return data
            
        shifted_data = np.zeros_like(data)
        if shift > 0:
            shifted_data[:, shift:] = data[:, :-shift]
        else:
            shifted_data[:, :shift] = data[:, -shift:]
        return shifted_data
    
    def channel_dropout(self, data: np.ndarray, dropout_prob: float = 0.1) -> np.ndarray:
        """Randomly set some channels to zero"""
        if dropout_prob == 0:
            return data
            
        mask = np.random.random(data.shape[0]) > dropout_prob
        augmented_data = data.copy()
        augmented_data[~mask, :] = 0
        return augmented_data
    
    def amplitude_scaling(self, data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random amplitude scaling"""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale_factor
    
    def __call__(self, data: np.ndarray, apply_prob: float = 0.5) -> np.ndarray:
        """Apply random augmentations"""
        if np.random.random() > apply_prob:
            return data
            
        augmented = data.copy()
        
        # Apply random combination of augmentations
        if np.random.random() < 0.7:
            augmented = self.add_gaussian_noise(augmented)
        if np.random.random() < 0.3:
            augmented = self.time_shift(augmented)
        if np.random.random() < 0.2:
            augmented = self.channel_dropout(augmented)
        if np.random.random() < 0.4:
            augmented = self.amplitude_scaling(augmented)
            
        return augmented

class EnhancedEEGDataset(Dataset):
    """Enhanced EEG Dataset with augmentation and validation"""
    
    def __init__(self, 
                 data: np.ndarray, 
                 labels: np.ndarray,
                 subjects: np.ndarray,
                 augmentation: Optional[EEGAugmentation] = None,
                 normalize: bool = True,
                 transform_func: Optional[callable] = None):
        
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.subjects = subjects.astype(np.int64)
        self.augmentation = augmentation
        self.normalize = normalize
        self.transform_func = transform_func
        
        if self.normalize:
            self.scaler = StandardScaler()
            # Normalize across all samples and channels
            original_shape = self.data.shape
            reshaped_data = self.data.reshape(-1, original_shape[-1])
            normalized_data = self.scaler.fit_transform(reshaped_data)
            self.data = normalized_data.reshape(original_shape)
        
        self._validate_data()
    
    def _validate_data(self):
        """Validate data integrity"""
        # Check for NaN or infinite values
        if np.isnan(self.data).any():
            logging.warning("NaN values found in data")
        if np.isinf(self.data).any():
            logging.warning("Infinite values found in data")
            
        # Check data ranges
        data_min, data_max = self.data.min(), self.data.max()
        if abs(data_min) > 1000 or abs(data_max) > 1000:
            logging.warning(f"Large data values detected: min={data_min:.2f}, max={data_max:.2f}")
        
        # Check class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        logging.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
        
        # Check for class imbalance
        min_count, max_count = counts.min(), counts.max()
        if max_count / min_count > 3:
            logging.warning(f"Significant class imbalance detected: {max_count/min_count:.2f}:1")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx].copy()
        label = self.labels[idx]
        subject = self.subjects[idx]
        
        # Apply augmentation if specified
        if self.augmentation is not None:
            sample = self.augmentation(sample)
        
        # Apply custom transform if specified  
        if self.transform_func is not None:
            sample = self.transform_func(sample)
            
        # Ensure proper tensor format
        sample = torch.FloatTensor(sample)
        # For EEG data: (channels, time_points) - no extra dimension needed
            
        return sample, torch.LongTensor([label])[0], torch.LongTensor([subject])[0]

class StratifiedDataSplitter:
    """Stratified data splitter with subject-aware splitting"""
    
    def __init__(self, 
                 random_state: int = 42,
                 test_size: float = 0.2,
                 val_size: float = 0.2,
                 stratify_by_subject: bool = True):
        
        self.random_state = random_state
        self.test_size = test_size
        self.val_size = val_size
        self.stratify_by_subject = stratify_by_subject
    
    def split_data(self, 
                   data: np.ndarray, 
                   labels: np.ndarray, 
                   subjects: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split data into train/val/test sets with stratification
        
        Returns:
            Dictionary containing train/val/test splits with data, labels, subjects
        """
        
        if self.stratify_by_subject:
            return self._split_by_subject(data, labels, subjects)
        else:
            return self._split_by_samples(data, labels, subjects)
    
    def _split_by_subject(self, 
                         data: np.ndarray, 
                         labels: np.ndarray, 
                         subjects: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Split ensuring no subject appears in multiple splits"""
        
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        # Calculate number of subjects for each split
        n_test_subjects = max(1, int(n_subjects * self.test_size))
        n_val_subjects = max(1, int(n_subjects * self.val_size))
        n_train_subjects = n_subjects - n_test_subjects - n_val_subjects
        
        if n_train_subjects < 1:
            raise ValueError("Not enough subjects for proper splitting")
        
        # Randomly shuffle subjects
        np.random.seed(self.random_state)
        shuffled_subjects = np.random.permutation(unique_subjects)
        
        # Assign subjects to splits
        train_subjects = shuffled_subjects[:n_train_subjects]
        val_subjects = shuffled_subjects[n_train_subjects:n_train_subjects + n_val_subjects]
        test_subjects = shuffled_subjects[n_train_subjects + n_val_subjects:]
        
        # Create masks for each split
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects) 
        test_mask = np.isin(subjects, test_subjects)
        
        splits = {
            'train': {
                'data': data[train_mask],
                'labels': labels[train_mask],
                'subjects': subjects[train_mask]
            },
            'val': {
                'data': data[val_mask],
                'labels': labels[val_mask], 
                'subjects': subjects[val_mask]
            },
            'test': {
                'data': data[test_mask],
                'labels': labels[test_mask],
                'subjects': subjects[test_mask]
            }
        }
        
        self._validate_splits(splits)
        return splits
    
    def _split_by_samples(self, 
                         data: np.ndarray, 
                         labels: np.ndarray, 
                         subjects: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Split by samples with stratification by labels"""
        
        # First split: train+val vs test
        combined_labels = [f"{label}_{subject}" for label, subject in zip(labels, subjects)]
        
        X_temp, X_test, y_temp, y_test, s_temp, s_test, combined_temp, combined_test = train_test_split(
            data, labels, subjects, combined_labels,
            test_size=self.test_size,
            stratify=combined_labels,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
            X_temp, y_temp, s_temp,
            test_size=val_size_adjusted,
            stratify=combined_temp,
            random_state=self.random_state
        )
        
        splits = {
            'train': {
                'data': X_train,
                'labels': y_train,
                'subjects': s_train
            },
            'val': {
                'data': X_val,
                'labels': y_val,
                'subjects': s_val
            },
            'test': {
                'data': X_test,
                'labels': y_test,
                'subjects': s_test
            }
        }
        
        self._validate_splits(splits)
        return splits
    
    def _validate_splits(self, splits: Dict[str, Dict[str, np.ndarray]]):
        """Validate the quality of data splits"""
        
        logging.info("=== Data Split Validation ===")
        
        for split_name, split_data in splits.items():
            n_samples = len(split_data['data'])
            n_subjects = len(np.unique(split_data['subjects']))
            unique_labels, counts = np.unique(split_data['labels'], return_counts=True)
            
            logging.info(f"{split_name.upper()} SET:")
            logging.info(f"  Samples: {n_samples}")
            logging.info(f"  Subjects: {n_subjects}")
            logging.info(f"  Class distribution: {dict(zip(unique_labels, counts))}")
            
            # Check for class balance within split
            if len(counts) > 1:
                imbalance_ratio = counts.max() / counts.min()
                if imbalance_ratio > 2.0:
                    logging.warning(f"  Class imbalance in {split_name}: {imbalance_ratio:.2f}:1")
        
        # Check for subject overlap
        train_subjects = set(splits['train']['subjects'])
        val_subjects = set(splits['val']['subjects'])
        test_subjects = set(splits['test']['subjects'])
        
        if train_subjects & val_subjects:
            logging.warning("Subject overlap between train and validation sets!")
        if train_subjects & test_subjects:
            logging.warning("Subject overlap between train and test sets!")
        if val_subjects & test_subjects:
            logging.warning("Subject overlap between validation and test sets!")
            
        logging.info("=== Validation Complete ===")

def create_enhanced_dataloaders(data: np.ndarray,
                              labels: np.ndarray, 
                              subjects: np.ndarray,
                              config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create enhanced data loaders with proper validation and augmentation
    
    Args:
        data: EEG data array (n_samples, n_channels, n_timepoints)
        labels: Label array (n_samples,)
        subjects: Subject array (n_samples,)
        config: Training configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Initialize data splitter
    splitter = StratifiedDataSplitter(
        random_state=config.data.random_seed,
        test_size=config.data.test_split,
        val_size=config.data.validation_split,
        stratify_by_subject=True  # Important: prevent data leakage
    )
    
    # Split the data
    splits = splitter.split_data(data, labels, subjects)
    
    # Create augmentation for training
    train_augmentation = None
    if config.data.augmentation:
        train_augmentation = EEGAugmentation(
            noise_factor=config.data.noise_factor,
            time_shift_max=5
        )
    
    # Create datasets
    train_dataset = EnhancedEEGDataset(
        splits['train']['data'],
        splits['train']['labels'], 
        splits['train']['subjects'],
        augmentation=train_augmentation,
        normalize=True
    )
    
    val_dataset = EnhancedEEGDataset(
        splits['val']['data'],
        splits['val']['labels'],
        splits['val']['subjects'], 
        augmentation=None,  # No augmentation for validation
        normalize=True
    )
    
    test_dataset = EnhancedEEGDataset(
        splits['test']['data'],
        splits['test']['labels'],
        splits['test']['subjects'],
        augmentation=None,  # No augmentation for testing
        normalize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    logging.info("Enhanced data loaders created successfully")
    return train_loader, val_loader, test_loader