"""
SEED Dataset Loader for Emotion Recognition
============================================

SEED (SJTU Emotion EEG Dataset) contains EEG signals while subjects watched 
emotional video clips. The dataset has 3 emotion classes:
- Positive (1)
- Neutral (0)
- Negative (-1)

Dataset details:
- 62 EEG channels (ESI NeuroScan system)
- 200 Hz sampling rate (original)
- 3 emotion classes
- 15 subjects
- 15 film clips per emotion (45 total)
- Each trial: ~4 minutes of continuous EEG

Download from: https://bcmi.sjtu.edu.cn/~seed/
"""

import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SEEDDataset(Dataset):
    """PyTorch Dataset for SEED"""
    
    def __init__(self, data, labels, subjects, transform=None):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.subjects = torch.LongTensor(subjects)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        s = self.subjects[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y, s


class SEEDDataLoader:
    """
    Data loader for SEED dataset
    """
    
    def __init__(
        self,
        data_path: str = './data/SEED',
        subjects: Optional[List[int]] = None,
        window_size: float = 2.0,  # seconds
        window_overlap: float = 0.5,  # 50% overlap
        target_freq: int = 250,  # Target sampling rate
        bandpass: Tuple[float, float] = (0.5, 50.0),  # Hz
        normalize_per_subject: bool = True
    ):
        """
        Initialize SEED data loader
        
        Args:
            data_path: Path to SEED dataset
            subjects: List of subject IDs (1-15), None for all
            window_size: Window size in seconds
            window_overlap: Overlap ratio (0-1)
            target_freq: Target resampling frequency
            bandpass: Bandpass filter range (low, high) in Hz
            normalize_per_subject: Whether to normalize per subject
        """
        self.data_path = Path(data_path)
        self.subjects = subjects if subjects is not None else list(range(1, 16))
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.target_freq = target_freq
        self.bandpass = bandpass
        self.normalize_per_subject = normalize_per_subject
        
        # SEED metadata
        self.original_freq = 200  # Hz
        self.n_channels = 62
        self.n_classes = 3
        
        # Emotion mapping: -1 (negative) -> 0, 0 (neutral) -> 1, 1 (positive) -> 2
        self.emotion_map = {-1: 0, 0: 1, 1: 2}
        
        print("SEED Data Loader initialized")
        print(f"Data path: {self.data_path}")
        print(f"Subjects: {self.subjects}")
        print(f"Window size: {self.window_size}s")
        print(f"Target frequency: {self.target_freq} Hz")
        
    def bandpass_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """Apply bandpass filter"""
        nyq = fs / 2
        low = self.bandpass[0] / nyq
        high = self.bandpass[1] / nyq
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)
    
    def create_windows(
        self,
        data: np.ndarray,
        label: int,
        subject_id: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create overlapping windows from continuous EEG data
        
        Args:
            data: EEG data (n_channels, n_samples)
            label: Emotion label
            subject_id: Subject ID
            
        Returns:
            Windowed data, labels, subject IDs
        """
        n_channels, n_samples = data.shape
        window_samples = int(self.window_size * self.target_freq)
        step_samples = int(window_samples * (1 - self.window_overlap))
        
        windows = []
        labels = []
        subjects = []
        
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window = data[:, start:end]
            windows.append(window)
            labels.append(label)
            subjects.append(subject_id)
            
        return (
            np.array(windows),
            np.array(labels),
            np.array(subjects)
        )
    
    def load_subject_data(self, subject_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data for a single subject
        
        Returns:
            data: (n_trials, n_channels, n_samples)
            labels: (n_trials,)
            subjects: (n_trials,)
        """
        # SEED files are typically named like: 1_20131027.mat, 1_20131030.mat, 1_20131107.mat
        # Each subject has 3 sessions
        subject_files = list(self.data_path.glob(f"{subject_id}_*.mat"))
        
        if not subject_files:
            raise FileNotFoundError(
                f"No data files found for subject {subject_id} in {self.data_path}"
            )
        
        all_windows = []
        all_labels = []
        all_subjects = []
        
        for file_path in subject_files:
            # Load .mat file
            mat_data = sio.loadmat(file_path)
            
            # SEED structure: Each file contains multiple trials
            # Keys are like: 'djc_eeg1', 'djc_eeg2', ..., 'djc_eeg15' (15 trials per emotion)
            # Labels are in a separate file or follow a pattern
            
            # Find all EEG data keys
            eeg_keys = [k for k in mat_data.keys() if 'eeg' in k.lower() and not k.startswith('__')]
            eeg_keys.sort()
            
            # SEED label pattern: Each session has 15 trials per emotion
            # Order: [pos, neg, neu, ...] (varies by session)
            # Load label file if available
            label_file = self.data_path / "label.mat"
            if label_file.exists():
                label_data = sio.loadmat(label_file)
                trial_labels = label_data['label'][0]  # (45,) array with -1, 0, 1
            else:
                # Default label pattern (may vary - check SEED documentation)
                trial_labels = np.array([1, -1, 0] * 15)  # Positive, Negative, Neutral repeated
            
            # Process each trial
            for trial_idx, key in enumerate(eeg_keys):
                if trial_idx >= len(trial_labels):
                    break
                    
                eeg_data = mat_data[key]  # (n_channels, n_samples)
                
                # Ensure correct shape
                if eeg_data.shape[0] != self.n_channels:
                    eeg_data = eeg_data.T
                
                # Apply bandpass filter
                eeg_data = self.bandpass_filter(eeg_data, self.original_freq)
                
                # Resample to target frequency
                if self.target_freq != self.original_freq:
                    n_samples_new = int(eeg_data.shape[1] * self.target_freq / self.original_freq)
                    eeg_data = resample(eeg_data, n_samples_new, axis=1)
                
                # Map emotion label
                original_label = trial_labels[trial_idx]
                mapped_label = self.emotion_map[original_label]
                
                # Create windows
                windows, labels, subjects = self.create_windows(
                    eeg_data,
                    mapped_label,
                    subject_id
                )
                
                all_windows.append(windows)
                all_labels.append(labels)
                all_subjects.append(subjects)
        
        if not all_windows:
            raise ValueError(f"No valid data loaded for subject {subject_id}")
        
        return (
            np.concatenate(all_windows, axis=0),
            np.concatenate(all_labels, axis=0),
            np.concatenate(all_subjects, axis=0)
        )
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data for all subjects
        
        Returns:
            X: (n_samples, n_channels, n_timepoints)
            y: (n_samples,)
            subjects: (n_samples,)
        """
        print(f"\nLoading SEED data for subjects: {self.subjects}")
        
        all_data = []
        all_labels = []
        all_subjects = []
        
        for subject_id in self.subjects:
            try:
                print(f"Loading subject {subject_id}...", end=' ')
                data, labels, subjects = self.load_subject_data(subject_id)
                
                all_data.append(data)
                all_labels.append(labels)
                all_subjects.append(subjects)
                
                print(f"✓ {len(data)} windows")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded successfully")
        
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        subjects = np.concatenate(all_subjects, axis=0)
        
        # Normalize per subject if requested
        if self.normalize_per_subject:
            print("\nNormalizing data per subject...")
            X_normalized = np.zeros_like(X)
            for subject_id in np.unique(subjects):
                subject_mask = subjects == subject_id
                subject_data = X[subject_mask]
                
                # Flatten for normalization
                n_samples = subject_data.shape[0]
                subject_data_flat = subject_data.reshape(n_samples, -1)
                
                # Normalize
                scaler = StandardScaler()
                subject_data_normalized = scaler.fit_transform(subject_data_flat)
                subject_data_normalized = subject_data_normalized.reshape(subject_data.shape)
                
                X_normalized[subject_mask] = subject_data_normalized
            
            X = X_normalized
        
        print(f"\nTotal SEED dataset loaded:")
        print(f"  Shape: {X.shape}")
        print(f"  Classes: {np.unique(y)} (0=Negative, 1=Neutral, 2=Positive)")
        print(f"  Subjects: {np.unique(subjects)}")
        print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y, subjects
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train/val/test data loaders
        
        Returns:
            train_loader, val_loader, test_loader
        """
        from sklearn.model_selection import train_test_split
        
        X, y, subjects = self.load_data()
        
        # Split by subject to avoid data leakage
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        
        # Calculate split sizes
        n_test = max(1, int(n_subjects * test_split))
        n_val = max(1, int(n_subjects * val_split))
        
        # Split subjects
        train_subjects, test_subjects = train_test_split(
            unique_subjects,
            test_size=n_test,
            random_state=random_seed
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects,
            test_size=n_val,
            random_state=random_seed
        )
        
        # Create masks
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)
        test_mask = np.isin(subjects, test_subjects)
        
        # Split data
        X_train, y_train, subj_train = X[train_mask], y[train_mask], subjects[train_mask]
        X_val, y_val, subj_val = X[val_mask], y[val_mask], subjects[val_mask]
        X_test, y_test, subj_test = X[test_mask], y[test_mask], subjects[test_mask]
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples from {len(train_subjects)} subjects")
        print(f"  Val:   {len(X_val)} samples from {len(val_subjects)} subjects")
        print(f"  Test:  {len(X_test)} samples from {len(test_subjects)} subjects")
        
        # Create datasets
        train_dataset = SEEDDataset(X_train, y_train, subj_train)
        val_dataset = SEEDDataset(X_val, y_val, subj_val)
        test_dataset = SEEDDataset(X_test, y_test, subj_test)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


def test_seed_loader():
    """Test SEED data loader"""
    loader = SEEDDataLoader(
        data_path='./data/SEED',
        subjects=[1, 2],
        window_size=2.0,
        target_freq=250
    )
    
    try:
        X, y, subjects = loader.load_data()
        print("\n✓ SEED loader test passed!")
        print(f"  Data shape: {X.shape}")
        print(f"  Label shape: {y.shape}")
    except Exception as e:
        print(f"\n✗ SEED loader test failed: {e}")


if __name__ == "__main__":
    test_seed_loader()
