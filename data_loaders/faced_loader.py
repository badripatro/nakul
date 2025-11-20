# Databricks notebook source
#!/usr/bin/env python3
"""
FACED Dataset Data Loader with TorchEEG Support
================================================

This module provides a complete data loader for the FACED (Facial Expression and EEG Dataset)
that works with the downloaded Processed_data.zip from Synapse.

Dataset Information:
--------------------
- Source: https://www.synapse.org/#!Synapse:syn50614194/files/
- 123 subjects
- 30 EEG channels + 2 mastoid references (32 total)
- 250 Hz sampling rate
- 7 emotion classes: Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised
- Pre-processed data in .pkl files (one per subject)

File Structure (after extraction):
----------------------------------
Processed_data/
├── sub000.pkl
├── sub001.pkl
├── sub002.pkl
└── ...

Each .pkl file contains:
- EEG data: (n_trials, n_channels, n_samples)
- Labels: emotion labels for each trial
- Subject metadata

TorchEEG Integration:
---------------------
from torcheeg.datasets import FACEDDataset
from torcheeg import transforms
from torcheeg.datasets.constants import FACED_CHANNEL_LOCATION_DICT

dataset = FACEDDataset(
    root_path='./Processed_data',
    chunk_size=250,  # 1 second at 250 Hz
    overlap=0,
    num_channel=30,
    online_transform=transforms.ToTensor(),
    offline_transform=transforms.BandDifferentialEntropy(),
    label_transform=transforms.Compose([
        transforms.Select('emotion'),
        transforms.Lambda(lambda x: x + 1)
    ])
)
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union, Callable
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Try to import TorchEEG if available
try:
    import torcheeg
    from torcheeg.datasets import FACEDDataset as TorchEEGFACEDDataset
    from torcheeg import transforms
    from torcheeg.datasets.constants import FACED_CHANNEL_LOCATION_DICT, FACED_ADJACENCY_MATRIX
    TORCHEEG_AVAILABLE = True
except ImportError:
    TORCHEEG_AVAILABLE = False
    print("Warning: TorchEEG not installed. Using manual implementation.")
    print("Install with: pip install torcheeg")


# FACED Channel names (30 EEG channels)
FACED_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FT7', 'FC3', 'FCz', 'FC4', 'FT8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

# Emotion labels mapping
EMOTION_LABELS = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad', 
    3: 'Angry',
    4: 'Fearful',
    5: 'Disgusted',
    6: 'Surprised'
}


class FACEDPyTorchDataset(Dataset):
    """PyTorch Dataset wrapper for FACED data"""
    
    def __init__(self, data, labels, subjects=None, sessions=None, transform=None):
        """
        Args:
            data: Tensor of shape (n_samples, n_channels, n_timepoints)
            labels: Tensor of shape (n_samples,)
            subjects: Tensor of subject IDs (optional)
            sessions: Tensor of session IDs (optional)
            transform: Optional transform to apply to data
        """
        self.data = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.subjects = torch.LongTensor(subjects) if subjects is not None else None
        self.sessions = torch.LongTensor(sessions) if sessions is not None else None
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        if self.subjects is not None and self.sessions is not None:
            return x, y, self.subjects[idx], self.sessions[idx]
        elif self.subjects is not None:
            return x, y, self.subjects[idx]
        else:
            return x, y


class FACEDDataLoader:
    """
    Complete data loader for FACED dataset
    
    Supports both TorchEEG integration and standalone usage.
    """
    
    def __init__(
        self,
        root_path: str = './Processed_data',
        data_path: Optional[str] = None,  # Alternative to root_path
        subjects: Optional[List[int]] = None,
        chunk_size: int = 250,  # Number of samples per chunk (250 = 1s at 250Hz)
        overlap: int = 0,  # Overlap between chunks
        num_channel: int = 30,  # 30 EEG channels (or 32 with mastoid)
        sampling_rate: int = 250,  # Hz
        bandpass_filter: Optional[Tuple[float, float]] = (0.5, 50.0),  # (low, high) Hz
        notch_filter: Optional[float] = 50.0,  # Power line frequency (Hz)
        normalize_method: str = 'zscore',  # 'zscore', 'minmax', or 'none'
        use_torcheeg: bool = False,  # Whether to use TorchEEG implementation
        online_transform: Optional[Callable] = None,  # TorchEEG online transform
        offline_transform: Optional[Callable] = None,  # TorchEEG offline transform
        label_transform: Optional[Callable] = None,  # TorchEEG label transform
        before_trial: Optional[Callable] = None,  # Hook before trial processing
        after_trial: Optional[Callable] = None,  # Hook after trial processing
        verbose: bool = True
    ):
        """
        Initialize FACED data loader
        
        Args:
            root_path: Path to Processed_data directory (TorchEEG style)
            data_path: Alternative path specification
            subjects: List of subject IDs (0-122), None for all
            chunk_size: Samples per chunk (250=1s, 500=2s at 250Hz)
            overlap: Overlapping samples between chunks
            num_channel: 30 (EEG only) or 32 (with mastoid)
            sampling_rate: Sampling frequency (250 Hz)
            bandpass_filter: (low_freq, high_freq) for bandpass filtering
            notch_filter: Frequency for notch filtering (power line noise)
            normalize_method: 'zscore', 'minmax', or 'none'
            use_torcheeg: Use TorchEEG if available
            online_transform: TorchEEG online transformation
            offline_transform: TorchEEG offline transformation
            label_transform: TorchEEG label transformation
            before_trial: Hook called before trial processing
            after_trial: Hook called after trial processing
            verbose: Print progress information
        """
        # Use data_path if provided, otherwise root_path
        self.root_path = Path(data_path) if data_path else Path(root_path)
        self.subjects = subjects
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.num_channel = num_channel
        self.sampling_rate = sampling_rate
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.normalize_method = normalize_method
        self.use_torcheeg = use_torcheeg and TORCHEEG_AVAILABLE
        self.verbose = verbose
        
        # TorchEEG transforms and hooks
        self.online_transform = online_transform
        self.offline_transform = offline_transform
        self.label_transform = label_transform
        self.before_trial = before_trial
        self.after_trial = after_trial
        
        # Dataset metadata
        self.n_channels = num_channel
        self.n_classes = 7
        self.channel_names = FACED_CHANNELS[:num_channel]
        self.emotion_labels = EMOTION_LABELS
        
        if self.verbose:
            print("="*70)
            print("FACED Dataset Loader Initialized")
            print("="*70)
            print(f"Root path: {self.root_path}")
            print(f"Chunk size: {self.chunk_size} samples ({self.chunk_size/self.sampling_rate:.2f}s)")
            print(f"Overlap: {self.overlap} samples")
            print(f"Channels: {self.num_channel}")
            print(f"Sampling rate: {self.sampling_rate} Hz")
            print(f"Bandpass filter: {self.bandpass_filter}")
            print(f"Notch filter: {self.notch_filter} Hz" if self.notch_filter else "Notch filter: None")
            print(f"Normalization: {self.normalize_method}")
            print(f"Using TorchEEG: {self.use_torcheeg}")
            print(f"Number of classes: {self.n_classes}")
            print("="*70)
    
    def extract_data(self):
        """Extract Processed_data.zip if not already extracted"""
        zip_file = self.root_path.parent / 'Processed_data.zip'
        
        if self.root_path.exists() and any(self.root_path.glob('sub*.pkl')):
            if self.verbose:
                print(f"✓ Data already extracted at {self.root_path}")
            return True
        
        if not zip_file.exists():
            print(f"✗ Zip file not found: {zip_file}")
            print(f"Please download from: https://www.synapse.org/#!Synapse:syn50614194/files/")
            return False
        
        if self.verbose:
            print(f"Extracting {zip_file}...")
        
        try:
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.root_path.parent)
            if self.verbose:
                print(f"✓ Extracted to {self.root_path}")
            return True
        except Exception as e:
            print(f"✗ Extraction failed: {e}")
            print("Please manually extract Processed_data.zip")
            return False
    
    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to EEG data
        
        Args:
            data: (n_channels, n_samples) or (n_samples,)
            
        Returns:
            Filtered data with same shape
        """
        if self.bandpass_filter is None:
            return data
        
        nyquist = self.sampling_rate / 2
        low = self.bandpass_filter[0] / nyquist
        high = self.bandpass_filter[1] / nyquist
        
        # Clamp to valid range
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        if low >= high:
            print(f"Warning: Invalid bandpass range [{low*nyquist}, {high*nyquist}] Hz")
            return data
        
        try:
            b, a = butter(5, [low, high], btype='band')
            if data.ndim == 1:
                return filtfilt(b, a, data)
            else:
                return filtfilt(b, a, data, axis=-1)
        except Exception as e:
            print(f"Warning: Bandpass filter failed: {e}")
            return data
    
    def apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line noise
        
        Args:
            data: (n_channels, n_samples) or (n_samples,)
            
        Returns:
            Filtered data with same shape
        """
        if self.notch_filter is None:
            return data
        
        nyquist = self.sampling_rate / 2
        freq = self.notch_filter / nyquist
        
        if freq <= 0 or freq >= 1:
            return data
        
        try:
            # Notch filter with Q=30 (bandwidth = freq/30)
            Q = 30.0
            w0 = freq
            bw = w0 / Q
            
            low = max(0.001, w0 - bw/2)
            high = min(0.999, w0 + bw/2)
            
            # Use bandstop filter
            b, a = butter(2, [low, high], btype='bandstop')
            if data.ndim == 1:
                return filtfilt(b, a, data)
            else:
                return filtfilt(b, a, data, axis=-1)
        except Exception as e:
            print(f"Warning: Notch filter failed: {e}")
            return data
    
    def load_subject_file(self, subject_id: int) -> Optional[Dict]:
        """
        Load .pkl file for a single subject
        
        Args:
            subject_id: Subject ID (0-122)
            
        Returns:
            Dictionary with 'data', 'labels', 'metadata'
        """
        subject_file = self.root_path / f"sub{subject_id:03d}.pkl"
        
        if not subject_file.exists():
            if self.verbose:
                print(f"✗ Subject {subject_id}: File not found at {subject_file}")
            return None
        
        try:
            with open(subject_file, 'rb') as f:
                subject_data = pickle.load(f)
            
            # Handle different possible .pkl formats
            if isinstance(subject_data, dict):
                # Expected format: {'data': array, 'labels': array, 'metadata': dict}
                return subject_data
            elif isinstance(subject_data, (list, tuple)):
                # Format: (data, labels) or (data, labels, metadata)
                if len(subject_data) == 2:
                    return {'data': subject_data[0], 'labels': subject_data[1]}
                elif len(subject_data) == 3:
                    return {'data': subject_data[0], 'labels': subject_data[1], 
                           'metadata': subject_data[2]}
            elif isinstance(subject_data, np.ndarray):
                # FACED format: Pure numpy array (n_trials, n_channels, n_samples)
                # FACED has 7 emotions × 4 trials = 28 trials per subject
                # Trial order: 4x Neutral, 4x Happy, 4x Sad, 4x Angry, 4x Fearful, 4x Disgusted, 4x Surprised
                n_trials = subject_data.shape[0]
                
                # Generate labels based on FACED trial structure
                if n_trials == 28:
                    # 7 emotions × 4 trials each
                    labels = np.repeat(np.arange(7), 4)
                elif n_trials % 7 == 0:
                    # Assume equal distribution across 7 emotions
                    trials_per_emotion = n_trials // 7
                    labels = np.repeat(np.arange(7), trials_per_emotion)
                else:
                    # Unknown structure, assign sequential labels
                    if self.verbose:
                        print(f"⚠ Subject {subject_id}: Unusual trial count {n_trials}, using cyclic labels")
                    labels = np.tile(np.arange(7), (n_trials // 7) + 1)[:n_trials]
                
                return {'data': subject_data, 'labels': labels}
            else:
                # Unknown format, try to extract data
                return {'data': subject_data}
                
        except Exception as e:
            if self.verbose:
                print(f"✗ Subject {subject_id}: Load error - {e}")
            return None
    
    def preprocess_trial(self, trial_data: np.ndarray) -> np.ndarray:
        """
        Preprocess a single trial
        
        Args:
            trial_data: (n_channels, n_samples)
            
        Returns:
            Preprocessed trial data
        """
        # Apply before_trial hook if provided
        if self.before_trial is not None:
            trial_data = self.before_trial(trial_data)
        
        # Apply bandpass filter
        if self.bandpass_filter is not None:
            trial_data = self.apply_bandpass_filter(trial_data)
        
        # Apply notch filter
        if self.notch_filter is not None:
            trial_data = self.apply_notch_filter(trial_data)
        
        # Baseline correction (remove DC offset)
        trial_data = trial_data - np.mean(trial_data, axis=-1, keepdims=True)
        
        return trial_data
    
    def create_chunks(self, trial_data: np.ndarray, label: int, 
                     subject_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split trial into overlapping chunks
        
        Args:
            trial_data: (n_channels, n_samples)
            label: Emotion label
            subject_id: Subject ID
            
        Returns:
            chunks: (n_chunks, n_channels, chunk_size)
            chunk_labels: (n_chunks,)
            chunk_subjects: (n_chunks,)
        """
        n_channels, n_samples = trial_data.shape
        
        if n_samples < self.chunk_size:
            # Zero-pad if trial is too short
            pad_width = ((0, 0), (0, self.chunk_size - n_samples))
            trial_data = np.pad(trial_data, pad_width, mode='constant')
            n_samples = self.chunk_size
        
        step_size = self.chunk_size - self.overlap
        n_chunks = (n_samples - self.chunk_size) // step_size + 1
        
        chunks = []
        for i in range(n_chunks):
            start = i * step_size
            end = start + self.chunk_size
            chunk = trial_data[:, start:end]
            
            # Apply offline transform if provided
            if self.offline_transform is not None:
                chunk = self.offline_transform(chunk)
            
            chunks.append(chunk)
        
        chunks = np.array(chunks)
        chunk_labels = np.full(n_chunks, label, dtype=np.int64)
        chunk_subjects = np.full(n_chunks, subject_id, dtype=np.int64)
        
        return chunks, chunk_labels, chunk_subjects
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess all subject data
        
        Returns:
            X: (n_samples, n_channels, chunk_size)
            y: (n_samples,) - emotion labels
            subjects: (n_samples,) - subject IDs
        """
        # Extract data if needed
        if not self.extract_data():
            raise FileNotFoundError("Could not find or extract data")
        
        # Find all subject files
        subject_files = sorted(self.root_path.glob('sub*.pkl'))
        
        if not subject_files:
            raise FileNotFoundError(f"No .pkl files found in {self.root_path}")
        
        # Extract subject IDs from filenames
        available_subjects = []
        for f in subject_files:
            try:
                subject_id = int(f.stem[3:])  # Extract number from 'sub000'
                available_subjects.append(subject_id)
            except:
                continue
        
        # Filter by requested subjects
        if self.subjects is not None:
            available_subjects = [s for s in available_subjects if s in self.subjects]
        
        if not available_subjects:
            raise ValueError(f"No valid subjects found. Available: {len(subject_files)}")
        
        if self.verbose:
            print(f"\nLoading {len(available_subjects)} subjects...")
            print(f"Subject IDs: {available_subjects[:10]}{'...' if len(available_subjects) > 10 else ''}")
        
        all_chunks = []
        all_labels = []
        all_subjects = []
        
        for subject_id in available_subjects:
            subject_data = self.load_subject_file(subject_id)
            
            if subject_data is None:
                continue
            
            # Extract data and labels
            trials = subject_data.get('data')  # (n_trials, n_channels, n_samples)
            labels = subject_data.get('labels')  # (n_trials,)
            
            if trials is None:
                if self.verbose:
                    print(f"✗ Subject {subject_id}: No data found")
                continue
            
            # Ensure correct shape
            if trials.ndim == 2:
                # Single trial: (n_channels, n_samples)
                trials = trials[np.newaxis, ...]
            
            n_trials = trials.shape[0]
            
            # Handle missing labels
            if labels is None:
                labels = np.zeros(n_trials, dtype=np.int64)
            
            subject_chunks = []
            subject_labels = []
            
            for trial_idx in range(n_trials):
                trial_data = trials[trial_idx]  # (n_channels, n_samples)
                trial_label = labels[trial_idx] if trial_idx < len(labels) else 0
                
                # Ensure correct number of channels
                if trial_data.shape[0] != self.n_channels:
                    # Take first n_channels if more available
                    trial_data = trial_data[:self.n_channels, :]
                
                # Preprocess trial
                trial_data = self.preprocess_trial(trial_data)
                
                # Create chunks
                chunks, chunk_labels, chunk_subjects = self.create_chunks(
                    trial_data, trial_label, subject_id
                )
                
                subject_chunks.append(chunks)
                subject_labels.append(chunk_labels)
            
            if subject_chunks:
                all_chunks.append(np.concatenate(subject_chunks, axis=0))
                all_labels.append(np.concatenate(subject_labels, axis=0))
                all_subjects.append(np.full(len(all_labels[-1]), subject_id, dtype=np.int64))
                
                if self.verbose:
                    print(f"✓ Subject {subject_id:03d}: {len(all_labels[-1]):5d} chunks from {n_trials:3d} trials")
        
        if not all_chunks:
            raise ValueError("No data could be loaded")
        
        X = np.concatenate(all_chunks, axis=0)
        y = np.concatenate(all_labels, axis=0)
        subjects = np.concatenate(all_subjects, axis=0)
        
        # Normalize if requested
        if self.normalize_method == 'zscore':
            X = self.zscore_normalize(X)
        elif self.normalize_method == 'minmax':
            X = self.minmax_normalize(X)
        
        if self.verbose:
            print("\n" + "="*70)
            print("FACED Dataset Loaded Successfully")
            print("="*70)
            print(f"Total samples: {len(X)}")
            print(f"Shape: {X.shape}")
            print(f"Subjects: {len(np.unique(subjects))}")
            print(f"Class distribution:")
            for class_id in range(self.n_classes):
                count = np.sum(y == class_id)
                print(f"  {self.emotion_labels[class_id]:12s}: {count:6d} ({100*count/len(y):.1f}%)")
            print("="*70)
        
        return X, y, subjects
    
    def zscore_normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        X_flat = X.reshape(len(X), -1)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X_flat)
        return X_normalized.reshape(X.shape)
    
    def minmax_normalize(self, X: np.ndarray, feature_range=(0, 1)) -> np.ndarray:
        """Min-max normalization"""
        X_min = X.min(axis=(1, 2), keepdims=True)
        X_max = X.max(axis=(1, 2), keepdims=True)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1  # Avoid division by zero
        X_normalized = (X - X_min) / X_range
        # Scale to feature_range
        X_normalized = X_normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
        return X_normalized
    
    def get_train_val_test_split(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        split_by_subject: bool = True,
        random_state: int = 42
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Split data into train/validation/test sets
        
        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            split_by_subject: If True, split by subject (prevents data leakage)
            random_state: Random seed
            
        Returns:
            (X_train, y_train, subj_train),
            (X_val, y_val, subj_val),
            (X_test, y_test, subj_test)
        """
        X, y, subjects = self.load_all_data()
        
        if split_by_subject:
            # Split by subject to prevent data leakage
            unique_subjects = np.unique(subjects)
            
            # Train/test split
            train_val_subjects, test_subjects = train_test_split(
                unique_subjects,
                test_size=test_size,
                random_state=random_state
            )
            
            # Train/val split
            train_subjects, val_subjects = train_test_split(
                train_val_subjects,
                test_size=val_size / (1 - test_size),
                random_state=random_state
            )
            
            # Create masks
            train_mask = np.isin(subjects, train_subjects)
            val_mask = np.isin(subjects, val_subjects)
            test_mask = np.isin(subjects, test_subjects)
            
        else:
            # Random split (may have data leakage across subjects)
            train_val_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=test_size,
                stratify=y,
                random_state=random_state
            )
            
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size / (1 - test_size),
                stratify=y[train_val_idx],
                random_state=random_state
            )
            
            train_mask = np.zeros(len(X), dtype=bool)
            val_mask = np.zeros(len(X), dtype=bool)
            test_mask = np.zeros(len(X), dtype=bool)
            
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            test_mask[test_idx] = True
        
        # Split data
        X_train, y_train, subj_train = X[train_mask], y[train_mask], subjects[train_mask]
        X_val, y_val, subj_val = X[val_mask], y[val_mask], subjects[val_mask]
        X_test, y_test, subj_test = X[test_mask], y[test_mask], subjects[test_mask]
        
        if self.verbose:
            print("\nData Split:")
            print(f"  Train: {len(X_train):5d} samples, {len(np.unique(subj_train)):3d} subjects")
            print(f"  Val:   {len(X_val):5d} samples, {len(np.unique(subj_val)):3d} subjects")
            print(f"  Test:  {len(X_test):5d} samples, {len(np.unique(subj_test)):3d} subjects")
        
        return ((X_train, y_train, subj_train),
                (X_val, y_val, subj_val),
                (X_test, y_test, subj_test))
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        test_size: float = 0.15,
        val_size: float = 0.15,
        test_split: float = None,  # Alias for compatibility
        val_split: float = None,   # Alias for compatibility
        split_by_subject: bool = True,
        num_workers: int = 4,
        random_state: int = 42,
        random_seed: int = None    # Alias for compatibility
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders for train/val/test
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # Handle parameter aliases for compatibility
        if test_split is not None:
            test_size = test_split
        if val_split is not None:
            val_size = val_split
        if random_seed is not None:
            random_state = random_seed
        
        # Get train/val/test splits
        (X_train, y_train, subj_train), \
        (X_val, y_val, subj_val), \
        (X_test, y_test, subj_test) = self.get_train_val_test_split(
            test_size=test_size,
            val_size=val_size,
            split_by_subject=split_by_subject,
            random_state=random_state
        )
        
        # Create datasets
        train_dataset = FACEDPyTorchDataset(
            X_train, y_train, subj_train,
            transform=self.online_transform
        )
        val_dataset = FACEDPyTorchDataset(
            X_val, y_val, subj_val,
            transform=self.online_transform
        )
        test_dataset = FACEDPyTorchDataset(
            X_test, y_test, subj_test,
            transform=self.online_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def use_torcheeg_dataset(self, **kwargs):
        """
        Use TorchEEG's FACEDDataset implementation
        
        Args:
            **kwargs: Additional arguments for TorchEEG FACEDDataset
            
        Returns:
            TorchEEG FACEDDataset instance
        """
        if not TORCHEEG_AVAILABLE:
            raise ImportError("TorchEEG not installed. Install with: pip install torcheeg")
        
        # Merge kwargs with current settings
        dataset_kwargs = {
            'root_path': str(self.root_path),
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'num_channel': self.num_channel,
            'online_transform': self.online_transform,
            'offline_transform': self.offline_transform,
            'label_transform': self.label_transform,
            'before_trial': self.before_trial,
            'after_trial': self.after_trial,
        }
        dataset_kwargs.update(kwargs)
        
        return TorchEEGFACEDDataset(**dataset_kwargs)


def test_faced_loader():
    """Test FACED data loader"""
    print("\n" + "="*70)
    print("Testing FACED Data Loader")
    print("="*70 + "\n")
    
    # Test with first 3 subjects
    loader = FACEDDataLoader(
        root_path='../data/Processed_data',
        subjects=None,  # Load all available
        chunk_size=250,  # 1 second chunks
        overlap=0,
        num_channel=30,
        bandpass_filter=(0.5, 50.0),
        normalize_method='zscore',
        verbose=True
    )
    
    try:
        # Test loading data
        X, y, subjects = loader.load_all_data()
        
        print(f"\n✅ Data loaded successfully!")
        print(f"   Shape: {X.shape}")
        print(f"   Labels: {y.shape}")
        print(f"   Data type: {X.dtype}")
        print(f"   Label range: {y.min()} to {y.max()}")
        
        # Test getting dataloaders
        print("\n" + "-"*70)
        print("Creating DataLoaders...")
        print("-"*70)
        
        train_loader, val_loader, test_loader = loader.get_dataloaders(
            batch_size=32,
            test_size=0.15,
            val_size=0.15,
            split_by_subject=True
        )
        
        print(f"\n✅ DataLoaders created!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test batch iteration
        print("\n" + "-"*70)
        print("Testing batch iteration...")
        print("-"*70)
        
        for batch_idx, (x, y, s) in enumerate(train_loader):
            print(f"   Batch {batch_idx}: x={x.shape}, y={y.shape}, subjects={s.shape}")
            if batch_idx >= 2:  # Show first 3 batches
                break
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    test_faced_loader()
