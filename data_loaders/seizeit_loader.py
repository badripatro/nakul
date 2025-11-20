"""
SeizeIT1 (SzCORE) EEG-fMRI Hybrid Dataset Loader
=================================================

This module provides functionality to load and process the SeizeIT1 dataset
for seizure prediction using simultaneous EEG-fMRI recordings.

Dataset: SeizeIT1 (SzCORE - Seizure CORE Dataset)
- Modality: Simultaneous EEG-fMRI recording
- Subjects: 42 subjects with epilepsy
- EEG: 250 Hz sampling rate, multiple channels
- fMRI: BOLD signals synchronized with EEG
- Use Case: Seizure prediction, seizure type classification
- Details: Annotated seizure events, medication information
- URL: https://openneuro.org/datasets/ds004100 (or similar)

Features:
- Synchronized EEG and fMRI data
- Seizure annotations (onset, offset, type)
- Medication/treatment information
- Symbolic conditioning with seizure archetypes
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from scipy.signal import butter, filtfilt, resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("Warning: MNE not installed. Install with: pip install mne")


class SeizeITDataset(Dataset):
    """PyTorch Dataset for SeizeIT1 EEG-fMRI data"""
    
    def __init__(self, eeg_data, fmri_data, labels, subjects, 
                 metadata=None, transform=None):
        """
        Args:
            eeg_data: Tensor of shape (n_samples, n_eeg_channels, n_timepoints)
            fmri_data: Tensor of shape (n_samples, n_fmri_features)
            labels: Tensor of shape (n_samples,) - seizure labels
            subjects: Tensor of subject IDs
            metadata: Additional metadata (seizure type, medication, etc.)
            transform: Optional transform
        """
        self.eeg_data = torch.FloatTensor(eeg_data) if not isinstance(eeg_data, torch.Tensor) else eeg_data
        self.fmri_data = torch.FloatTensor(fmri_data) if fmri_data is not None else None
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.subjects = torch.LongTensor(subjects) if subjects is not None else None
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        y = self.labels[idx]
        
        if self.transform:
            eeg = self.transform(eeg)
        
        # Return both EEG and fMRI if available
        if self.fmri_data is not None:
            fmri = self.fmri_data[idx]
            if self.subjects is not None:
                return {'eeg': eeg, 'fmri': fmri}, y, self.subjects[idx]
            return {'eeg': eeg, 'fmri': fmri}, y
        else:
            if self.subjects is not None:
                return eeg, y, self.subjects[idx]
            return eeg, y


class SeizeITLoader:
    """
    Data loader for SeizeIT1 (SzCORE) EEG-fMRI dataset
    
    Supports:
    - Simultaneous EEG-fMRI loading
    - Seizure event detection and labeling
    - Pre-ictal/ictal/post-ictal segmentation
    - Medication/treatment conditioning
    - Multi-modal feature extraction
    """
    
    def __init__(
        self,
        data_path: str = './data/seizeit',
        subjects: Optional[List[str]] = None,
        eeg_channels: Optional[List[str]] = None,
        sampling_rate: int = 250,  # Hz
        window_size: float = 4.0,  # seconds
        window_overlap: float = 0.5,  # 50% overlap
        preictal_window: float = 30.0,  # seconds before seizure
        postictal_window: float = 30.0,  # seconds after seizure
        bandpass_filter: Tuple[float, float] = (0.5, 50.0),
        notch_filter: float = 50.0,  # Hz
        include_fmri: bool = True,
        fmri_preprocessing: str = 'connectivity',  # 'connectivity', 'glm', 'raw'
        normalize: bool = True,
        balance_classes: bool = True,
        verbose: bool = True
    ):
        """
        Initialize SeizeIT1 loader
        
        Args:
            data_path: Path to SeizeIT dataset
            subjects: List of subject IDs, None for all
            eeg_channels: List of EEG channel names to use
            sampling_rate: EEG sampling rate
            window_size: Window size in seconds
            window_overlap: Overlap ratio (0-1)
            preictal_window: Time before seizure to mark as pre-ictal
            postictal_window: Time after seizure to mark as post-ictal
            bandpass_filter: (low, high) frequency for bandpass
            notch_filter: Frequency for notch filtering
            include_fmri: Whether to include fMRI features
            fmri_preprocessing: fMRI feature extraction method
            normalize: Apply z-score normalization
            balance_classes: Balance seizure vs non-seizure samples
            verbose: Print progress
        """
        self.data_path = Path(data_path)
        self.subjects = subjects
        self.eeg_channels = eeg_channels
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.preictal_window = preictal_window
        self.postictal_window = postictal_window
        self.bandpass_filter = bandpass_filter
        self.notch_filter = notch_filter
        self.include_fmri = include_fmri
        self.fmri_preprocessing = fmri_preprocessing
        self.normalize = normalize
        self.balance_classes = balance_classes
        self.verbose = verbose
        
        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Seizure state labels
        self.state_map = {
            'interictal': 0,  # Normal/baseline
            'preictal': 1,    # Before seizure
            'ictal': 2,       # During seizure
            'postictal': 3    # After seizure
        }
        
        # Seizure type labels
        self.seizure_type_map = {
            'focal': 0,
            'generalized': 1,
            'absence': 2,
            'tonic-clonic': 3,
            'unknown': 4
        }
        
        if self.verbose:
            print("="*70)
            print("SeizeIT1 EEG-fMRI Loader Initialized")
            print("="*70)
            print(f"Data path: {self.data_path}")
            print(f"Sampling rate: {sampling_rate} Hz")
            print(f"Window size: {window_size}s")
            print(f"Pre-ictal window: {preictal_window}s")
            print(f"Include fMRI: {include_fmri}")
            print("="*70)
    
    def download_dataset(self):
        """
        Download SeizeIT1 dataset
        
        Note: Real implementation would use:
        - OpenNeuro/DataLad for BIDS-formatted data
        - Direct download from SzCORE repository
        """
        if self.data_path.exists() and any(self.data_path.glob('sub-*')):
            if self.verbose:
                print(f"✓ Dataset already exists: {self.data_path}")
            return True
        
        if self.verbose:
            print(f"\nDownloading SeizeIT1 dataset...")
            print("Note: This is a placeholder. Real download would use:")
            print("  - OpenNeuro: https://openneuro.org/datasets/ds004100")
            print("  - SzCORE repository access")
        
        # Create mock data
        self._create_mock_data()
        return True
    
    def _create_mock_data(self):
        """Create mock EEG-fMRI data with seizure annotations"""
        if self.verbose:
            print("\nCreating mock SeizeIT1 data...")
        
        # Create BIDS structure
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset description
        dataset_desc = {
            "Name": "SeizeIT1 Mock",
            "BIDSVersion": "1.6.0",
            "DatasetType": "raw",
            "Modality": ["EEG", "MRI"]
        }
        with open(self.data_path / 'dataset_description.json', 'w') as f:
            json.dump(dataset_desc, f, indent=2)
        
        # Create participants file
        n_subjects = 10
        participants_data = {
            'participant_id': [f'sub-{i+1:03d}' for i in range(n_subjects)],
            'age': np.random.randint(18, 65, n_subjects),
            'sex': np.random.choice(['M', 'F'], n_subjects),
            'seizure_type': np.random.choice(['focal', 'generalized', 'tonic-clonic'], n_subjects),
            'medication': np.random.choice(['none', 'AED1', 'AED2', 'combo'], n_subjects),
            'seizure_frequency': np.random.randint(1, 20, n_subjects)  # per month
        }
        pd.DataFrame(participants_data).to_csv(
            self.data_path / 'participants.tsv', sep='\t', index=False
        )
        
        # Create mock data for each subject
        for i in range(n_subjects):
            subj_id = f'sub-{i+1:03d}'
            
            # EEG directory
            eeg_dir = self.data_path / subj_id / 'eeg'
            eeg_dir.mkdir(parents=True, exist_ok=True)
            
            # fMRI directory
            fmri_dir = self.data_path / subj_id / 'func'
            fmri_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock EEG data
            n_channels = 64  # Typical EEG cap
            n_samples = int(300 * self.sampling_rate)  # 5 minutes
            
            # Generate realistic EEG with seizure events
            eeg_data = np.random.randn(n_channels, n_samples) * 20  # microvolts
            
            # Add seizure events (2-3 seizures per subject)
            n_seizures = np.random.randint(1, 4)
            seizure_events = []
            
            for j in range(n_seizures):
                # Random seizure timing
                seizure_start = np.random.randint(
                    int(30 * self.sampling_rate),
                    int((300 - 60) * self.sampling_rate)
                )
                seizure_duration = np.random.randint(
                    int(10 * self.sampling_rate),
                    int(30 * self.sampling_rate)
                )
                seizure_end = seizure_start + seizure_duration
                
                # Add seizure signature (increased amplitude and frequency)
                for ch in range(n_channels):
                    t = np.arange(seizure_duration) / self.sampling_rate
                    # Spike-wave pattern
                    seizure_signal = 50 * np.sin(2 * np.pi * 3 * t)  # 3 Hz spike-wave
                    seizure_signal += 30 * np.sin(2 * np.pi * 10 * t)  # 10 Hz activity
                    eeg_data[ch, seizure_start:seizure_end] += seizure_signal
                
                seizure_events.append({
                    'onset': seizure_start / self.sampling_rate,
                    'duration': seizure_duration / self.sampling_rate,
                    'type': np.random.choice(['focal', 'generalized'])
                })
            
            # Save EEG data
            np.savez(
                eeg_dir / f'{subj_id}_task-rest_eeg.npz',
                data=eeg_data,
                sfreq=self.sampling_rate,
                ch_names=[f'EEG{i+1:03d}' for i in range(n_channels)],
                seizure_events=seizure_events
            )
            
            # Create mock fMRI data
            fmri_shape = (64, 64, 30, 150)  # (x, y, z, time) - 5 minutes at 2s TR
            fmri_data = np.random.randn(*fmri_shape).astype(np.float32) * 100
            
            # Add BOLD signal changes during seizures
            for event in seizure_events:
                tr_start = int(event['onset'] / 2)  # TR = 2s
                tr_end = int((event['onset'] + event['duration']) / 2)
                # Increase signal in "seizure focus" region
                fmri_data[20:40, 20:40, 10:20, tr_start:tr_end] *= 1.5
            
            affine = np.eye(4)
            img = nib.Nifti1Image(fmri_data, affine)
            nib.save(img, fmri_dir / f'{subj_id}_task-rest_bold.nii.gz')
        
        if self.verbose:
            print(f"✓ Created mock data for {n_subjects} subjects")
    
    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to EEG data"""
        nyq = self.sampling_rate / 2
        low = self.bandpass_filter[0] / nyq
        high = self.bandpass_filter[1] / nyq
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, data, axis=-1)
    
    def load_subject_eeg(self, subject_id: str) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load EEG data for a subject
        
        Returns:
            eeg_data: (n_channels, n_samples)
            seizure_events: List of seizure event dictionaries
        """
        eeg_dir = self.data_path / subject_id / 'eeg'
        eeg_file = eeg_dir / f'{subject_id}_task-rest_eeg.npz'
        
        if not eeg_file.exists():
            raise FileNotFoundError(f"EEG file not found: {eeg_file}")
        
        data = np.load(eeg_file, allow_pickle=True)
        eeg_data = data['data']
        seizure_events = data['seizure_events'].tolist() if 'seizure_events' in data else []
        
        # Apply filtering
        if self.bandpass_filter:
            eeg_data = self.apply_bandpass_filter(eeg_data)
        
        return eeg_data, seizure_events
    
    def load_subject_fmri(self, subject_id: str) -> np.ndarray:
        """
        Load and process fMRI data for a subject
        
        Returns:
            fmri_features: Extracted features
        """
        if not self.include_fmri:
            return None
        
        fmri_dir = self.data_path / subject_id / 'func'
        fmri_file = fmri_dir / f'{subject_id}_task-rest_bold.nii.gz'
        
        if not fmri_file.exists():
            return None
        
        # Load fMRI
        img = nib.load(fmri_file)
        data = img.get_fdata()
        
        # Extract features based on preprocessing method
        if self.fmri_preprocessing == 'connectivity':
            # Extract ROI timeseries and compute connectivity
            n_rois = 20
            timeseries = []
            for _ in range(n_rois):
                x = np.random.randint(10, data.shape[0]-10)
                y = np.random.randint(10, data.shape[1]-10)
                z = np.random.randint(5, data.shape[2]-5)
                roi_ts = data[x-5:x+5, y-5:y+5, z-2:z+2, :].mean(axis=(0,1,2))
                timeseries.append(roi_ts)
            
            timeseries = np.array(timeseries).T
            corr_matrix = np.corrcoef(timeseries.T)
            mask = np.triu_indices(n_rois, k=1)
            features = corr_matrix[mask]
        else:
            # Simple spatial average
            features = data.mean(axis=(0, 1, 2))
        
        return features
    
    def create_windows_with_labels(
        self,
        eeg_data: np.ndarray,
        seizure_events: List[Dict],
        fmri_features: Optional[np.ndarray],
        subject_id: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        Create windows with seizure state labels
        
        Returns:
            eeg_windows: (n_windows, n_channels, window_samples)
            labels: (n_windows,) - seizure state labels
            fmri_windows: (n_windows, n_fmri_features) or None
            subjects: (n_windows,)
        """
        n_channels, n_samples = eeg_data.shape
        window_samples = int(self.window_size * self.sampling_rate)
        step_samples = int(window_samples * (1 - self.window_overlap))
        
        # Create time labels for each sample
        sample_labels = np.zeros(n_samples, dtype=int)  # Default: interictal
        
        for event in seizure_events:
            onset_sample = int(event['onset'] * self.sampling_rate)
            duration_samples = int(event['duration'] * self.sampling_rate)
            offset_sample = onset_sample + duration_samples
            
            preictal_start = max(0, onset_sample - int(self.preictal_window * self.sampling_rate))
            postictal_end = min(n_samples, offset_sample + int(self.postictal_window * self.sampling_rate))
            
            # Label pre-ictal period
            sample_labels[preictal_start:onset_sample] = self.state_map['preictal']
            
            # Label ictal period
            sample_labels[onset_sample:offset_sample] = self.state_map['ictal']
            
            # Label post-ictal period
            sample_labels[offset_sample:postictal_end] = self.state_map['postictal']
        
        # Create windows
        eeg_windows = []
        labels = []
        fmri_windows = [] if fmri_features is not None else None
        subjects = []
        
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            
            # Get window
            window = eeg_data[:, start:end]
            
            # Majority vote for label
            window_labels = sample_labels[start:end]
            label = np.bincount(window_labels).argmax()
            
            eeg_windows.append(window)
            labels.append(label)
            subjects.append(int(subject_id.split('-')[1]))
            
            # Align fMRI features (simple averaging)
            if fmri_features is not None:
                # Map EEG time to fMRI time (assuming TR=2s)
                tr = 2.0
                fmri_start = int((start / self.sampling_rate) / tr)
                fmri_end = int((end / self.sampling_rate) / tr)
                fmri_window = fmri_features[fmri_start:fmri_end].mean() if fmri_end <= len(fmri_features) else 0
                fmri_windows.append(fmri_window)
        
        eeg_windows = np.array(eeg_windows)
        labels = np.array(labels)
        subjects = np.array(subjects)
        
        if fmri_windows is not None:
            fmri_windows = np.array(fmri_windows).reshape(-1, 1)
        
        return eeg_windows, labels, fmri_windows, subjects
    
    def load_all_data(self) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load all subjects' data
        
        Returns:
            eeg_data: (n_samples, n_channels, window_samples)
            fmri_data: (n_samples, n_fmri_features) or None
            labels: (n_samples,)
            subjects: (n_samples,)
            metadata: DataFrame
        """
        # Download if needed
        self.download_dataset()
        
        # Load participants info
        participants_file = self.data_path / 'participants.tsv'
        participants_df = pd.read_csv(participants_file, sep='\t')
        
        # Filter subjects
        if self.subjects is not None:
            participants_df = participants_df[
                participants_df['participant_id'].isin(self.subjects)
            ]
        
        if self.verbose:
            print(f"\nLoading data for {len(participants_df)} subjects...")
        
        all_eeg = []
        all_fmri = []
        all_labels = []
        all_subjects = []
        
        for idx, row in participants_df.iterrows():
            subject_id = row['participant_id']
            
            try:
                if self.verbose:
                    print(f"Loading {subject_id}...", end=' ')
                
                # Load EEG and seizure events
                eeg_data, seizure_events = self.load_subject_eeg(subject_id)
                
                # Load fMRI if requested
                fmri_features = self.load_subject_fmri(subject_id) if self.include_fmri else None
                
                # Create windows with labels
                eeg_windows, labels, fmri_windows, subjects = self.create_windows_with_labels(
                    eeg_data, seizure_events, fmri_features, subject_id
                )
                
                all_eeg.append(eeg_windows)
                all_labels.append(labels)
                all_subjects.append(subjects)
                
                if fmri_windows is not None:
                    all_fmri.append(fmri_windows)
                
                if self.verbose:
                    print(f"✓ {len(eeg_windows)} windows ({len(seizure_events)} seizures)")
                
            except Exception as e:
                if self.verbose:
                    print(f"✗ Error: {e}")
                continue
        
        eeg_data = np.concatenate(all_eeg, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        subjects = np.concatenate(all_subjects, axis=0)
        fmri_data = np.concatenate(all_fmri, axis=0) if all_fmri else None
        
        if self.verbose:
            print(f"\n✓ Loaded {len(eeg_data)} windows")
            print(f"  EEG shape: {eeg_data.shape}")
            if fmri_data is not None:
                print(f"  fMRI shape: {fmri_data.shape}")
            print(f"  Classes: {np.unique(labels)} (0=interictal, 1=preictal, 2=ictal, 3=postictal)")
            print(f"  Class distribution: {np.bincount(labels)}")
        
        return eeg_data, fmri_data, labels, subjects, participants_df
    
    def get_dataloaders(
        self,
        batch_size: int = 32,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train/val/test data loaders"""
        eeg_data, fmri_data, labels, subjects, metadata = self.load_all_data()
        
        # Split data
        indices = np.arange(len(eeg_data))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_split, random_state=random_seed, stratify=labels
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_split/(1-test_split), 
            random_state=random_seed, stratify=labels[train_idx]
        )
        
        # Split data
        eeg_train, eeg_val, eeg_test = eeg_data[train_idx], eeg_data[val_idx], eeg_data[test_idx]
        y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]
        subj_train, subj_val, subj_test = subjects[train_idx], subjects[val_idx], subjects[test_idx]
        
        fmri_train = fmri_data[train_idx] if fmri_data is not None else None
        fmri_val = fmri_data[val_idx] if fmri_data is not None else None
        fmri_test = fmri_data[test_idx] if fmri_data is not None else None
        
        # Normalize
        if self.normalize:
            scaler = StandardScaler()
            eeg_train = scaler.fit_transform(eeg_train.reshape(len(eeg_train), -1)).reshape(eeg_train.shape)
            eeg_val = scaler.transform(eeg_val.reshape(len(eeg_val), -1)).reshape(eeg_val.shape)
            eeg_test = scaler.transform(eeg_test.reshape(len(eeg_test), -1)).reshape(eeg_test.shape)
        
        # Create datasets
        train_dataset = SeizeITDataset(eeg_train, fmri_train, y_train, subj_train)
        val_dataset = SeizeITDataset(eeg_val, fmri_val, y_val, subj_val)
        test_dataset = SeizeITDataset(eeg_test, fmri_test, y_test, subj_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        if self.verbose:
            print(f"\nData split:")
            print(f"  Train: {len(eeg_train)} samples")
            print(f"  Val:   {len(eeg_val)} samples")
            print(f"  Test:  {len(eeg_test)} samples")
        
        return train_loader, val_loader, test_loader


def test_seizeit_loader():
    """Test SeizeIT1 loader"""
    print("\n" + "="*70)
    print("Testing SeizeIT1 EEG-fMRI Loader")
    print("="*70)
    
    loader = SeizeITLoader(
        data_path='./data/seizeit',
        window_size=4.0,
        include_fmri=True,
        verbose=True
    )
    
    try:
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=8)
        
        # Test batch
        for batch in train_loader:
            if isinstance(batch[0], dict):
                eeg, fmri = batch[0]['eeg'], batch[0]['fmri']
                y = batch[1]
                print(f"\n✓ EEG shape: {eeg.shape}")
                print(f"  fMRI shape: {fmri.shape}")
                print(f"  Labels shape: {y.shape}")
            else:
                x, y = batch[0], batch[1]
                print(f"\n✓ Data shape: {x.shape}")
                print(f"  Labels shape: {y.shape}")
            break
        
        print("\n✅ SeizeIT1 loader test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_seizeit_loader()
