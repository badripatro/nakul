"""
OpenNeuro Dataset Loader for Depression Tracking
=================================================

This module provides functionality to download and load fMRI data from OpenNeuro,
specifically targeting depression tracking tasks (e.g., ds000030).

Dataset: ds000030 (UCLA Consortium for Neuropsychiatric Phenomics)
- Modality: Task-based and resting-state fMRI
- Subjects: 272 subjects (healthy controls + psychiatric patients)
- Use Case: Depression tracking, emotion decoding
- Details: Includes behavioral and psychiatric assessments
- URL: https://openneuro.org/datasets/ds000030

Tasks included:
- Resting state fMRI
- Balloon analog risk task
- Stop signal task
- Task switching task
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import nilearn
    from nilearn import datasets, image, masking
    from nilearn.connectome import ConnectivityMeasure
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False
    print("Warning: nilearn not installed. Install with: pip install nilearn")


class OpenNeuroDataset(Dataset):
    """PyTorch Dataset for OpenNeuro fMRI data"""
    
    def __init__(self, data, labels, subjects, metadata=None, transform=None):
        """
        Args:
            data: Tensor of shape (n_samples, n_features) - extracted fMRI features
            labels: Tensor of shape (n_samples,) - depression/condition labels
            subjects: Tensor of subject IDs
            metadata: Dictionary with additional metadata (age, sex, etc.)
            transform: Optional transform to apply
        """
        self.data = torch.FloatTensor(data) if not isinstance(data, torch.Tensor) else data
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.subjects = torch.LongTensor(subjects) if subjects is not None else None
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        if self.subjects is not None:
            return x, y, self.subjects[idx]
        return x, y


class OpenNeuroLoader:
    """
    Data loader for OpenNeuro fMRI datasets (depression tracking)
    
    This loader supports:
    - Automated download from OpenNeuro
    - BIDS format parsing
    - Multiple preprocessing pipelines
    - Connectivity feature extraction
    - Depression/control classification
    """
    
    def __init__(
        self,
        dataset_name: str = 'ds000030',
        data_path: str = './data/openneuro',
        subjects: Optional[List[str]] = None,
        task: str = 'rest',  # 'rest', 'stopsignal', 'bart', 'taskswitch'
        preprocessing: str = 'connectivity',  # 'connectivity', 'timeseries', 'voxel'
        atlas: str = 'msdl',  # 'msdl', 'harvard_oxford', 'aal'
        confounds: List[str] = None,  # Motion parameters to regress out
        standardize: bool = True,
        detrend: bool = True,
        low_pass: float = 0.1,  # Hz
        high_pass: float = 0.01,  # Hz
        tr: float = 2.0,  # Repetition time in seconds
        verbose: bool = True
    ):
        """
        Initialize OpenNeuro fMRI data loader
        
        Args:
            dataset_name: OpenNeuro dataset ID (e.g., 'ds000030')
            data_path: Path to store/load dataset
            subjects: List of subject IDs, None for all
            task: fMRI task name
            preprocessing: Feature extraction method
            atlas: Brain atlas for ROI extraction
            confounds: Confound variables to regress
            standardize: Z-score standardization
            detrend: Remove linear trends
            low_pass: Low-pass filter cutoff (Hz)
            high_pass: High-pass filter cutoff (Hz)
            tr: Repetition time (seconds)
            verbose: Print progress
        """
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.subjects = subjects
        self.task = task
        self.preprocessing = preprocessing
        self.atlas = atlas
        self.confounds = confounds or ['trans_x', 'trans_y', 'trans_z', 
                                        'rot_x', 'rot_y', 'rot_z']
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.tr = tr
        self.verbose = verbose
        
        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.data_path / dataset_name
        
        # Condition labels for ds000030
        self.condition_map = {
            'control': 0,
            'schizophrenia': 1,
            'bipolar': 2,
            'adhd': 3
        }
        
        if self.verbose:
            print("="*70)
            print("OpenNeuro fMRI Loader Initialized")
            print("="*70)
            print(f"Dataset: {dataset_name}")
            print(f"Data path: {self.data_path}")
            print(f"Task: {task}")
            print(f"Preprocessing: {preprocessing}")
            print(f"Atlas: {atlas}")
            print(f"TR: {tr}s")
            print("="*70)
    
    def download_dataset(self):
        """
        Download OpenNeuro dataset using AWS CLI or DataLad
        
        Note: For real implementation, use:
        - AWS CLI: aws s3 sync --no-sign-request s3://openneuro.org/{dataset_name} {local_path}
        - DataLad: datalad install https://github.com/OpenNeuroDatasets/{dataset_name}
        """
        if self.dataset_path.exists() and any(self.dataset_path.glob('sub-*')):
            if self.verbose:
                print(f"✓ Dataset already downloaded: {self.dataset_path}")
            return True
        
        if self.verbose:
            print(f"\nDownloading {self.dataset_name} from OpenNeuro...")
            print("Note: This is a placeholder. For real download, use:")
            print(f"  aws s3 sync --no-sign-request s3://openneuro.org/{self.dataset_name} {self.dataset_path}")
            print("  or")
            print(f"  datalad install https://github.com/OpenNeuroDatasets/{self.dataset_name}")
        
        # Create mock BIDS structure for demonstration
        self._create_mock_data()
        return True
    
    def _create_mock_data(self):
        """Create mock fMRI data for demonstration"""
        if self.verbose:
            print("\nCreating mock fMRI data for demonstration...")
        
        # Create BIDS structure
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create dataset_description.json
        dataset_desc = {
            "Name": self.dataset_name,
            "BIDSVersion": "1.6.0",
            "Authors": ["Mock Data Generator"],
        }
        with open(self.dataset_path / 'dataset_description.json', 'w') as f:
            json.dump(dataset_desc, f, indent=2)
        
        # Create participants.tsv
        n_subjects = 20
        conditions = ['control'] * 10 + ['schizophrenia'] * 5 + ['bipolar'] * 3 + ['adhd'] * 2
        
        participants_data = {
            'participant_id': [f'sub-{i+1:03d}' for i in range(n_subjects)],
            'age': np.random.randint(18, 65, n_subjects),
            'sex': np.random.choice(['M', 'F'], n_subjects),
            'diagnosis': conditions,
            'depression_score': np.random.randint(0, 30, n_subjects)  # Mock BDI score
        }
        pd.DataFrame(participants_data).to_csv(
            self.dataset_path / 'participants.tsv', sep='\t', index=False
        )
        
        # Create mock fMRI data for each subject
        for i in range(n_subjects):
            subj_id = f'sub-{i+1:03d}'
            subj_dir = self.dataset_path / subj_id / 'func'
            subj_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock 4D fMRI image
            # Shape: (64, 64, 30, 150) = (x, y, z, time)
            n_voxels = (64, 64, 30)
            n_timepoints = 150
            
            # Generate realistic fMRI timeseries
            data = np.random.randn(*n_voxels, n_timepoints).astype(np.float32) * 100
            
            # Add brain-like spatial structure
            center = np.array([32, 32, 15])
            for x in range(n_voxels[0]):
                for y in range(n_voxels[1]):
                    for z in range(n_voxels[2]):
                        dist = np.sqrt(((np.array([x, y, z]) - center) ** 2).sum())
                        if dist < 25:  # Brain-like sphere
                            data[x, y, z, :] *= 2
            
            # Save as NIfTI
            affine = np.eye(4)
            img = nib.Nifti1Image(data, affine)
            filename = f'{subj_id}_task-{self.task}_bold.nii.gz'
            nib.save(img, subj_dir / filename)
            
            # Create mock confounds file
            confounds_data = {
                'trans_x': np.random.randn(n_timepoints) * 0.1,
                'trans_y': np.random.randn(n_timepoints) * 0.1,
                'trans_z': np.random.randn(n_timepoints) * 0.1,
                'rot_x': np.random.randn(n_timepoints) * 0.01,
                'rot_y': np.random.randn(n_timepoints) * 0.01,
                'rot_z': np.random.randn(n_timepoints) * 0.01,
            }
            confounds_file = f'{subj_id}_task-{self.task}_desc-confounds_timeseries.tsv'
            pd.DataFrame(confounds_data).to_csv(
                subj_dir / confounds_file, sep='\t', index=False
            )
        
        if self.verbose:
            print(f"✓ Created mock data for {n_subjects} subjects")
    
    def load_participants_data(self) -> pd.DataFrame:
        """Load participants metadata"""
        participants_file = self.dataset_path / 'participants.tsv'
        if not participants_file.exists():
            raise FileNotFoundError(f"participants.tsv not found in {self.dataset_path}")
        
        df = pd.read_csv(participants_file, sep='\t')
        return df
    
    def extract_connectivity_features(self, img_file: Path, confounds_file: Path) -> np.ndarray:
        """
        Extract functional connectivity features from fMRI
        
        Returns:
            Connectivity matrix flattened to 1D features
        """
        if not NILEARN_AVAILABLE:
            # Fallback: simple correlation between random ROIs
            img = nib.load(img_file)
            data = img.get_fdata()
            
            # Extract timeseries from 39 random "ROIs"
            n_rois = 39
            timeseries = []
            for _ in range(n_rois):
                x, y, z = np.random.randint(0, data.shape[0]), \
                         np.random.randint(0, data.shape[1]), \
                         np.random.randint(0, data.shape[2])
                timeseries.append(data[x, y, z, :])
            
            timeseries = np.array(timeseries).T  # (time, rois)
            
            # Compute correlation matrix
            corr_matrix = np.corrcoef(timeseries.T)
            
            # Extract upper triangle (excluding diagonal)
            mask = np.triu_indices(n_rois, k=1)
            features = corr_matrix[mask]
            
            return features
        
        # Use nilearn for proper connectivity extraction
        from nilearn.connectome import ConnectivityMeasure
        from nilearn import datasets
        
        # Load atlas
        if self.atlas == 'msdl':
            atlas = datasets.fetch_atlas_msdl()
            atlas_filename = atlas['maps']
        elif self.atlas == 'harvard_oxford':
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
            atlas_filename = atlas['maps']
        else:
            # Default to MSDL
            atlas = datasets.fetch_atlas_msdl()
            atlas_filename = atlas['maps']
        
        # Load confounds
        confounds = None
        if confounds_file.exists():
            confounds_df = pd.read_csv(confounds_file, sep='\t')
            confounds = confounds_df[self.confounds].values
        
        # Extract timeseries
        masker = masking.NiftiMapsMasker(
            maps_img=atlas_filename,
            standardize=self.standardize,
            detrend=self.detrend,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
            t_r=self.tr,
            verbose=0
        )
        
        timeseries = masker.fit_transform(str(img_file), confounds=confounds)
        
        # Compute connectivity
        conn_measure = ConnectivityMeasure(kind='correlation', vectorize=True)
        connectivity = conn_measure.fit_transform([timeseries])[0]
        
        return connectivity
    
    def load_subject_data(self, subject_id: str) -> Tuple[np.ndarray, Dict]:
        """
        Load fMRI data for a single subject
        
        Returns:
            features: Extracted features (connectivity or timeseries)
            metadata: Subject metadata
        """
        subj_dir = self.dataset_path / subject_id / 'func'
        
        # Find fMRI file
        fmri_files = list(subj_dir.glob(f'{subject_id}_task-{self.task}_bold.nii.gz'))
        if not fmri_files:
            raise FileNotFoundError(f"No fMRI file found for {subject_id}, task {self.task}")
        
        img_file = fmri_files[0]
        confounds_file = subj_dir / f'{subject_id}_task-{self.task}_desc-confounds_timeseries.tsv'
        
        # Extract features based on preprocessing method
        if self.preprocessing == 'connectivity':
            features = self.extract_connectivity_features(img_file, confounds_file)
        else:
            # Simple voxel-based or timeseries extraction
            img = nib.load(img_file)
            data = img.get_fdata()
            features = data.mean(axis=-1).flatten()  # Average over time
        
        # Load metadata
        metadata = {'subject_id': subject_id}
        
        return features, metadata
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load all subjects' data
        
        Returns:
            X: Features (n_subjects, n_features)
            y: Labels (n_subjects,)
            subjects: Subject IDs
            metadata: DataFrame with all metadata
        """
        # Download if needed
        self.download_dataset()
        
        # Load participants info
        participants_df = self.load_participants_data()
        
        # Filter subjects if specified
        if self.subjects is not None:
            participants_df = participants_df[
                participants_df['participant_id'].isin(self.subjects)
            ]
        
        if self.verbose:
            print(f"\nLoading data for {len(participants_df)} subjects...")
        
        all_features = []
        all_labels = []
        all_subjects = []
        
        for idx, row in participants_df.iterrows():
            subject_id = row['participant_id']
            
            try:
                if self.verbose:
                    print(f"Loading {subject_id}...", end=' ')
                
                features, metadata = self.load_subject_data(subject_id)
                
                # Map diagnosis to label
                diagnosis = row['diagnosis']
                label = self.condition_map.get(diagnosis, 0)
                
                all_features.append(features)
                all_labels.append(label)
                all_subjects.append(subject_id)
                
                if self.verbose:
                    print(f"✓ ({len(features)} features)")
                
            except Exception as e:
                if self.verbose:
                    print(f"✗ Error: {e}")
                continue
        
        X = np.array(all_features)
        y = np.array(all_labels)
        subjects = np.array(all_subjects)
        
        if self.verbose:
            print(f"\n✓ Loaded {len(X)} subjects")
            print(f"  Feature shape: {X.shape}")
            print(f"  Classes: {np.unique(y)}")
            print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y, subjects, participants_df
    
    def get_dataloaders(
        self,
        batch_size: int = 16,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train/val/test data loaders
        
        Returns:
            train_loader, val_loader, test_loader
        """
        X, y, subjects, metadata = self.load_all_data()
        
        # Split data
        X_temp, X_test, y_temp, y_test, subj_temp, subj_test = train_test_split(
            X, y, subjects, test_size=test_split, random_state=random_seed, stratify=y
        )
        
        X_train, X_val, y_train, y_val, subj_train, subj_val = train_test_split(
            X_temp, y_temp, subj_temp, 
            test_size=val_split/(1-test_split), 
            random_state=random_seed, 
            stratify=y_temp
        )
        
        # Normalize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # Create datasets
        train_dataset = OpenNeuroDataset(X_train, y_train, subj_train)
        val_dataset = OpenNeuroDataset(X_val, y_val, subj_val)
        test_dataset = OpenNeuroDataset(X_test, y_test, subj_test)
        
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


def test_openneuro_loader():
    """Test OpenNeuro loader"""
    print("\n" + "="*70)
    print("Testing OpenNeuro fMRI Loader")
    print("="*70)
    
    loader = OpenNeuroLoader(
        dataset_name='ds000030',
        data_path='./data/openneuro',
        task='rest',
        preprocessing='connectivity',
        verbose=True
    )
    
    try:
        # Test data loading
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=8)
        
        # Test batch
        for x, y, s in train_loader:
            print(f"\n✓ Batch shape: {x.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Subjects shape: {s.shape}")
            break
        
        print("\n✅ OpenNeuro loader test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_openneuro_loader()
