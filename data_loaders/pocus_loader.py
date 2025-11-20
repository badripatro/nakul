"""
POCUS (Point-of-Care Ultrasound) Dataset Loader
================================================

This module provides functionality to load and process POCUS datasets
for cardiac function staging and vital archetypes.

Dataset: POCUS Dataset (Stanford AIMI)
- Modality: Point-of-care ultrasound (cardiac, lung)
- Use Case: Cardiac function staging, symbolic vital archetypes
- Details: Annotated ultrasound clips with diagnostic labels
- Includes: Ejection fraction estimation, wall motion abnormalities
- URL: https://stanfordaimi.azurewebsites.net/datasets/

Cardiac assessments:
- Ejection Fraction (EF): Normal (>50%), Moderate (30-50%), Reduced (<30%)
- Wall Motion: Normal, Hypokinetic, Akinetic, Dyskinetic
- Valvular function assessments
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not installed. Install with: pip install opencv-python")


class POCUSDataset(Dataset):
    """PyTorch Dataset for POCUS ultrasound videos/images"""
    
    def __init__(self, sequences, labels, metadata=None, transform=None):
        """
        Args:
            sequences: Tensor of shape (n_samples, n_frames, height, width) or (n_samples, n_frames, channels, height, width)
            labels: Tensor of shape (n_samples,) - diagnostic labels
            metadata: Additional metadata (EF, wall motion, etc.)
            transform: Optional transform
        """
        self.sequences = sequences
        self.labels = torch.LongTensor(labels) if not isinstance(labels, torch.Tensor) else labels
        self.metadata = metadata
        self.transform = transform
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        y = self.labels[idx]
        
        if self.transform:
            # Apply transform to each frame
            frames = []
            for i in range(seq.shape[0]):
                frame = seq[i]
                transformed = self.transform(image=frame.numpy())
                frames.append(torch.FloatTensor(transformed['image']))
            seq = torch.stack(frames)
        
        return seq, y


class POCUSLoader:
    """
    Data loader for POCUS (Point-of-Care Ultrasound) dataset
    
    Supports:
    - Cardiac ultrasound video loading
    - Ejection fraction classification
    - Wall motion assessment
    - Temporal sequence processing
    - Frame sampling strategies
    """
    
    def __init__(
        self,
        data_path: str = './data/POCUS',
        view_type: str = 'cardiac',  # 'cardiac', 'lung', 'abdominal'
        assessment_type: str = 'ef',  # 'ef' (ejection fraction), 'wall_motion', 'valve'
        image_size: Tuple[int, int] = (224, 224),
        n_frames: int = 16,  # Number of frames to sample per clip
        frame_sampling: str = 'uniform',  # 'uniform', 'random', 'keyframe'
        grayscale: bool = True,
        normalize: bool = True,
        augment: bool = False,
        verbose: bool = True
    ):
        """
        Initialize POCUS loader
        
        Args:
            data_path: Path to POCUS dataset
            view_type: Type of ultrasound view
            assessment_type: Type of clinical assessment
            image_size: Target frame size (height, width)
            n_frames: Number of frames to sample from each video
            frame_sampling: Strategy for sampling frames
            grayscale: Convert to grayscale
            normalize: Apply normalization
            augment: Apply data augmentation
            verbose: Print progress
        """
        self.data_path = Path(data_path)
        self.view_type = view_type
        self.assessment_type = assessment_type
        self.image_size = image_size
        self.n_frames = n_frames
        self.frame_sampling = frame_sampling
        self.grayscale = grayscale
        self.normalize = normalize
        self.augment = augment
        self.verbose = verbose
        
        # Label mappings
        if assessment_type == 'ef':
            # Ejection Fraction classification
            self.class_map = {
                'normal': 0,      # EF > 50%
                'moderate': 1,    # EF 30-50%
                'reduced': 2      # EF < 30%
            }
        elif assessment_type == 'wall_motion':
            # Wall motion abnormality
            self.class_map = {
                'normal': 0,
                'hypokinetic': 1,
                'akinetic': 2,
                'dyskinetic': 3
            }
        else:
            # Generic binary classification
            self.class_map = {
                'normal': 0,
                'abnormal': 1
            }
        
        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("="*70)
            print("POCUS Ultrasound Loader Initialized")
            print("="*70)
            print(f"Data path: {self.data_path}")
            print(f"View type: {view_type}")
            print(f"Assessment: {assessment_type}")
            print(f"Image size: {image_size}")
            print(f"Frames per clip: {n_frames}")
            print("="*70)
    
    def download_dataset(self):
        """
        Download POCUS dataset
        
        Note: Requires Stanford AIMI access or similar
        """
        if self.data_path.exists() and any(self.data_path.glob('*')):
            if self.verbose:
                print(f"✓ Dataset already exists: {self.data_path}")
            return True
        
        if self.verbose:
            print(f"\nDownloading POCUS dataset...")
            print("Note: This requires access to Stanford AIMI or similar platforms.")
            print("Access information:")
            print("  - Stanford AIMI: https://stanfordaimi.azurewebsites.net/")
            print("  - Some POCUS datasets on PhysioNet or OpenICPSR")
            print("\nCreating mock data for demonstration...")
        
        # Create mock data
        self._create_mock_data()
        return True
    
    def _create_mock_data(self):
        """Create mock ultrasound video data"""
        if self.verbose:
            print("\nCreating mock POCUS data...")
        
        # Create class directories
        if self.assessment_type == 'ef':
            classes = ['normal', 'moderate', 'reduced']
            class_counts = [50, 30, 20]
        elif self.assessment_type == 'wall_motion':
            classes = ['normal', 'hypokinetic', 'akinetic', 'dyskinetic']
            class_counts = [40, 30, 20, 10]
        else:
            classes = ['normal', 'abnormal']
            class_counts = [50, 50]
        
        # Create metadata file
        metadata = {
            'video_id': [],
            'class': [],
            'ef_value': [],
            'heart_rate': [],
            'view': []
        }
        
        video_id = 0
        
        for class_name, count in zip(classes, class_counts):
            class_dir = self.data_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(count):
                video_id += 1
                
                # Create mock ultrasound video (30 frames)
                n_total_frames = 30
                frames = []
                
                for frame_idx in range(n_total_frames):
                    # Create ultrasound-like frame
                    frame = np.random.rand(256, 256) * 255
                    
                    # Add sector shape (characteristic of ultrasound)
                    mask = np.zeros((256, 256), dtype=np.float32)
                    for y in range(256):
                        width = int((y / 256) * 256)
                        start_x = (256 - width) // 2
                        end_x = start_x + width
                        mask[y, start_x:end_x] = 1.0
                    
                    frame = frame * mask
                    
                    # Add cardiac motion (pulsating circle for heart chamber)
                    phase = (frame_idx / n_total_frames) * 2 * np.pi
                    
                    if class_name == 'normal':
                        # Good contraction (EF > 50%)
                        radius_systole = 30
                        radius_diastole = 60
                    elif class_name == 'moderate':
                        # Moderate contraction (EF 30-50%)
                        radius_systole = 40
                        radius_diastole = 60
                    else:  # reduced or abnormal
                        # Poor contraction (EF < 30%)
                        radius_systole = 50
                        radius_diastole = 60
                    
                    # Animate between systole and diastole
                    radius = radius_systole + (radius_diastole - radius_systole) * (0.5 + 0.5 * np.sin(phase))
                    
                    center_x, center_y = 128, 180
                    
                    for x in range(256):
                        for y in range(256):
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            if dist < radius:
                                frame[y, x] = np.random.randint(50, 150)
                    
                    # Add speckle noise
                    speckle = np.random.gamma(1, 1, (256, 256))
                    frame = frame * speckle
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    frames.append(frame)
                
                # Save as video or image sequence
                video_path = class_dir / f'video_{video_id:04d}.npy'
                np.save(video_path, np.array(frames))
                
                # Generate mock EF value
                if class_name == 'normal':
                    ef_value = np.random.randint(50, 70)
                elif class_name == 'moderate':
                    ef_value = np.random.randint(30, 50)
                else:
                    ef_value = np.random.randint(15, 30)
                
                # Update metadata
                metadata['video_id'].append(video_id)
                metadata['class'].append(class_name)
                metadata['ef_value'].append(ef_value)
                metadata['heart_rate'].append(np.random.randint(60, 100))
                metadata['view'].append(self.view_type)
        
        # Save metadata
        pd.DataFrame(metadata).to_csv(self.data_path / 'metadata.csv', index=False)
        
        if self.verbose:
            print(f"✓ Created mock data: {video_id} videos")
    
    def sample_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Sample frames from video
        
        Args:
            frames: Array of shape (n_frames, height, width)
            
        Returns:
            Sampled frames of shape (n_frames_target, height, width)
        """
        n_total_frames = len(frames)
        
        if n_total_frames <= self.n_frames:
            # Repeat frames if not enough
            indices = np.linspace(0, n_total_frames-1, self.n_frames, dtype=int)
        elif self.frame_sampling == 'uniform':
            # Uniformly sample frames
            indices = np.linspace(0, n_total_frames-1, self.n_frames, dtype=int)
        elif self.frame_sampling == 'random':
            # Randomly sample frames
            indices = np.sort(np.random.choice(n_total_frames, self.n_frames, replace=False))
        else:
            # Default to uniform
            indices = np.linspace(0, n_total_frames-1, self.n_frames, dtype=int)
        
        return frames[indices]
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame"""
        # Resize
        if CV2_AVAILABLE:
            frame = cv2.resize(frame, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            frame_pil = Image.fromarray(frame)
            frame_pil = frame_pil.resize(self.image_size[::-1], Image.BILINEAR)
            frame = np.array(frame_pil)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def load_video(self, video_path: Path) -> np.ndarray:
        """
        Load and preprocess video
        
        Returns:
            Video sequence of shape (n_frames, height, width)
        """
        # Load frames
        if video_path.suffix == '.npy':
            frames = np.load(video_path)
        elif video_path.suffix in ['.mp4', '.avi']:
            # Load video file
            if not CV2_AVAILABLE:
                raise ImportError("opencv-python required for video loading")
            
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.grayscale:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(frame)
            cap.release()
            frames = np.array(frames)
        else:
            raise ValueError(f"Unsupported video format: {video_path.suffix}")
        
        # Sample frames
        frames = self.sample_frames(frames)
        
        # Preprocess each frame
        processed_frames = []
        for frame in frames:
            frame = self.preprocess_frame(frame)
            processed_frames.append(frame)
        
        return np.array(processed_frames)
    
    def load_class_data(self, class_name: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load all videos from a class
        
        Returns:
            sequences: List of video sequences
            labels: List of labels
        """
        class_dir = self.data_path / class_name
        
        if not class_dir.exists():
            return [], []
        
        # Find all videos
        video_files = list(class_dir.glob('*.npy')) + list(class_dir.glob('*.mp4')) + list(class_dir.glob('*.avi'))
        
        sequences = []
        labels = []
        
        for video_file in video_files:
            try:
                # Load video
                seq = self.load_video(video_file)
                sequences.append(seq)
                
                # Assign label
                label = self.class_map[class_name]
                labels.append(label)
                
            except Exception as e:
                if self.verbose:
                    print(f"  Error loading {video_file.name}: {e}")
                continue
        
        return sequences, labels
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load all data
        
        Returns:
            X: Sequences (n_samples, n_frames, height, width)
            y: Labels (n_samples,)
            metadata: DataFrame with metadata
        """
        # Download if needed
        self.download_dataset()
        
        if self.verbose:
            print("\nLoading POCUS dataset...")
        
        all_sequences = []
        all_labels = []
        
        for class_name in self.class_map.keys():
            if self.verbose:
                print(f"Loading {class_name}...", end=' ')
            
            sequences, labels = self.load_class_data(class_name)
            
            all_sequences.extend(sequences)
            all_labels.extend(labels)
            
            if self.verbose:
                print(f"✓ {len(sequences)} videos")
        
        # Convert to arrays
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        # Load metadata
        metadata_file = self.data_path / 'metadata.csv'
        metadata = pd.read_csv(metadata_file) if metadata_file.exists() else pd.DataFrame()
        
        # Add channel dimension: (n_samples, n_frames, 1, height, width)
        if len(X.shape) == 4:
            X = X[:, :, np.newaxis, :, :]
        
        # Normalize
        if self.normalize:
            mean = X.mean()
            std = X.std()
            X = (X - mean) / (std + 1e-8)
        
        if self.verbose:
            print(f"\n✓ Loaded {len(X)} sequences")
            print(f"  Sequence shape: {X.shape}")
            print(f"  Classes: {np.unique(y)}")
            print(f"  Class distribution: {np.bincount(y)}")
        
        return X, y, metadata
    
    def get_dataloaders(
        self,
        batch_size: int = 8,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train/val/test data loaders"""
        X, y, metadata = self.load_all_data()
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_seed, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_split/(1-test_split), 
            random_state=random_seed, 
            stratify=y_temp
        )
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_val = torch.FloatTensor(X_val)
        X_test = torch.FloatTensor(X_test)
        
        # Create datasets
        train_dataset = POCUSDataset(X_train, y_train)
        val_dataset = POCUSDataset(X_val, y_val)
        test_dataset = POCUSDataset(X_test, y_test)
        
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


def test_pocus_loader():
    """Test POCUS loader"""
    print("\n" + "="*70)
    print("Testing POCUS Ultrasound Loader")
    print("="*70)
    
    loader = POCUSLoader(
        data_path='./data/POCUS',
        view_type='cardiac',
        assessment_type='ef',
        n_frames=8,
        image_size=(112, 112),
        verbose=True
    )
    
    try:
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=4)
        
        # Test batch
        for x, y in train_loader:
            print(f"\n✓ Sequence shape: {x.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
            break
        
        print("\n✅ POCUS loader test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_pocus_loader()
