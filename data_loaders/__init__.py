"""
Unified Data Loader Interface for Multi-Dataset EEG Benchmarking
================================================================

Provides a unified interface for loading different EEG datasets:
- BCI-IV-2a: Motor imagery
- SEED: Emotion recognition (62 channels, 3 classes)
- FACED: Emotion recognition with video (32 channels, 7 classes)
"""

from typing import Tuple, Optional, Dict, Any
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_loaders.bci_loader import BCIDataLoader
from data_loaders.seed_loader import SEEDDataLoader
from data_loaders.faced_loader import FACEDDataLoader
from data_loaders.busi_loader import BUSILoader
from data_loaders.seizeit_loader import SeizeITLoader
from data_loaders.openneuro_loader import OpenNeuroLoader


# Dataset configurations
DATASET_CONFIGS = {
    'bci': {
        'name': 'BCI Competition IV 2a',
        'n_channels': 22,
        'n_classes': 4,
        'sampling_freq': 250,
        'n_timepoints': 1000,
        'task': 'motor_imagery',
        'loader_class': BCIDataLoader
    },
    'seed': {
        'name': 'SEED Emotion Dataset',
        'n_channels': 62,
        'n_classes': 3,
        'sampling_freq': 250,
        'window_size': 2.0,
        'task': 'emotion_recognition',
        'loader_class': SEEDDataLoader
    },
    'faced': {
        'name': 'FACED Emotion Dataset',
        'n_channels': 30,  # 30 EEG channels (or 32 with mastoid)
        'n_classes': 7,
        'sampling_freq': 250,
        'chunk_size': 500,  # 2 seconds at 250 Hz
        'task': 'emotion_recognition',
        'loader_class': FACEDDataLoader
    },
    'busi': {
        'name': 'BUSI Breast Ultrasound',
        'n_channels': 3,  # RGB channels
        'n_classes': 3,  # normal, benign, malignant
        'image_size': (224, 224),
        'modality': 'ultrasound',
        'task': 'tumor_classification',
        'loader_class': BUSILoader
    },
    'seizeit': {
        'name': 'SeizeIT1 EEG-fMRI',
        'n_channels': 64,  # EEG channels
        'n_classes': 4,  # interictal, preictal, ictal, postictal
        'sampling_freq': 250,
        'n_timepoints': 500,  # samples per window
        'modality': 'eeg_fmri',
        'task': 'seizure_prediction',
        'loader_class': SeizeITLoader
    },
    'openneuro': {
        'name': 'OpenNeuro ds000030 fMRI',
        'n_channels': None,  # fMRI voxels
        'n_classes': 2,  # control vs patient (or task-based)
        'tr': None,  # Variable per sequence
        'modality': 'fmri',
        'task': 'depression_tracking',
        'loader_class': OpenNeuroLoader
    }
}


class UnifiedDataLoader:
    """
    Unified interface for loading different EEG datasets
    """
    
    def __init__(
        self,
        dataset_name: str,
        data_path: Optional[str] = None,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize unified data loader
        
        Args:
            dataset_name: Dataset name ('bci', 'seed', 'faced')
            data_path: Path to dataset (if None, uses default)
            batch_size: Batch size for data loaders
            **kwargs: Additional dataset-specific arguments
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_CONFIGS.keys())}"
            )
        
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        self.batch_size = batch_size
        
        # Set default data path if not provided
        if data_path is None:
            data_path = self._get_default_data_path()
        
        # Initialize dataset-specific loader
        loader_class = self.config['loader_class']
        
        if dataset_name == 'bci':
            self.loader = loader_class(data_path=data_path)
        elif dataset_name == 'seed':
            window_size = kwargs.get('window_size', self.config.get('window_size', 2.0))
            window_overlap = kwargs.get('window_overlap', 0.5)
            subjects = kwargs.get('subjects', None)
            
            self.loader = loader_class(
                data_path=data_path,
                window_size=window_size,
                window_overlap=window_overlap,
                subjects=subjects
            )
        elif dataset_name == 'faced':
            # TorchEEG-compatible parameters
            chunk_size = kwargs.get('chunk_size', self.config.get('chunk_size', 500))
            overlap = kwargs.get('overlap', 0)
            num_channel = kwargs.get('num_channel', self.config.get('n_channels', 30))
            subjects = kwargs.get('subjects', None)
            
            # Legacy window_size parameter support
            if 'window_size' in kwargs:
                window_size = kwargs['window_size']
                chunk_size = int(window_size * self.config['sampling_freq'])
            
            self.loader = loader_class(
                data_path=data_path,
                chunk_size=chunk_size,
                overlap=overlap,
                num_channel=num_channel,
                subjects=subjects
            )
        elif dataset_name == 'busi':
            image_size = kwargs.get('image_size', self.config.get('image_size', (224, 224)))
            use_masks = kwargs.get('use_masks', True)
            augment = kwargs.get('augment', False)
            
            self.loader = loader_class(
                data_path=data_path,
                image_size=image_size,
                use_masks=use_masks,
                augment=augment
            )
        elif dataset_name == 'seizeit':
            subjects = kwargs.get('subjects', None)
            window_size = kwargs.get('window_size', 4.0)
            window_overlap = kwargs.get('window_overlap', 0.5)
            include_fmri = kwargs.get('include_fmri', True)
            
            self.loader = loader_class(
                data_path=data_path,
                subjects=subjects,
                window_size=window_size,
                window_overlap=window_overlap,
                include_fmri=include_fmri
            )
        elif dataset_name == 'openneuro':
            subjects = kwargs.get('subjects', None)
            task = kwargs.get('task', 'rest')
            preprocessing = kwargs.get('preprocessing', 'connectivity')
            
            self.loader = loader_class(
                data_path=data_path,
                subjects=subjects,
                task=task,
                preprocessing=preprocessing
            )
        
        print(f"\nUnified Data Loader initialized for: {self.config['name']}")
        if self.config.get('n_channels'):
            print(f"  Channels: {self.config['n_channels']}")
        print(f"  Classes: {self.config['n_classes']}")
        print(f"  Task: {self.config['task']}")
    
    def _get_default_data_path(self) -> str:
        """Get default data path for dataset"""
        base_path = Path(__file__).parent.parent.parent / 'data'
        
        if self.dataset_name == 'bci':
            return str(base_path)
        elif self.dataset_name == 'seed':
            return str(base_path / 'SEED')
        elif self.dataset_name == 'faced':
            return str(Path(__file__).parent.parent / 'data' / 'Processed_data')
        elif self.dataset_name == 'busi':
            return str(base_path / 'BUSI')
        elif self.dataset_name == 'seizeit':
            return str(base_path / 'seizeit')
        elif self.dataset_name == 'openneuro':
            return str(base_path / 'openneuro' / 'ds000030')
        
        return str(base_path)
    
    def get_dataloaders(
        self,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train/val/test data loaders
        
        Args:
            val_split: Validation split ratio
            test_split: Test split ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            train_loader, val_loader, test_loader
        """
        return self.loader.get_dataloaders(
            batch_size=self.batch_size,
            val_split=val_split,
            test_split=test_split,
            random_seed=random_seed
        )
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        info = {
            'name': self.config['name'],
            'dataset_key': self.dataset_name,
            'n_channels': self.config.get('n_channels'),
            'n_classes': self.config['n_classes'],
            'task': self.config['task'],
            'batch_size': self.batch_size
        }
        
        # Add optional fields if they exist
        if 'sampling_freq' in self.config:
            info['sampling_freq'] = self.config['sampling_freq']
        if 'modality' in self.config:
            info['modality'] = self.config['modality']
        if 'image_size' in self.config:
            info['image_size'] = self.config['image_size']
            
        return info
    
    @staticmethod
    def get_available_datasets() -> Dict[str, Dict[str, Any]]:
        """Get information about all available datasets"""
        return DATASET_CONFIGS.copy()
    
    @staticmethod
    def print_available_datasets():
        """Print information about all available datasets"""
        print("\n" + "="*70)
        print("Available Datasets")
        print("="*70)
        
        for key, config in DATASET_CONFIGS.items():
            print(f"\n{key.upper()}: {config['name']}")
            if config.get('n_channels'):
                print(f"  Channels: {config['n_channels']}")
            print(f"  Classes: {config['n_classes']}")
            if config.get('sampling_freq'):
                print(f"  Sampling: {config['sampling_freq']} Hz")
            if config.get('modality'):
                print(f"  Modality: {config['modality']}")
            print(f"  Task: {config['task']}")
        
        print("\n" + "="*70)


def get_dataset_loader(
    dataset_name: str,
    data_path: Optional[str] = None,
    batch_size: int = 32,
    **kwargs
) -> UnifiedDataLoader:
    """
    Convenience function to get a dataset loader
    
    Args:
        dataset_name: Dataset name ('bci', 'seed', 'faced')
        data_path: Path to dataset
        batch_size: Batch size
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        UnifiedDataLoader instance
    """
    return UnifiedDataLoader(
        dataset_name=dataset_name,
        data_path=data_path,
        batch_size=batch_size,
        **kwargs
    )


def test_unified_loader():
    """Test unified data loader"""
    print("Testing Unified Data Loader...")
    print("-" * 70)
    
    # Print available datasets
    UnifiedDataLoader.print_available_datasets()
    
    # Test BCI loader
    print("\n\nTest 1: BCI Dataset")
    print("-" * 70)
    try:
        bci_loader = UnifiedDataLoader('bci', batch_size=32)
        info = bci_loader.get_dataset_info()
        print(f"✓ BCI loader initialized")
        print(f"  Info: {info}")
    except Exception as e:
        print(f"✗ BCI loader failed: {e}")
    
    # Test SEED loader
    print("\n\nTest 2: SEED Dataset")
    print("-" * 70)
    try:
        seed_loader = UnifiedDataLoader(
            'seed',
            batch_size=32,
            window_size=2.0,
            subjects=[1, 2]
        )
        info = seed_loader.get_dataset_info()
        print(f"✓ SEED loader initialized")
        print(f"  Info: {info}")
    except Exception as e:
        print(f"✗ SEED loader failed: {e}")
    
    # Test FACED loader
    print("\n\nTest 3: FACED Dataset")
    print("-" * 70)
    try:
        faced_loader = UnifiedDataLoader(
            'faced',
            batch_size=32,
            window_size=2.0
        )
        info = faced_loader.get_dataset_info()
        print(f"✓ FACED loader initialized")
        print(f"  Info: {info}")
    except Exception as e:
        print(f"✗ FACED loader failed: {e}")
    
    print("\n" + "="*70)
    print("Unified Data Loader Tests Complete")
    print("="*70)


if __name__ == "__main__":
    test_unified_loader()
