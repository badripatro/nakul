#!/usr/bin/env python3
"""
Test script for multi-dataset data loaders
"""

import sys
import os
from pathlib import Path

# Get the directory of this script
script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Change to script directory to ensure relative imports work
os.chdir(script_dir)

from data_loaders import UnifiedDataLoader, DATASET_CONFIGS


def test_unified_loader():
    """Test unified data loader interface"""
    print("\n" + "="*70)
    print("Testing Unified Data Loader Interface")
    print("="*70)
    
    # Print available datasets
    UnifiedDataLoader.print_available_datasets()
    
    # Test 1: Check dataset configs
    print("\n\nTest 1: Dataset Configurations")
    print("-" * 70)
    for name, config in DATASET_CONFIGS.items():
        print(f"{name}:")
        for key, value in config.items():
            if key != 'loader_class':
                print(f"  {key}: {value}")
    print("✓ Dataset configurations loaded successfully")
    
    # Test 2: Initialize BCI loader
    print("\n\nTest 2: BCI Dataset Loader Initialization")
    print("-" * 70)
    try:
        bci_loader = UnifiedDataLoader('bci', batch_size=32)
        info = bci_loader.get_dataset_info()
        print(f"✓ BCI loader initialized")
        print(f"  Name: {info['name']}")
        print(f"  Channels: {info['n_channels']}")
        print(f"  Classes: {info['n_classes']}")
        print(f"  Sampling: {info['sampling_freq']} Hz")
    except Exception as e:
        print(f"✗ BCI loader failed: {e}")
    
    # Test 3: Initialize SEED loader
    print("\n\nTest 3: SEED Dataset Loader Initialization")
    print("-" * 70)
    try:
        seed_loader = UnifiedDataLoader(
            'seed',
            batch_size=32,
            window_size=2.0,
            window_overlap=0.5
        )
        info = seed_loader.get_dataset_info()
        print(f"✓ SEED loader initialized")
        print(f"  Name: {info['name']}")
        print(f"  Channels: {info['n_channels']}")
        print(f"  Classes: {info['n_classes']}")
        print(f"  Sampling: {info['sampling_freq']} Hz")
    except Exception as e:
        print(f"✗ SEED loader failed: {e}")
    
    # Test 4: Initialize FACED loader
    print("\n\nTest 4: FACED Dataset Loader Initialization")
    print("-" * 70)
    try:
        faced_loader = UnifiedDataLoader(
            'faced',
            batch_size=32,
            window_size=2.0,
            window_overlap=0.5
        )
        info = faced_loader.get_dataset_info()
        print(f"✓ FACED loader initialized")
        print(f"  Name: {info['name']}")
        print(f"  Channels: {info['n_channels']}")
        print(f"  Classes: {info['n_classes']}")
        print(f"  Sampling: {info['sampling_freq']} Hz")
    except Exception as e:
        print(f"✗ FACED loader failed: {e}")
    
    # Test 5: Test error handling
    print("\n\nTest 5: Error Handling")
    print("-" * 70)
    try:
        invalid_loader = UnifiedDataLoader('invalid_dataset')
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)


def test_seed_loader_detailed():
    """Test SEED loader in detail if data is available"""
    from data_loaders.seed_loader import SEEDDataLoader
    
    print("\n" + "="*70)
    print("Testing SEED Loader (Detailed)")
    print("="*70)
    
    try:
        loader = SEEDDataLoader(
            data_path='./data/SEED',
            subjects=[1],
            window_size=2.0,
            window_overlap=0.5,
            target_freq=250
        )
        
        print("\n✓ SEED loader created")
        print(f"  Original freq: {loader.original_freq} Hz")
        print(f"  Target freq: {loader.target_freq} Hz")
        print(f"  Channels: {loader.n_channels}")
        print(f"  Classes: {loader.n_classes}")
        print(f"  Emotion map: {loader.emotion_map}")
        
    except FileNotFoundError as e:
        print(f"\n⚠ SEED data not found: {e}")
        print("  To test SEED loader, download data from:")
        print("  https://bcmi.sjtu.edu.cn/~seed/")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def test_faced_loader_detailed():
    """Test FACED loader in detail if data is available"""
    from data_loaders.faced_loader import FACEDDataLoader
    
    print("\n" + "="*70)
    print("Testing FACED Loader (Detailed)")
    print("="*70)
    
    try:
        loader = FACEDDataLoader(
            data_path='./data/FACED',
            window_size=2.0,
            window_overlap=0.5,
            sampling_freq=250
        )
        
        print("\n✓ FACED loader created")
        print(f"  Sampling freq: {loader.sampling_freq} Hz")
        print(f"  Channels: {loader.n_channels}")
        print(f"  Classes: {loader.n_classes}")
        print(f"  Emotion names: {loader.emotion_names}")
        
    except FileNotFoundError as e:
        print(f"\n⚠ FACED data not found: {e}")
        print("  To test FACED loader, download data from:")
        print("  https://github.com/OpenGVLab/FACED")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    # Test unified loader
    test_unified_loader()
    
    # Test individual loaders if data is available
    test_seed_loader_detailed()
    test_faced_loader_detailed()
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Download SEED dataset from: https://bcmi.sjtu.edu.cn/~seed/")
    print("  2. Download FACED dataset from: https://github.com/OpenGVLab/FACED")
    print("  3. Place datasets in data/SEED and data/FACED directories")
    print("  4. Run training: python train_all_models.py --dataset all")
    print("="*70 + "\n")
