"""
Test Script for All Medical Imaging Data Loaders
=================================================

This script tests all 5 medical imaging data loaders:
1. OpenNeuro (fMRI) - Depression tracking
2. SeizeIT1 (EEG-fMRI) - Seizure prediction
3. LIDC-IDRI (CT) - Lung cancer staging
4. BUSI (Ultrasound) - Breast tumor classification
5. POCUS (Ultrasound) - Cardiac function staging
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add data_loaders to path
sys.path.insert(0, str(Path(__file__).parent / 'data_loaders'))

from openneuro_loader import OpenNeuroLoader, test_openneuro_loader
from seizeit_loader import SeizeITLoader, test_seizeit_loader
from lidc_loader import LIDCLoader, test_lidc_loader
from busi_loader import BUSILoader, test_busi_loader
from pocus_loader import POCUSLoader, test_pocus_loader


def print_separator(title="", char="=", width=80):
    """Print a formatted separator"""
    if title:
        title = f" {title} "
        padding = (width - len(title)) // 2
        print(f"\n{char * padding}{title}{char * padding}")
    else:
        print(f"\n{char * width}")


def print_test_result(test_name, passed):
    """Print test result with emoji"""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\n{test_name}: {status}")


def test_all_loaders(quick_test=True):
    """
    Test all data loaders
    
    Args:
        quick_test: If True, uses mock data and small batch sizes
    """
    results = {}
    
    print_separator("Medical Imaging Data Loaders Test Suite", "=", 80)
    print("\nTesting all 5 medical imaging data loaders...")
    print("Mode:", "QUICK TEST (mock data)" if quick_test else "FULL TEST (real data)")
    
    # ========================================================================
    # Test 1: OpenNeuro fMRI Loader
    # ========================================================================
    print_separator("Test 1: OpenNeuro fMRI Loader (Depression Tracking)", "━", 80)
    
    try:
        loader = OpenNeuroLoader(
            dataset_name='ds000030',
            data_path='./data/openneuro',
            task='rest',
            preprocessing='connectivity',
            verbose=True
        )
        
        # Test data loading
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=4)
        
        # Test batch iteration
        for x, y, s in train_loader:
            print(f"\n✓ Batch loaded successfully")
            print(f"  Features shape: {x.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Subjects shape: {s.shape}")
            assert len(x.shape) == 2, "Expected 2D features"
            assert len(y.shape) == 1, "Expected 1D labels"
            break
        
        results['openneuro'] = True
        print_test_result("OpenNeuro fMRI Loader", True)
        
    except Exception as e:
        print(f"\n✗ OpenNeuro test failed: {e}")
        results['openneuro'] = False
        print_test_result("OpenNeuro fMRI Loader", False)
    
    # ========================================================================
    # Test 2: SeizeIT1 EEG-fMRI Loader
    # ========================================================================
    print_separator("Test 2: SeizeIT1 EEG-fMRI Loader (Seizure Prediction)", "━", 80)
    
    try:
        loader = SeizeITLoader(
            data_path='./data/seizeit',
            window_size=2.0,
            include_fmri=True,
            verbose=True
        )
        
        # Test data loading
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=4)
        
        # Test batch iteration
        for batch in train_loader:
            if isinstance(batch[0], dict):
                eeg, fmri = batch[0]['eeg'], batch[0]['fmri']
                y = batch[1]
                print(f"\n✓ Multi-modal batch loaded")
                print(f"  EEG shape: {eeg.shape}")
                print(f"  fMRI shape: {fmri.shape}")
                print(f"  Labels shape: {y.shape}")
                assert len(eeg.shape) == 3, "Expected 3D EEG data"
            else:
                x, y = batch[0], batch[1]
                print(f"\n✓ Batch loaded")
                print(f"  Data shape: {x.shape}")
                print(f"  Labels shape: {y.shape}")
            break
        
        results['seizeit'] = True
        print_test_result("SeizeIT1 EEG-fMRI Loader", True)
        
    except Exception as e:
        print(f"\n✗ SeizeIT1 test failed: {e}")
        results['seizeit'] = False
        print_test_result("SeizeIT1 EEG-fMRI Loader", False)
    
    # ========================================================================
    # Test 3: LIDC-IDRI CT Loader
    # ========================================================================
    print_separator("Test 3: LIDC-IDRI Lung CT Loader (Cancer Staging)", "━", 80)
    
    try:
        loader = LIDCLoader(
            data_path='./data/LIDC-IDRI',
            patch_size=(8, 32, 32),  # Smaller for quick test
            use_3d=True,
            verbose=True
        )
        
        # Test data loading
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=2)
        
        # Test batch iteration
        for x, y in train_loader:
            print(f"\n✓ CT batch loaded")
            print(f"  Patch shape: {x.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
            assert len(x.shape) == 4, "Expected 4D CT patches (batch, depth, height, width)"
            break
        
        results['lidc'] = True
        print_test_result("LIDC-IDRI CT Loader", True)
        
    except Exception as e:
        print(f"\n✗ LIDC-IDRI test failed: {e}")
        results['lidc'] = False
        print_test_result("LIDC-IDRI CT Loader", False)
    
    # ========================================================================
    # Test 4: BUSI Ultrasound Loader
    # ========================================================================
    print_separator("Test 4: BUSI Breast Ultrasound Loader (Tumor Classification)", "━", 80)
    
    try:
        loader = BUSILoader(
            data_path='./data/BUSI',
            image_size=(112, 112),  # Smaller for quick test
            use_masks=True,
            grayscale=True,
            verbose=True
        )
        
        # Test data loading
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=4)
        
        # Test batch iteration
        for batch in train_loader:
            if len(batch) == 3:
                x, mask, y = batch
                print(f"\n✓ Ultrasound batch with masks loaded")
                print(f"  Image shape: {x.shape}")
                print(f"  Mask shape: {mask.shape}")
                print(f"  Labels shape: {y.shape}")
            else:
                x, y = batch
                print(f"\n✓ Ultrasound batch loaded")
                print(f"  Image shape: {x.shape}")
                print(f"  Labels shape: {y.shape}")
            break
        
        results['busi'] = True
        print_test_result("BUSI Ultrasound Loader", True)
        
    except Exception as e:
        print(f"\n✗ BUSI test failed: {e}")
        results['busi'] = False
        print_test_result("BUSI Ultrasound Loader", False)
    
    # ========================================================================
    # Test 5: POCUS Cardiac Ultrasound Loader
    # ========================================================================
    print_separator("Test 5: POCUS Cardiac Ultrasound Loader (Function Staging)", "━", 80)
    
    try:
        loader = POCUSLoader(
            data_path='./data/POCUS',
            view_type='cardiac',
            assessment_type='ef',
            n_frames=4,  # Fewer frames for quick test
            image_size=(112, 112),
            verbose=True
        )
        
        # Test data loading
        train_loader, val_loader, test_loader = loader.get_dataloaders(batch_size=2)
        
        # Test batch iteration
        for x, y in train_loader:
            print(f"\n✓ Video sequence batch loaded")
            print(f"  Sequence shape: {x.shape}")
            print(f"  Labels shape: {y.shape}")
            print(f"  Value range: [{x.min():.3f}, {x.max():.3f}]")
            assert len(x.shape) >= 4, "Expected at least 4D video data"
            break
        
        results['pocus'] = True
        print_test_result("POCUS Ultrasound Loader", True)
        
    except Exception as e:
        print(f"\n✗ POCUS test failed: {e}")
        results['pocus'] = False
        print_test_result("POCUS Ultrasound Loader", False)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_separator("Test Summary", "=", 80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    failed_tests = total_tests - passed_tests
    
    print(f"\nTotal tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"\nSuccess rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed results:")
    for loader_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {loader_name:20s}: {status}")
    
    print_separator("", "=", 80)
    
    return all(results.values())


def benchmark_loaders():
    """Benchmark loading speed of all loaders"""
    import time
    
    print_separator("Benchmark: Data Loading Speed", "=", 80)
    
    benchmarks = {}
    
    # Benchmark each loader
    loaders_config = {
        'OpenNeuro': (OpenNeuroLoader, {'data_path': './data/openneuro', 'verbose': False}),
        'SeizeIT1': (SeizeITLoader, {'data_path': './data/seizeit', 'verbose': False}),
        'LIDC': (LIDCLoader, {'data_path': './data/LIDC-IDRI', 'patch_size': (8, 32, 32), 'verbose': False}),
        'BUSI': (BUSILoader, {'data_path': './data/BUSI', 'image_size': (112, 112), 'verbose': False}),
        'POCUS': (POCUSLoader, {'data_path': './data/POCUS', 'n_frames': 4, 'image_size': (112, 112), 'verbose': False})
    }
    
    for name, (LoaderClass, config) in loaders_config.items():
        try:
            print(f"\nBenchmarking {name}...")
            
            start_time = time.time()
            loader = LoaderClass(**config)
            train_loader, _, _ = loader.get_dataloaders(batch_size=8)
            
            # Time first batch
            batch_start = time.time()
            for batch in train_loader:
                batch_time = time.time() - batch_start
                break
            
            total_time = time.time() - start_time
            
            benchmarks[name] = {
                'total_time': total_time,
                'first_batch_time': batch_time
            }
            
            print(f"  Total loading time: {total_time:.2f}s")
            print(f"  First batch time: {batch_time:.3f}s")
            
        except Exception as e:
            print(f"  ✗ Benchmark failed: {e}")
    
    # Summary
    print_separator("Benchmark Summary", "-", 80)
    print(f"\n{'Loader':<15} {'Total Time':>12} {'Batch Time':>12}")
    print("-" * 40)
    for name, times in benchmarks.items():
        print(f"{name:<15} {times['total_time']:>10.2f}s {times['first_batch_time']:>10.3f}s")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test medical imaging data loaders')
    parser.add_argument('--full', action='store_true', help='Run full tests with real data')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark tests')
    parser.add_argument('--loader', type=str, choices=['openneuro', 'seizeit', 'lidc', 'busi', 'pocus'],
                       help='Test specific loader only')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_loaders()
    elif args.loader:
        # Test specific loader
        print_separator(f"Testing {args.loader} loader", "=", 80)
        
        if args.loader == 'openneuro':
            success = test_openneuro_loader()
        elif args.loader == 'seizeit':
            success = test_seizeit_loader()
        elif args.loader == 'lidc':
            success = test_lidc_loader()
        elif args.loader == 'busi':
            success = test_busi_loader()
        elif args.loader == 'pocus':
            success = test_pocus_loader()
        
        sys.exit(0 if success else 1)
    else:
        # Test all loaders
        quick_test = not args.full
        success = test_all_loaders(quick_test=quick_test)
        sys.exit(0 if success else 1)
