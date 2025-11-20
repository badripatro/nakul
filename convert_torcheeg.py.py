# Databricks notebook source
#!/usr/bin/env python3
"""
Convert FACED Dataset to TorchEEG Format
=========================================

This script converts the current Processed_data format (pure numpy arrays)
to TorchEEG-compatible format (dictionaries with labels).

Input format:  sub000.pkl → np.ndarray (28, 32, 7500)
Output format: sub000.pkl → {'data': array, 'emotion': array, 'label': array, ...}
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_subject_file(input_file, output_file, subject_id, verbose=False):
    """
    Convert a single subject file to TorchEEG format
    
    Args:
        input_file: Path to input .pkl file
        output_file: Path to output .pkl file
        subject_id: Subject ID number
        verbose: Print details
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load original file
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        # Check if already in correct format
        if isinstance(data, dict) and 'data' in data and 'label' in data:
            if verbose:
                print(f"  ✓ Subject {subject_id:03d} already in TorchEEG format")
            # Just copy it
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        
        # Convert numpy array to TorchEEG format
        if isinstance(data, np.ndarray):
            n_trials = data.shape[0]
            
            # Generate labels based on FACED structure
            # 7 emotions × 4 trials = 28 trials
            if n_trials == 28:
                labels = np.repeat(np.arange(7), 4)
            elif n_trials % 7 == 0:
                trials_per_emotion = n_trials // 7
                labels = np.repeat(np.arange(7), trials_per_emotion)
            else:
                # Unknown structure
                labels = np.tile(np.arange(7), (n_trials // 7) + 1)[:n_trials]
            
            # Create TorchEEG-compatible dictionary
            torcheeg_data = {
                'data': data,                           # (n_trials, n_channels, n_samples)
                'emotion': labels,                      # (n_trials,) - emotion labels
                'label': labels,                        # Alias for 'emotion'
                'subject': subject_id,                  # Subject ID
                'trial_id': np.arange(n_trials),       # Trial indices
                'n_trials': n_trials,
                'n_channels': data.shape[1],
                'n_samples': data.shape[2],
                'sampling_rate': 250,                  # Hz
                'emotion_names': {
                    0: 'Neutral',
                    1: 'Happy',
                    2: 'Sad',
                    3: 'Angry',
                    4: 'Fearful',
                    5: 'Disgusted',
                    6: 'Surprised'
                }
            }
            
            # Save in TorchEEG format
            with open(output_file, 'wb') as f:
                pickle.dump(torcheeg_data, f)
            
            if verbose:
                print(f"  ✓ Subject {subject_id:03d}: {n_trials} trials, labels: {np.unique(labels)}")
            
            return True
        else:
            print(f"  ✗ Subject {subject_id:03d}: Unknown format {type(data)}")
            return False
            
    except Exception as e:
        print(f"  ✗ Subject {subject_id:03d}: Error - {e}")
        return False


def convert_dataset(input_dir, output_dir, verbose=False):
    """
    Convert entire FACED dataset to TorchEEG format
    
    Args:
        input_dir: Input directory with sub*.pkl files
        output_dir: Output directory for converted files
        verbose: Print details
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("FACED Dataset → TorchEEG Format Conversion")
    print("="*70)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()
    
    # Find all subject files
    subject_files = sorted(input_path.glob('sub*.pkl'))
    
    if not subject_files:
        print(f"✗ No sub*.pkl files found in {input_path}")
        return False
    
    print(f"Found {len(subject_files)} subject files")
    print()
    
    # Convert each file
    success_count = 0
    fail_count = 0
    
    for subject_file in tqdm(subject_files, desc="Converting"):
        # Extract subject ID from filename
        try:
            subject_id = int(subject_file.stem[3:])  # 'sub000' → 0
        except ValueError:
            print(f"✗ Could not parse subject ID from {subject_file.name}")
            fail_count += 1
            continue
        
        # Convert file
        output_file = output_path / subject_file.name
        
        if convert_subject_file(subject_file, output_file, subject_id, verbose):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("="*70)
    print("Conversion Complete")
    print("="*70)
    print(f"✓ Successful: {success_count} files")
    if fail_count > 0:
        print(f"✗ Failed:     {fail_count} files")
    print()
    
    # Create info file
    info_file = output_path / 'README.txt'
    with open(info_file, 'w') as f:
        f.write("FACED Dataset - TorchEEG Format\n")
        f.write("="*70 + "\n\n")
        f.write(f"Converted from: {input_path}\n")
        f.write(f"Total files: {success_count}\n")
        f.write(f"Format: TorchEEG-compatible dictionaries\n\n")
        f.write("File structure:\n")
        f.write("  sub000.pkl, sub001.pkl, ..., sub122.pkl\n\n")
        f.write("Each file contains:\n")
        f.write("  {\n")
        f.write("    'data': np.ndarray (n_trials, n_channels, n_samples),\n")
        f.write("    'emotion': np.ndarray (n_trials,) - labels 0-6,\n")
        f.write("    'label': np.ndarray (n_trials,) - same as emotion,\n")
        f.write("    'subject': int - subject ID,\n")
        f.write("    'trial_id': np.ndarray - trial indices,\n")
        f.write("    ...\n")
        f.write("  }\n\n")
        f.write("Usage with TorchEEG:\n")
        f.write("  from torcheeg.datasets import FACEDDataset\n")
        f.write("  dataset = FACEDDataset(root_path='./Processed_data_torcheeg')\n")
    
    print(f"ℹ️  Info file created: {info_file}")
    print()
    
    # Test loading one file
    print("Testing converted format...")
    try:
        test_file = output_path / 'sub000.pkl'
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        print(f"✓ Loaded {test_file.name}")
        print(f"  Keys: {list(test_data.keys())}")
        print(f"  Data shape: {test_data['data'].shape}")
        print(f"  Labels: {test_data['emotion']}")
        print(f"  Label distribution: {np.bincount(test_data['emotion'])}")
        print()
        print("✅ Conversion successful!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Convert FACED dataset to TorchEEG format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default paths
  python3 convert_to_torcheeg_format.py
  
  # Convert with custom paths
  python3 convert_to_torcheeg_format.py \\
      --input ../data/Processed_data \\
      --output ../data/Processed_data_torcheeg
  
  # Verbose output
  python3 convert_to_torcheeg_format.py --verbose

After conversion, use with TorchEEG:
  from torcheeg.datasets import FACEDDataset
  dataset = FACEDDataset(root_path='../data/Processed_data_torcheeg')
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='../data/Processed_data',
        help='Input directory with original .pkl files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../data/Processed_data_torcheeg',
        help='Output directory for TorchEEG-compatible files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    # Convert
    try:
        success = convert_dataset(args.input, args.output, args.verbose)
        
        if success:
            print()
            print("="*70)
            print("Next Steps:")
            print("="*70)
            print()
            print("1. Test with TorchEEG:")
            print(f"   cd {args.output}")
            print("   python3 -c \"")
            print("   from torcheeg.datasets import FACEDDataset")
            print("   from torcheeg import transforms")
            print(f"   dataset = FACEDDataset(root_path='{args.output}')")
            print("   print(f'Dataset size: {len(dataset)}')\"")
            print()
            print("2. Use in your code:")
            print("   from torcheeg.datasets import FACEDDataset")
            print("   dataset = FACEDDataset(")
            print(f"       root_path='{args.output}',")
            print("       chunk_size=250,")
            print("       online_transform=transforms.ToTensor()")
            print("   )")
            print()
            
            sys.exit(0)
        else:
            print("\n✗ Conversion failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
