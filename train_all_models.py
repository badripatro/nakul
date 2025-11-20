# Databricks notebook source
"""Unified Training Script for Multiple EEG Models on Multiple Datasets
========================================================================

Trains a set of best-performing architectures sequentially on multiple datasets:

Models included:
  - EEGNet
  - DeepConvNet
  - ShallowConvNet
  - MambaEEGNet ("mamba")
  - SpectralMambaFixed ("spectral_mamba")
  - NAKUL ("nakul") - Multi-Scale State Space Models with Learned Frequency Bands, Dynamic Kernels, and Graph Spatial Mixing

Datasets supported:
  - BCI-IV-2a: Motor imagery (22 channels, 4 classes, 250Hz)
  - SEED: Emotion recognition (62 channels, 3 classes, 250Hz)
  - FACED: Emotion recognition with video (32 channels, 7 classes, 250Hz)

Features:
  - Multi-dataset support with unified interface
  - Single data load and stratified train/val/test split per dataset
  - Per-model hyperparameter dictionary (can override via CLI)
  - Consistent channel-wise normalization
  - Early stopping (patience configurable)
  - Checkpoint saving of best validation model
  - JSON results summary per model and dataset

Usage:
  # Train on BCI dataset only
  python train_all_models.py --dataset bci --models eegnet,nakul
  
  # Train on SEED dataset
  python train_all_models.py --dataset seed --models eegnet,nakul --window_size 2.0
  
  # Train on FACED dataset
  python train_all_models.py --dataset faced --models eegnet,nakul --window_size 2.0
  
  # Train on ALL datasets
  python train_all_models.py --dataset all --models eegnet,nakul

Outputs stored in: results_multi/<timestamp>/<dataset>/<model_name>/
"""

import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from models import create_model, DEFAULT_HYPERPARAMETERS
from data_loaders.bci_loader import BCIDataLoader
from data_loaders import UnifiedDataLoader, DATASET_CONFIGS


def normalize_eeg_channels(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True) + 1e-8
    return (x - mean) / std


def build_optimizer(model: nn.Module, cfg: Dict):
    name = cfg.get('optimizer', 'AdamW').lower()
    lr = cfg.get('learning_rate', 0.001)
    wd = cfg.get('weight_decay', 0.0)
    if name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer {name}")


def build_scheduler(optimizer, cfg: Dict, epochs: int, steps_per_epoch: int):
    name = cfg.get('scheduler', 'None')
    name_low = name.lower()
    if name_low == 'onecyclelr':
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.get('learning_rate', 0.001),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
            anneal_strategy='cos'
        )
    if name_low == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=cfg.get('eta_min', 1e-6)
        )
    if name_low == 'steplr':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=50,
            gamma=0.5
        )
    return None


def early_stopping_check(best_val, current_val, no_improve_epochs, patience, min_delta):
    if current_val > best_val + min_delta:
        return False, 0  # improved
    no_improve_epochs += 1
    if no_improve_epochs >= patience:
        return True, no_improve_epochs
    return False, no_improve_epochs


def train_single_model(
    model_name: str,
    dataset_name: str,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_channels: int,
    n_classes: int,
    n_samples: int,
    output_root: str,
    base_cfg: Dict,
    epochs_override: int = None,
    patience: int = 25,
    min_delta: float = 1e-3
) -> Dict:
    """
    Train a single model on a specific dataset
    
    Args:
        model_name: Name of the model
        dataset_name: Name of the dataset
        device: PyTorch device
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        n_channels: Number of EEG channels
        n_classes: Number of classes
        n_samples: Number of time samples
        output_root: Output directory
        base_cfg: Model configuration
        epochs_override: Override number of epochs
        patience: Early stopping patience
        min_delta: Minimum improvement threshold
        
    Returns:
        Dictionary with training results
    """
    cfg = base_cfg.copy()
    if epochs_override is not None:
        cfg['epochs'] = epochs_override

    epochs = cfg.get('epochs', 50)
    batch_size = cfg.get('batch_size', 32)
    lr = cfg.get('learning_rate', 0.001)
    weight_decay = cfg.get('weight_decay', 0.0)

    # Create model with dataset-specific parameters
    model = create_model(
        model_name,
        n_classes=n_classes,
        n_channels=n_channels,
        n_samples=n_samples
    )
    model.to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs, len(train_loader))
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    no_improve_epochs = 0

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch in train_loader:
            # Handle different batch formats
            if isinstance(batch, dict):
                # Dictionary format (e.g., BUSI returns dict directly)
                if 'eeg' in batch:
                    xb = batch['eeg']
                elif 'fmri' in batch:
                    xb = batch['fmri']
                elif 'image' in batch:
                    xb = batch['image']
                elif 'data' in batch:
                    xb = batch['data']
                else:
                    xb = batch[list(batch.keys())[0]]  # First data key
                yb = batch['label']
            elif len(batch) == 3:  # (data, label, subject/session)
                xb, yb, _ = batch
                # Check if xb is a dict (SeizeIT with fMRI)
                if isinstance(xb, dict):
                    xb = xb['eeg']  # Use EEG data for now
            elif len(batch) == 2:
                xb, yb = batch
                # Check if xb is a dict
                if isinstance(xb, dict):
                    xb = xb['eeg']  # Use EEG data for now
            else:
                xb, yb = batch
            
            xb = normalize_eeg_channels(xb.to(device))
            yb = yb.to(device)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if scheduler and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss_total = 0.0
            val_correct = 0
            val_total = 0
            for batch in val_loader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    if 'eeg' in batch:
                        xb = batch['eeg']
                    elif 'fmri' in batch:
                        xb = batch['fmri']
                    elif 'image' in batch:
                        xb = batch['image']
                    elif 'data' in batch:
                        xb = batch['data']
                    else:
                        xb = batch[list(batch.keys())[0]]
                    yb = batch['label']
                elif len(batch) == 3:
                    xb, yb, _ = batch
                    if isinstance(xb, dict):
                        xb = xb['eeg']
                elif len(batch) == 2:
                    xb, yb = batch
                    if isinstance(xb, dict):
                        xb = xb['eeg']
                else:
                    xb, yb = batch
                
                xb = normalize_eeg_channels(xb.to(device))
                yb = yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss_total += loss.item()
                preds = out.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)
        
        val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total
        
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        improved = val_acc > best_val_acc + min_delta
        if improved:
            best_val_acc = val_acc
            best_state = model.state_dict()
            no_improve_epochs = 0
        else:
            stop, no_improve_epochs = early_stopping_check(
                best_val_acc, val_acc, no_improve_epochs, patience, min_delta
            )
            if stop:
                print(f"[{model_name}] Early stopping at epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{dataset_name}|{model_name}] Epoch {epoch}/{epochs} | "
                  f"Train Loss {train_loss:.4f} Acc {train_acc:.3f} | "
                  f"Val Loss {val_loss:.4f} Acc {val_acc:.3f} | LR {current_lr:.6f}")

    # Load best state for test
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    test_preds = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            # Handle different batch formats
            if isinstance(batch, dict):
                if 'eeg' in batch:
                    xb = batch['eeg']
                elif 'fmri' in batch:
                    xb = batch['fmri']
                elif 'image' in batch:
                    xb = batch['image']
                elif 'data' in batch:
                    xb = batch['data']
                else:
                    xb = batch[list(batch.keys())[0]]
                yb = batch['label']
            elif len(batch) == 3:
                xb, yb, _ = batch
                if isinstance(xb, dict):
                    xb = xb['eeg']
            elif len(batch) == 2:
                xb, yb = batch
                if isinstance(xb, dict):
                    xb = xb['eeg']
            else:
                xb, yb = batch
            
            xb = normalize_eeg_channels(xb.to(device))
            yb = yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            test_correct += (preds == yb).sum().item()
            test_total += yb.size(0)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(yb.cpu().numpy())
    
    test_acc = test_correct / test_total
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    conf_matrix = confusion_matrix(test_labels, test_preds)

    duration = time.time() - start_time

    # Prepare output path & save artifacts
    model_dir = os.path.join(output_root, dataset_name, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'state_dict': model.state_dict(),
        'best_val_acc': best_val_acc,
        'n_channels': n_channels,
        'n_classes': n_classes,
        'n_samples': n_samples
    }, os.path.join(model_dir, 'best_model.pt'))
    
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'epochs_trained': len(history['train_loss']),
            'duration_sec': duration,
            'confusion_matrix': conf_matrix.tolist()
        }, f, indent=2)

    print(f"\n[{dataset_name}|{model_name}] Results:")
    print(f"  Best Val Acc: {best_val_acc:.4f}")
    print(f"  Test Acc: {test_acc:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Duration: {duration:.2f}s")

    return {
        'model': model_name,
        'dataset': dataset_name,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'duration_sec': duration
    }


def main():
    parser = argparse.ArgumentParser(description="Train multiple EEG models on multiple datasets")
    parser.add_argument('--dataset', type=str, default='bci',
                        choices=['bci', 'seed', 'faced', 'busi', 'seizeit', 'openneuro', 'all'],
                        help='Dataset to use: bci, seed, faced, busi, seizeit, openneuro, or all')
    parser.add_argument('--models', type=str,
                        default='eegnet,deepconvnet,shallowconvnet,mamba,spectral_mamba,nakul',
                        help='Comma-separated list of model names')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base data directory')
    parser.add_argument('--output_root', type=str, default='results_multi',
                        help='Root output directory')
    parser.add_argument('--epochs_override', type=int, default=None,
                        help='Override epochs for all models')
    parser.add_argument('--patience', type=int, default=25,
                        help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Minimum improvement for early stopping')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--window_size', type=float, default=2.0,
                        help='Window size in seconds (for SEED/FACED)')
    parser.add_argument('--window_overlap', type=float, default=0.5,
                        help='Window overlap ratio (for SEED/FACED)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto/cpu/cuda')
    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device)
    print(f"Using device: {device}")

    # Timestamped root
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = os.path.join(args.output_root, timestamp)
    os.makedirs(output_root, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config['timestamp'] = timestamp
    config['device'] = str(device)
    with open(os.path.join(output_root, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Determine datasets to run
    if args.dataset == 'all':
        datasets_to_run = ['bci', 'seed', 'faced', 'busi', 'seizeit', 'openneuro']
    else:
        datasets_to_run = [args.dataset]
    
    print(f"\nDatasets to run: {datasets_to_run}")
    
    # Print available datasets
    print("\n" + "="*70)
    UnifiedDataLoader.print_available_datasets()
    
    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]
    print(f"\nModels to train: {selected_models}")

    all_results = []
    
    for dataset_name in datasets_to_run:
        print(f"\n{'='*70}")
        print(f"Training on dataset: {dataset_name.upper()}")
        print(f"{'='*70}\n")
        
        try:
            # Load dataset
            if dataset_name == 'bci':
                # Use existing BCI loader
                loader = BCIDataLoader(
                    data_path=args.data_dir,
                    subjects=list(range(1, 10)),
                    filter_data=True
                )
                X, y, subjects_data = loader.load_data()
                
                # Split data
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
                )
                
                train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
                val_ds = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
                test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
                
                train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
                
                n_channels = 22
                n_classes = 4
                n_samples = 1000
                
            else:
                # Use unified loader for SEED/FACED
                # Handle FACED special case - data is in Processed_data/
                if dataset_name == 'faced':
                    data_path = os.path.join(args.data_dir, 'Processed_data')
                else:
                    data_path = os.path.join(args.data_dir, dataset_name.upper())
                
                unified_loader = UnifiedDataLoader(
                    dataset_name=dataset_name,
                    data_path=data_path,
                    batch_size=args.batch_size,
                    window_size=args.window_size,
                    window_overlap=args.window_overlap
                )
                
                train_loader, val_loader, test_loader = unified_loader.get_dataloaders(
                    val_split=0.15,
                    test_split=0.15,
                    random_seed=42
                )
                
                dataset_info = unified_loader.get_dataset_info()
                n_channels = dataset_info.get('n_channels', 3)  # Default to 3 for images
                n_classes = dataset_info['n_classes']
                
                # Get actual data dimensions from a sample batch
                sample_batch = next(iter(train_loader))
                if isinstance(sample_batch, dict):
                    if 'eeg' in sample_batch:
                        sample_data = sample_batch['eeg']
                    elif 'fmri' in sample_batch:
                        sample_data = sample_batch['fmri']
                    elif 'image' in sample_batch:
                        sample_data = sample_batch['image']
                    else:
                        sample_data = sample_batch[list(sample_batch.keys())[0]]
                elif len(sample_batch) >= 2:
                    sample_data = sample_batch[0]
                    if isinstance(sample_data, dict):
                        sample_data = sample_data.get('eeg', sample_data.get('fmri', sample_data[list(sample_data.keys())[0]]))
                else:
                    sample_data = sample_batch
                
                # Extract actual dimensions from data
                if len(sample_data.shape) == 4:  # (batch, 1, channels, timepoints) or (batch, channels, height, width)
                    actual_channels = sample_data.shape[2]
                    actual_samples = sample_data.shape[3]
                elif len(sample_data.shape) == 3:  # (batch, channels, timepoints/features)
                    actual_channels = sample_data.shape[1]
                    actual_samples = sample_data.shape[2]
                else:
                    actual_channels = n_channels
                    actual_samples = 1000  # Default
                
                # Override with actual dimensions if they differ significantly
                if n_channels is None or n_channels != actual_channels:
                    print(f"  Note: Using actual data dimensions: channels={actual_channels}")
                    n_channels = actual_channels
                
                # Calculate samples per window (only for time-series data)
                if 'sampling_freq' in dataset_info and actual_samples > 100:
                    n_samples = actual_samples
                elif 'image_size' in dataset_info:
                    n_samples = dataset_info['image_size'][0] * dataset_info['image_size'][1]
                else:
                    n_samples = actual_samples if actual_samples > 100 else 1000
            
            print(f"\nDataset loaded: {dataset_name}")
            print(f"  Channels: {n_channels}")
            print(f"  Classes: {n_classes}")
            if 'sampling_freq' in dataset_info:
                print(f"  Samples per window: {n_samples}")
            else:
                print(f"  Input size: {n_samples}")
            print(f"  Training batches: {len(train_loader)}")
            print(f"  Validation batches: {len(val_loader)}")
            print(f"  Test batches: {len(test_loader)}")
            
        except Exception as e:
            print(f"✗ Error loading dataset {dataset_name}: {e}")
            print(f"Skipping dataset {dataset_name}")
            continue
        
        # Train all models on this dataset
        dataset_results = []
        for model_name in selected_models:
            base_cfg = DEFAULT_HYPERPARAMETERS.get(model_name)
            if base_cfg is None:
                print(f"Skipping unknown model '{model_name}'")
                continue
            
            print(f"\n{'='*70}")
            print(f"Training {model_name} on {dataset_name}")
            print(f"{'='*70}\n")
            
            try:
                model_result = train_single_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    device=device,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    n_channels=n_channels,
                    n_classes=n_classes,
                    n_samples=n_samples,
                    output_root=output_root,
                    base_cfg=base_cfg,
                    epochs_override=args.epochs_override,
                    patience=args.patience,
                    min_delta=args.min_delta
                )
                dataset_results.append(model_result)
                all_results.append(model_result)
            
            except Exception as e:
                print(f"✗ Error training {model_name} on {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save per-dataset results
        dataset_output_dir = os.path.join(output_root, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        with open(os.path.join(dataset_output_dir, 'results.json'), 'w') as f:
            json.dump(dataset_results, f, indent=2)
        
        # Print dataset summary
        print(f"\n{'='*70}")
        print(f"Summary for {dataset_name}:")
        print(f"{'='*70}")
        for r in dataset_results:
            print(f"  {r['model']:20s} | Val: {r['best_val_acc']:.4f} | Test: {r['test_acc']:.4f} | F1: {r['test_f1']:.4f}")

    # Save aggregated results
    with open(os.path.join(output_root, 'aggregate_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary table
    print(f"\n{'='*70}")
    print("FINAL SUMMARY - All Models & Datasets")
    print(f"{'='*70}\n")
    
    # Group by model
    models_dict = {}
    for r in all_results:
        if r['model'] not in models_dict:
            models_dict[r['model']] = {}
        models_dict[r['model']][r['dataset']] = {
            'test_acc': r['test_acc'],
            'test_f1': r['test_f1']
        }
    
    # Print table
    print(f"{'Model':<25} | {'Dataset':<10} | {'Test Acc':<10} | {'Test F1':<10}")
    print("-" * 70)
    for model_name in sorted(models_dict.keys()):
        for dataset_name in sorted(models_dict[model_name].keys()):
            result = models_dict[model_name][dataset_name]
            print(f"{model_name:<25} | {dataset_name:<10} | "
                  f"{result['test_acc']:.4f}     | {result['test_f1']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Training complete! Results saved to: {output_root}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
