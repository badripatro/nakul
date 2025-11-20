# Databricks notebook source
#!/usr/bin/env python3
"""
Model Package for BCI Competition IV Dataset 2a
===============================================

This package provides comprehensive model functionality including:
- Model creation and initialization
- Model comparison and analysis
- Parameter counting and FLOP calculation
- Throughput benchmarking
- Recommended hyperparameters
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Optional, Union, Callable

# Import individual models
from .eegnet import EEGNet
from .deepconvnet import DeepConvNet
from .shallowconvnet import ShallowConvNet
from .mamba import MambaEEGNet
from .spectral_mamba_fixed import SpectralMambaMemory, SpectralMambaMemoryLite
from .nakul import NAKUL

# Model registry
MODELS = {
    'eegnet': EEGNet,
    'deepconvnet': DeepConvNet,
    'shallowconvnet': ShallowConvNet,
    'mamba': MambaEEGNet,
    'spectral_mamba': SpectralMambaMemory,
    'spectral_mamba_memory': SpectralMambaMemory,
    'spectral_mamba_lite': SpectralMambaMemoryLite,
    # NAKUL - Multi-Scale State Space Models with Learned Frequency Bands, Dynamic Kernels, and Graph Spatial Mixing
    'nakul': NAKUL,
}
# Default hyperparameters for each model
DEFAULT_HYPERPARAMETERS = {
    'eegnet': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'weight_decay': 1e-4,
        'dropout_rate': 0.25,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'epochs': 200
    },
    'deepconvnet': {
        'learning_rate': 0.0005,
        'batch_size': 32,
        'weight_decay': 5e-4,
        'dropout_rate': 0.5,
        'optimizer': 'AdamW',
        'scheduler': 'CosineAnnealingLR',
        'epochs': 300
    },
    'shallowconvnet': {
        'learning_rate': 0.001,
        'batch_size': 64,
        'weight_decay': 1e-4,
        'dropout_rate': 0.2,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'epochs': 150
    },
    'mamba': {
        'learning_rate': 0.0005,
        'batch_size': 32,
        'weight_decay': 1e-3,
        'dropout_rate': 0.1,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'epochs': 250
    },
    'spectral_mamba': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'weight_decay': 0.01,
        'dropout_rate': 0.3,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'epochs': 150
    },
    'spectral_mamba_memory': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'weight_decay': 0.01,
        'dropout_rate': 0.3,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'epochs': 150
    },
    'spectral_mamba_lite': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'weight_decay': 0.01,
        'dropout_rate': 0.3,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'epochs': 150
    },
    'nakul': {
        'learning_rate': 0.0008,
        'batch_size': 24,
        'weight_decay': 0.01,
        'dropout_rate': 0.2,
        'optimizer': 'AdamW',
        'scheduler': 'OneCycleLR',
        'epochs': 200,
        'eta_min': 1e-6
    }
}


def list_available_models() -> List[str]:
    """
    List all available models
    
    Returns:
        List[str]: List of model names
    """
    return list(MODELS.keys())


def create_model(model_name: str, n_classes: int = 4, n_channels: int = 22, 
                n_samples: int = 500, **kwargs) -> nn.Module:
    """
    Create a model instance
    
    Args:
        model_name (str): Name of the model
        n_classes (int): Number of output classes
        n_channels (int): Number of input channels
        n_samples (int): Number of time samples
        **kwargs: Additional model-specific arguments
        
    Returns:
        nn.Module: Model instance
    """
    if model_name.lower() not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODELS.keys())}")
    
    model_class = MODELS[model_name.lower()]
    
    # Create model with appropriate parameters
    try:
        model = model_class(
            n_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples,
            **kwargs
        )
    except TypeError:
        # Fallback for models with different parameter names
        try:
            model = model_class(
                num_classes=n_classes,
                num_channels=n_channels,
                num_samples=n_samples,
                **kwargs
            )
        except TypeError:
            # Generic fallback
            model = model_class(**kwargs)
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        Dict[str, int]: Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def calculate_flops(model: nn.Module, input_shape: tuple, device: str = 'cpu') -> int:
    """
    Calculate FLOPs (Floating Point Operations) for a model
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape (excluding batch dimension)
        device (str): Device to run calculation on
        
    Returns:
        int: Estimated FLOPs
    """
    model.eval()
    
    try:
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape, device=device)
        
        # Simple FLOP estimation
        param_count = sum(p.numel() for p in model.parameters())
        input_size = np.prod(input_shape)
        
        # Rough estimation: 2 operations per parameter per input element
        estimated_flops = 2 * param_count * input_size
        
        return estimated_flops
        
    except Exception as e:
        print(f"FLOP calculation failed: {e}")
        return 0


def benchmark_throughput(model: nn.Module, input_shape: tuple, 
                        batch_size: int = 32, num_runs: int = 50, 
                        device: str = 'cpu') -> Dict[str, float]:
    """
    Benchmark model throughput
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape (excluding batch dimension)
        batch_size (int): Batch size for benchmarking
        num_runs (int): Number of benchmark runs
        device (str): Device to run benchmark on
        
    Returns:
        Dict[str, float]: Throughput statistics
    """
    model.eval()
    model.to(device)
    
    # Create test data
    test_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            model(test_input)
    
    # Synchronize
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            model(test_input)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    throughput = batch_size / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'std_inference_time_ms': std_time * 1000,
        'throughput_samples_per_sec': throughput,
        'batch_size': batch_size,
        'num_runs': num_runs
    }


def get_model_memory_usage(model: nn.Module, input_shape: tuple, 
                          batch_size: int = 1, device: str = 'cuda') -> Dict[str, float]:
    """
    Estimate model memory usage
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input tensor shape
        batch_size (int): Batch size
        device (str): Device to estimate memory for
        
    Returns:
        Dict[str, float]: Memory usage statistics in MB
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    
    # Activation memory (rough estimate)
    input_memory = batch_size * np.prod(input_shape) * 4 / 1024**2  # float32
    
    # Gradient memory (same as parameters for backprop)
    gradient_memory = param_memory
    
    # Total estimated memory
    total_memory = param_memory + input_memory + gradient_memory
    
    return {
        'parameter_memory_mb': param_memory,
        'input_memory_mb': input_memory,
        'gradient_memory_mb': gradient_memory,
        'total_estimated_memory_mb': total_memory
    }


def get_model_summary(model_name: str, n_classes: int = 4, n_channels: int = 22, 
                     n_samples: int = 500, device: str = 'cpu') -> Dict:
    """
    Get comprehensive model summary
    
    Args:
        model_name (str): Name of the model
        n_classes (int): Number of classes
        n_channels (int): Number of channels
        n_samples (int): Number of samples
        device (str): Device to run analysis on
        
    Returns:
        Dict: Comprehensive model summary
    """
    try:
        # Create model
        model = create_model(model_name, n_classes, n_channels, n_samples)
        model.to(device)
        
        input_shape = (n_channels, n_samples)
        
        # Get parameter count
        param_info = count_parameters(model)
        
        # Test forward pass
        dummy_input = torch.randn(1, *input_shape, device=device)
        model.eval()
        
        with torch.no_grad():
            try:
                output = model(dummy_input)
                output_shape = tuple(output.shape)
                forward_pass_success = True
            except Exception as e:
                output_shape = f"Error: {str(e)}"
                forward_pass_success = False
        
        # Calculate FLOPs
        flops = calculate_flops(model, input_shape, device)
        
        # Benchmark throughput
        throughput_stats = benchmark_throughput(model, input_shape, batch_size=16, 
                                               num_runs=20, device=device)
        
        # Memory usage
        memory_stats = get_model_memory_usage(model, input_shape, batch_size=32, device=device)
        
        # Model size in MB
        model_size_mb = param_info['total_parameters'] * 4 / (1024 * 1024)  # float32
        
        summary = {
            'model_name': model_name,
            'model_class': model.__class__.__name__,
            'input_shape': (1, *input_shape),
            'output_shape': output_shape,
            'forward_pass_success': forward_pass_success,
            
            # Parameters
            'total_parameters': param_info['total_parameters'],
            'trainable_parameters': param_info['trainable_parameters'],
            'non_trainable_parameters': param_info['non_trainable_parameters'],
            'model_size_mb': model_size_mb,
            
            # Computational complexity
            'estimated_flops': flops,
            
            # Performance
            'avg_inference_time_ms': throughput_stats['avg_inference_time_ms'],
            'std_inference_time_ms': throughput_stats['std_inference_time_ms'],
            'throughput_samples_per_sec': throughput_stats['throughput_samples_per_sec'],
            
            # Memory
            'parameter_memory_mb': memory_stats['parameter_memory_mb'],
            'estimated_total_memory_mb': memory_stats['total_estimated_memory_mb']
        }
        
        return summary
        
    except Exception as e:
        return {
            'model_name': model_name,
            'error': str(e)
        }


def compare_models(model_names: Optional[List[str]] = None, 
                  n_classes: int = 4, n_channels: int = 22, 
                  n_samples: int = 500, device: str = 'cpu') -> Dict:
    """
    Compare multiple models
    
    Args:
        model_names (List[str], optional): List of model names to compare
        n_classes (int): Number of classes
        n_channels (int): Number of channels  
        n_samples (int): Number of samples
        device (str): Device to run comparison on
        
    Returns:
        Dict: Comparison results for all models
    """
    if model_names is None:
        model_names = list_available_models()
    
    comparison = {}
    
    for model_name in model_names:
        try:
            summary = get_model_summary(model_name, n_classes, n_channels, n_samples, device)
            comparison[model_name] = summary
        except Exception as e:
            comparison[model_name] = {
                'model_name': model_name,
                'error': str(e)
            }
    
    return comparison


def get_recommended_hyperparameters(model_name: str) -> Dict:
    """
    Get recommended hyperparameters for a model
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Dict: Recommended hyperparameters
    """
    if model_name.lower() not in DEFAULT_HYPERPARAMETERS:
        # Return default hyperparameters
        return {
            'learning_rate': 0.001,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'dropout_rate': 0.25,
            'optimizer': 'AdamW',
            'scheduler': 'OneCycleLR',
            'epochs': 200
        }
    
    return DEFAULT_HYPERPARAMETERS[model_name.lower()].copy()


def print_model_comparison_table(comparison: Dict, sort_by: str = 'total_parameters'):
    """
    Print a formatted comparison table
    
    Args:
        comparison (Dict): Model comparison results
        sort_by (str): Column to sort by
    """
    # Filter successful models
    successful_models = {k: v for k, v in comparison.items() if 'error' not in v}
    
    if not successful_models:
        print("No successful model comparisons to display")
        return
    
    # Sort models
    if sort_by in successful_models[list(successful_models.keys())[0]]:
        sorted_models = sorted(successful_models.items(), 
                             key=lambda x: x[1].get(sort_by, 0), reverse=True)
    else:
        sorted_models = list(successful_models.items())
    
    # Print header
    print(f"\n{'Model':<15} {'Params':<12} {'Size(MB)':<10} {'FLOPs':<15} {'Time(ms)':<12} {'Throughput':<15}")
    print("-" * 90)
    
    # Print each model
    for model_name, stats in sorted_models:
        params = f"{stats.get('total_parameters', 0):,}"
        size_mb = f"{stats.get('model_size_mb', 0):.2f}"
        flops = f"{stats.get('estimated_flops', 0):,}"
        time_ms = f"{stats.get('avg_inference_time_ms', 0):.2f}"
        throughput = f"{stats.get('throughput_samples_per_sec', 0):.1f}"
        
        print(f"{model_name:<15} {params:<12} {size_mb:<10} {flops:<15} {time_ms:<12} {throughput:<15}")
    
    # Print failed models
    failed_models = {k: v for k, v in comparison.items() if 'error' in v}
    if failed_models:
        print(f"\nFailed Models:")
        for model_name, info in failed_models.items():
            print(f"  {model_name}: {info['error']}")


# Aliases for compatibility
MODEL_REGISTRY = MODELS
MODEL_HYPERPARAMETERS = DEFAULT_HYPERPARAMETERS
get_model = create_model  # Alias for backwards compatibility

# Export main functions
__all__ = [
    'list_available_models',
    'create_model',
    'get_model', 
    'count_parameters',
    'calculate_flops',
    'benchmark_throughput',
    'get_model_memory_usage',
    'get_model_summary',
    'compare_models',
    'get_recommended_hyperparameters',
    'print_model_comparison_table',
    'MODELS',
    'DEFAULT_HYPERPARAMETERS',
    'MODEL_REGISTRY',
    'MODEL_HYPERPARAMETERS'
]
