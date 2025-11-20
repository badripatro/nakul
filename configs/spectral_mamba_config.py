"""
Configuration for SpectralMambaMemory Model Training
====================================================

This module contains configuration settings for training the SpectralMambaMemory
model on BCI Competition IV Dataset 2a.
"""

import torch


# ============================================================================
# Model Configurations
# ============================================================================

SPECTRAL_MAMBA_CONFIG = {
    'standard': {
        'model_type': 'standard',
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,
        'hidden_dim': 128,
        'mem_dim': 64,
        'n_mamba_blocks': 3,
        'use_archetype': False,
        'archetype_dim': 32,
        'dropout_rate': 0.3,
    },
    'lite': {
        'model_type': 'lite',
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,
        'hidden_dim': 64,
        'mem_dim': 32,
        'n_mamba_blocks': 2,
        'dropout_rate': 0.2,
    },
    'deep': {
        'model_type': 'standard',
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,
        'hidden_dim': 256,
        'mem_dim': 128,
        'n_mamba_blocks': 4,
        'use_archetype': False,
        'dropout_rate': 0.4,
    },
    'with_archetype': {
        'model_type': 'standard',
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,
        'hidden_dim': 128,
        'mem_dim': 64,
        'n_mamba_blocks': 3,
        'use_archetype': True,
        'archetype_dim': 32,
        'dropout_rate': 0.3,
    },
    'optimized_85': {
        'model_type': 'fixed',
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,
        'F1': 12,
        'D': 2,
        'F2': 24,
        'dropout_rate': 0.3,
        'sampling_rate': 250
    },
    'optimized_90': {
        'model_type': 'fixed',
        'n_classes': 4,
        'n_channels': 22,
        'n_samples': 1000,
        'F1': 16,
        'D': 2,
        'F2': 32,
        'dropout_rate': 0.3,
        'sampling_rate': 250
    }
}


# ============================================================================
# Training Configurations
# ============================================================================

TRAINING_CONFIG = {
    'standard': {
        'epochs': 300,
        'batch_size': 32,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 10,
        'patience': 50,
        'min_delta': 0.001,
    },
    'lite': {
        'epochs': 200,
        'batch_size': 64,
        'learning_rate': 0.002,
        'weight_decay': 0.005,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'patience': 30,
        'min_delta': 0.001,
    },
    'deep': {
        'epochs': 400,
        'batch_size': 16,
        'learning_rate': 0.0005,
        'weight_decay': 0.02,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'warmup_epochs': 20,
        'patience': 60,
        'min_delta': 0.001,
    },
    'fast': {
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.003,
        'weight_decay': 0.01,
        'optimizer': 'adam',
        'scheduler': 'step',
        'warmup_epochs': 5,
        'patience': 20,
        'min_delta': 0.001,
    },
    'optimized_85': {
        'description': 'OPTIMIZED for 85% test accuracy - Balanced config',
        'epochs': 200,
        'batch_size': 48,
        'learning_rate': 0.0015,
        'weight_decay': 0.015,
        'optimizer': 'adamw',
        'scheduler': 'onecycle',
        'warmup_epochs': 0,
        'patience': 30,
        'min_delta': 0.0005,
    },
    'optimized_90': {
        'description': 'OPTIMIZED for 90% test accuracy - FIXED model',
        'epochs': 150,
        'batch_size': 64,         # Larger batch for stability
        'learning_rate': 0.001,   # Higher LR for faster convergence
        'weight_decay': 0.01,
        'dropout_rate': 0.3,
        'optimizer': 'adamw',      # normalized lowercase for trainer
        'scheduler': 'onecycle',   # match trainer expectation
        'scheduler_params': {
            'max_lr': 0.001,
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
            'div_factor': 25.0,
            'final_div_factor': 10000.0
        },
        'early_stopping_patience': 20,
        'model_params': {
            'F1': 16,              # EEGNet-style filters
            'D': 2,
            'F2': 32,
            'dropout_rate': 0.3
        }
    },
    'debug_fixed': {
        'description': 'DEBUG quick run for fixed spectral model',
        'epochs': 15,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'optimizer': 'adamw',
        'scheduler': 'onecycle',
        'max_lr': 0.001,
        'pct_start': 0.3,
        'patience': 200,          # effectively disable early stopping
        'min_delta': 0.0005,
        'warmup_epochs': 0,
    },
}


# ============================================================================
# Data Augmentation Configurations
# ============================================================================

AUGMENTATION_CONFIG = {
    'none': {
        'use_augmentation': False,
    },
    'light': {
        'use_augmentation': True,
        'gaussian_noise_std': 0.01,
        'time_shift_max': 10,
        'amplitude_scale_range': (0.95, 1.05),
    },
    'medium': {
        'use_augmentation': True,
        'gaussian_noise_std': 0.02,
        'time_shift_max': 20,
        'amplitude_scale_range': (0.9, 1.1),
        'time_mask_max': 50,
    },
    'heavy': {
        'use_augmentation': True,
        'gaussian_noise_std': 0.03,
        'time_shift_max': 30,
        'amplitude_scale_range': (0.85, 1.15),
        'time_mask_max': 100,
        'channel_dropout': 0.1,
    }
}


# ============================================================================
# Evaluation Configurations
# ============================================================================

EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'kappa'],
    'save_predictions': True,
    'save_memory_states': True,
    'visualize_attention': False,
    'compute_confusion_matrix': True,
}


# ============================================================================
# Preset Configurations (combines model + training + augmentation)
# ============================================================================

PRESET_CONFIGS = {
    'baseline': {
        'model': 'standard',
        'training': 'standard',
        'augmentation': 'none',
        'description': 'Baseline configuration without augmentation'
    },
    'lite_fast': {
        'model': 'lite',
        'training': 'fast',
        'augmentation': 'light',
        'description': 'Fast training with lightweight model'
    },
    'high_accuracy': {
        'model': 'optimized_85',
        'training': 'optimized_85',
        'augmentation': 'light',
        'description': 'FIXED model optimized for 85% test accuracy'
    },
    'ultra_accuracy': {
        'model': 'optimized_90',
        'training': 'optimized_90',
        'augmentation': 'light',
        'description': 'FIXED model optimized for 90% test accuracy'
    },
    'robust': {
        'model': 'standard',
        'training': 'standard',
        'augmentation': 'heavy',
        'description': 'Standard model with heavy augmentation for robustness'
    },
    'archetype_guided': {
        'model': 'with_archetype',
        'training': 'standard',
        'augmentation': 'medium',
        'description': 'Model with archetype guidance mechanism'
    },
    'debug_fixed': {
        'model': 'optimized_90',
        'training': 'debug_fixed',
        'augmentation': 'none',
        'description': 'Debug preset (no augmentation) to test generalization quickly'
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_config(preset_name='baseline'):
    """
    Get complete configuration for a preset.
    
    Args:
        preset_name: Name of the preset configuration
        
    Returns:
        dict: Complete configuration including model, training, and augmentation
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                        f"Available: {list(PRESET_CONFIGS.keys())}")
    
    preset = PRESET_CONFIGS[preset_name]
    
    config = {
        'preset_name': preset_name,
        'description': preset['description'],
        'model': SPECTRAL_MAMBA_CONFIG[preset['model']],
        'training': TRAINING_CONFIG[preset['training']],
        'augmentation': AUGMENTATION_CONFIG[preset['augmentation']],
        'evaluation': EVAL_CONFIG.copy()
    }
    
    return config


def get_model_config(model_name='standard'):
    """Get model configuration by name."""
    if model_name not in SPECTRAL_MAMBA_CONFIG:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(SPECTRAL_MAMBA_CONFIG.keys())}")
    return SPECTRAL_MAMBA_CONFIG[model_name]


def get_training_config(training_name='standard'):
    """Get training configuration by name."""
    if training_name not in TRAINING_CONFIG:
        raise ValueError(f"Unknown training config: {training_name}. "
                        f"Available: {list(TRAINING_CONFIG.keys())}")
    return TRAINING_CONFIG[training_name]


def print_config(config):
    """Pretty print configuration."""
    print("=" * 80)
    print(f"Configuration: {config.get('preset_name', 'Custom')}")
    print("=" * 80)
    
    if 'description' in config:
        print(f"\nDescription: {config['description']}")
    
    print("\n--- Model Configuration ---")
    for key, value in config['model'].items():
        print(f"  {key}: {value}")
    
    print("\n--- Training Configuration ---")
    for key, value in config['training'].items():
        print(f"  {key}: {value}")
    
    print("\n--- Augmentation Configuration ---")
    for key, value in config['augmentation'].items():
        print(f"  {key}: {value}")
    
    print("=" * 80)


if __name__ == "__main__":
    print("Available Preset Configurations:")
    print("-" * 80)
    
    for preset_name, preset_info in PRESET_CONFIGS.items():
        print(f"\n{preset_name}:")
        print(f"  Description: {preset_info['description']}")
        print(f"  Model: {preset_info['model']}")
        print(f"  Training: {preset_info['training']}")
        print(f"  Augmentation: {preset_info['augmentation']}")
    
    print("\n" + "=" * 80)
    print("Example: Getting baseline configuration")
    print("=" * 80)
    
    config = get_config('baseline')
    print_config(config)