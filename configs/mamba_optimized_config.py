"""
Optimized Configuration for Mamba Model
Based on analysis of previous training runs
"""

def get_mamba_optimized_config():
    """
    Optimized hyperparameters for Mamba on BCI-IV-2a dataset
    
    Key Changes from anti_overfitting:
    - Higher learning rate (0.001 vs 0.0002)
    - Lower weight decay (0.001 vs 0.02)
    - Lower dropout (0.1 vs 0.4) - Mamba already has internal regularization
    - Longer patience (20 vs 12) - Allow more time to improve
    - Removed heavy augmentation - Focus on learning patterns first
    """
    
    config = {
        # Training parameters
        'epochs': 150,
        'batch_size': 32,
        'device': 'auto',
        'num_workers': 4,
        'pin_memory': True,
        'verbose': True,
        'save_freq': 25,
        'plot_results': True,
        'log_level': 'INFO',
        'output_dir': './results_mamba_optimized',
        'save_model': True,
        'save_predictions': True,
        
        # Optimizer - Higher LR for Mamba
        'optimizer': {
            'name': 'AdamW',
            'learning_rate': 0.001,  # Increased from 0.0002
            'weight_decay': 0.001,    # Decreased from 0.02
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        
        # Scheduler - Gentler reduction
        'scheduler': {
            'name': 'ReduceLROnPlateau',
            'factor': 0.5,
            'patience': 10,          # Wait longer before reducing LR
            'min_lr': 1e-6,
            'mode': 'max'
        },
        
        # Regularization - Lighter for Mamba
        'regularization': {
            'dropout': 0.1,           # Mamba has built-in regularization
            'label_smoothing': 0.05,  # Reduced
            'gradient_clipping': 1.0, # Less aggressive
            'mixup_alpha': 0.0,       # Disabled for now
            'cutmix_alpha': 0.0       # Disabled for now
        },
        
        # Early stopping - More patient
        'early_stopping': {
            'patience': 20,           # Increased from 12
            'min_delta': 0.003,       # More lenient
            'monitor': 'val_acc',
            'mode': 'max',
            'restore_best_weights': True
        },
        
        # Data configuration
        'data': {
            'batch_size': 32,
            'validation_split': 0.2,
            'test_split': 0.2,
            'random_seed': 42,
            'stratify': True,
            'augmentation': False,    # Simplified - focus on core learning
            'noise_factor': 0.05      # Minimal noise
        },
        
        # Model-specific (Mamba)
        'model': {
            'dropout_rate': 0.1,
            'd_model': 64
        }
    }
    
    return config


def get_mongoose_optimized_config():
    """
    Optimized hyperparameters for Mongoose on BCI-IV-2a dataset
    """
    
    config = {
        # Training parameters
        'epochs': 150,
        'batch_size': 16,          # Smaller batch for larger model
        'device': 'auto',
        'num_workers': 4,
        'pin_memory': True,
        'verbose': True,
        'save_freq': 25,
        'plot_results': True,
        'log_level': 'INFO',
        'output_dir': './results_mongoose_optimized',
        'save_model': True,
        'save_predictions': True,
        
        # Optimizer
        'optimizer': {
            'name': 'AdamW',
            'learning_rate': 0.0005,  # Medium LR
            'weight_decay': 0.005,     # Moderate regularization
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        
        # Scheduler
        'scheduler': {
            'name': 'OneCycleLR',
            'max_lr': 0.001,
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
            'div_factor': 25.0,
            'final_div_factor': 10000.0
        },
        
        # Regularization - Moderate for Mongoose
        'regularization': {
            'dropout': 0.5,           # Keep high - Mongoose benefits from it
            'label_smoothing': 0.1,
            'gradient_clipping': 1.0,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 0.0
        },
        
        # Early stopping
        'early_stopping': {
            'patience': 25,           # Very patient for larger model
            'min_delta': 0.003,
            'monitor': 'val_acc',
            'mode': 'max',
            'restore_best_weights': True
        },
        
        # Data configuration
        'data': {
            'batch_size': 16,
            'validation_split': 0.2,
            'test_split': 0.2,
            'random_seed': 42,
            'stratify': True,
            'augmentation': True,
            'noise_factor': 0.08
        },
        
        # Model-specific (Mongoose)
        'model': {
            'dropout_rate': 0.5,
            'd_model': 64,
            'n_blocks': 3
        }
    }
    
    return config


if __name__ == "__main__":
    import json
    
    print("Mamba Optimized Configuration:")
    print(json.dumps(get_mamba_optimized_config(), indent=2))
    
    print("\n" + "="*50 + "\n")
    
    print("Mongoose Optimized Configuration:")
    print(json.dumps(get_mongoose_optimized_config(), indent=2))
