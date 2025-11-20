"""
Enhanced Training Configuration for BCI Models
Addresses overfitting and poor validation accuracy issues
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

@dataclass
class OptimizerConfig:
    """Optimizer configuration"""
    name: str = "AdamW"
    learning_rate: float = 0.0005  # Much lower than original 0.003
    weight_decay: float = 0.01     # Increased from 0.0001
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

@dataclass 
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    name: str = "CosineAnnealingWarmRestarts"  # Better than OneCycleLR for this case
    T_0: int = 20  # Initial restart period
    T_mult: int = 2  # Period multiplication factor
    eta_min: float = 1e-6  # Minimum learning rate
    warmup_epochs: int = 5  # Reduced warmup
    # Additional parameters for different schedulers
    factor: float = 0.5  # For ReduceLROnPlateau
    patience: int = 8  # For ReduceLROnPlateau  
    min_lr: float = 1e-7  # For ReduceLROnPlateau
    T_max: int = 60  # For CosineAnnealingLR
    step_size: int = 10  # For StepLR
    gamma: float = 0.7  # For StepLR

@dataclass
class RegularizationConfig:
    """Regularization configuration"""
    dropout: float = 0.3  # Increased dropout
    label_smoothing: float = 0.05  # Reduced from 0.1
    gradient_clipping: float = 0.5  # Reduced from 1.0
    mixup_alpha: float = 0.2  # Add mixup augmentation
    cutmix_alpha: float = 0.2  # Add cutmix augmentation
    
@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration"""
    patience: int = 15  # Reduced from 20
    min_delta: float = 0.001  # Minimum improvement
    monitor: str = "val_acc"  # Monitor validation accuracy
    mode: str = "max"  # Maximize validation accuracy
    restore_best_weights: bool = True

@dataclass
class DataConfig:
    """Data configuration"""
    batch_size: int = 16  # Reduced batch size
    validation_split: float = 0.2
    test_split: float = 0.2
    random_seed: int = 42
    stratify: bool = True  # Ensure balanced splits
    augmentation: bool = True
    noise_factor: float = 0.05  # Add gaussian noise
    
@dataclass
class ModelConfig:
    """Model-specific configuration"""
    # EEGNet specific
    eegnet_dropout: float = 0.25
    eegnet_f1: int = 8
    eegnet_f2: int = 16
    eegnet_d: int = 2
    
    # DeepConvNet specific  
    deepconvnet_dropout: float = 0.5
    deepconvnet_channels: List[int] = field(default_factory=lambda: [25, 50, 100, 200])
    
    # ShallowConvNet specific
    shallowconvnet_dropout: float = 0.5
    shallowconvnet_filters: int = 40

@dataclass
class TrainingConfig:
    """Main training configuration"""
    # Basic settings
    epochs: int = 100
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Monitoring
    verbose: bool = True
    save_freq: int = 25
    plot_results: bool = True
    log_level: str = "INFO"
    
    # Output
    output_dir: str = "./results_improved"
    save_model: bool = True
    save_predictions: bool = True
    
    # Configuration objects
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        # Extract nested configs
        optimizer = OptimizerConfig(**config_dict.pop('optimizer', {}))
        scheduler = SchedulerConfig(**config_dict.pop('scheduler', {}))
        regularization = RegularizationConfig(**config_dict.pop('regularization', {}))
        early_stopping = EarlyStoppingConfig(**config_dict.pop('early_stopping', {}))
        data = DataConfig(**config_dict.pop('data', {}))
        model = ModelConfig(**config_dict.pop('model', {}))
        
        return cls(
            optimizer=optimizer,
            scheduler=scheduler,
            regularization=regularization,
            early_stopping=early_stopping,
            data=data,
            model=model,
            **config_dict
        )

# Predefined configurations for different scenarios
ANTI_OVERFITTING_CONFIG = TrainingConfig(
    epochs=80,
    optimizer=OptimizerConfig(
        learning_rate=0.0002,  # Very low learning rate
        weight_decay=0.02      # High weight decay
    ),
    scheduler=SchedulerConfig(
        name="ReduceLROnPlateau",
        factor=0.5,
        patience=8,
        min_lr=1e-7
    ),
    regularization=RegularizationConfig(
        dropout=0.4,           # High dropout
        label_smoothing=0.1,
        gradient_clipping=0.3,
        mixup_alpha=0.3,
        cutmix_alpha=0.3
    ),
    early_stopping=EarlyStoppingConfig(
        patience=12,
        min_delta=0.005
    ),
    data=DataConfig(
        batch_size=8,          # Very small batch size
        augmentation=True,
        noise_factor=0.1
    )
)

BALANCED_CONFIG = TrainingConfig(
    epochs=60,
    optimizer=OptimizerConfig(
        learning_rate=0.0008,
        weight_decay=0.005
    ),
    scheduler=SchedulerConfig(
        name="CosineAnnealingLR",
        T_max=60,
        eta_min=1e-6
    ),
    regularization=RegularizationConfig(
        dropout=0.25,
        label_smoothing=0.05,
        gradient_clipping=0.5
    ),
    data=DataConfig(
        batch_size=32,
        augmentation=True,
        noise_factor=0.03
    )
)

FAST_CONFIG = TrainingConfig(
    epochs=30,
    optimizer=OptimizerConfig(
        learning_rate=0.001,
        weight_decay=0.001
    ),
    scheduler=SchedulerConfig(
        name="StepLR",
        step_size=10,
        gamma=0.7
    ),
    regularization=RegularizationConfig(
        dropout=0.2,
        label_smoothing=0.02,
        gradient_clipping=1.0
    ),
    data=DataConfig(
        batch_size=64,
        augmentation=False
    )
)

def get_config(config_name: str = "anti_overfitting") -> TrainingConfig:
    """Get predefined configuration by name"""
    configs = {
        "anti_overfitting": ANTI_OVERFITTING_CONFIG,
        "balanced": BALANCED_CONFIG, 
        "fast": FAST_CONFIG,
        "default": TrainingConfig()
    }
    
    if config_name not in configs:
        print(f"Warning: Config '{config_name}' not found. Using default.")
        config_name = "default"
        
    return configs[config_name]

def save_config(config: TrainingConfig, filepath: str):
    """Save configuration to file"""
    import json
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"Configuration saved to {filepath}")

def load_config(filepath: str) -> TrainingConfig:
    """Load configuration from file"""
    import json
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return TrainingConfig.from_dict(config_dict)