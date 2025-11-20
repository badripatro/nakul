"""
Enhanced Trainer with Advanced Regularization Techniques
Addresses overfitting with dropout, weight decay, and proper monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, StepLR, OneCycleLR
)
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MixupCutmixLoss:
    """Mixup and Cutmix augmentation with loss calculation"""
    
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 0.2, prob: float = 0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        """Apply cutmix augmentation for EEG data"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # For EEG data, apply cutmix along time dimension
        _, channels, time_points = x.shape
        cut_ratio = np.sqrt(1. - lam)
        cut_w = int(time_points * cut_ratio)
        
        # Uniformly sample the cut region
        cx = np.random.randint(time_points)
        bbx1 = np.clip(cx - cut_w // 2, 0, time_points)
        bbx2 = np.clip(cx + cut_w // 2, 0, time_points)
        
        x[:, :, bbx1:bbx2] = x[index, :, bbx1:bbx2]
        
        # Adjust lambda to match the actual ratio
        lam = 1 - ((bbx2 - bbx1) / time_points)
        
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        """Apply random mixup or cutmix"""
        if np.random.random() < self.prob:
            if np.random.random() < 0.5 and self.mixup_alpha > 0:
                return self.mixup_data(x, y, self.mixup_alpha)
            elif self.cutmix_alpha > 0:
                return self.cutmix_data(x, y, self.cutmix_alpha)
        
        return x, y, None, 1.0

class EnhancedLossFunction(nn.Module):
    """Enhanced loss function with label smoothing and focal loss"""
    
    def __init__(self, 
                 num_classes: int,
                 label_smoothing: float = 0.1,
                 focal_alpha: Optional[torch.Tensor] = None,
                 focal_gamma: float = 2.0,
                 use_focal: bool = False):
        super().__init__()
        
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal
        
        if self.use_focal:
            self.focal_loss = self._focal_loss
        else:
            self.focal_loss = self._cross_entropy_loss
    
    def _cross_entropy_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Standard cross-entropy with label smoothing"""
        if self.label_smoothing > 0:
            # Label smoothing
            log_probs = F.log_softmax(inputs, dim=-1)
            smooth_targets = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1 - self.label_smoothing
            )
            smooth_targets += self.label_smoothing / self.num_classes
            return (-smooth_targets * log_probs).sum(dim=-1).mean()
        else:
            return F.cross_entropy(inputs, targets)
    
    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss implementation"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.focal_gamma * ce_loss
        
        if self.focal_alpha is not None:
            alpha_t = self.focal_alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                targets_b: Optional[torch.Tensor] = None, lam: float = 1.0) -> torch.Tensor:
        """Forward pass with support for mixup/cutmix"""
        
        if targets_b is not None:
            # Mixup/Cutmix loss
            loss_a = self.focal_loss(inputs, targets)
            loss_b = self.focal_loss(inputs, targets_b)
            return lam * loss_a + (1 - lam) * loss_b
        else:
            return self.focal_loss(inputs, targets)

class EarlyStoppingMonitor:
    """Enhanced early stopping with multiple criteria"""
    
    def __init__(self, 
                 patience: int = 15,
                 min_delta: float = 0.001,
                 mode: str = 'max',
                 restore_best_weights: bool = True,
                 monitor_metrics: List[str] = ['val_acc', 'val_loss']):
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.monitor_metrics = monitor_metrics
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        self.is_better = self._init_is_better_func()
    
    def _init_is_better_func(self):
        if self.mode == 'max':
            return lambda current, best: current > best + self.min_delta
        else:
            return lambda current, best: current < best - self.min_delta
    
    def __call__(self, metrics: Dict[str, float], model: nn.Module) -> bool:
        """Check if early stopping criteria are met"""
        
        # Use primary metric for early stopping decision
        primary_metric = self.monitor_metrics[0]
        current_score = metrics.get(primary_metric, None)
        
        if current_score is None:
            logging.warning(f"Metric {primary_metric} not found in metrics")
            return False
        
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logging.info("Restored best model weights")
            
        return self.early_stop

class MetricsTracker:
    """Enhanced metrics tracking with detailed monitoring"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rate': [], 'gradient_norm': [], 
            'epoch_time': []
        }
        self.best_metrics = {
            'best_val_acc': 0.0,
            'best_val_f1': 0.0,
            'best_epoch': 0
        }
    
    def update(self, metrics: Dict[str, float], epoch: int):
        """Update metrics history"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        # Update best metrics
        if 'val_acc' in metrics:
            if metrics['val_acc'] > self.best_metrics['best_val_acc']:
                self.best_metrics['best_val_acc'] = metrics['val_acc']
                self.best_metrics['best_epoch'] = epoch
        
        if 'val_f1' in metrics:
            if metrics['val_f1'] > self.best_metrics['best_val_f1']:
                self.best_metrics['best_val_f1'] = metrics['val_f1']
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get latest metrics"""
        current = {}
        for key, values in self.history.items():
            if values:
                current[key] = values[-1]
        return current
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc', color='blue')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.history['learning_rate'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Gradient norm plot
        axes[1, 1].plot(self.history['gradient_norm'], color='orange')
        axes[1, 1].set_title('Gradient Norm')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training history plot saved to {save_path}")
        
        return fig

class EnhancedTrainer:
    """Enhanced trainer with advanced regularization and monitoring"""
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 device: torch.device):
        
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        self._setup_regularization()
        self._setup_monitoring()
        
        logging.info("Enhanced trainer initialized")
        logging.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logging.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def _setup_optimizer(self):
        """Setup optimizer with proper weight decay"""
        optimizer_config = self.config.optimizer
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to bias and batch norm parameters
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': optimizer_config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        if optimizer_config.name.lower() == 'adamw':
            self.optimizer = AdamW(
                param_groups,
                lr=optimizer_config.learning_rate,
                betas=optimizer_config.betas,
                eps=optimizer_config.eps
            )
        elif optimizer_config.name.lower() == 'sgd':
            self.optimizer = SGD(
                param_groups,
                lr=optimizer_config.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_config.name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_config = self.config.scheduler
        
        if scheduler_config.name == 'CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=getattr(scheduler_config, 'T_max', self.config.epochs),
                eta_min=getattr(scheduler_config, 'eta_min', 1e-6)
            )
        elif scheduler_config.name == 'CosineAnnealingWarmRestarts':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=getattr(scheduler_config, 'T_0', 20),
                T_mult=getattr(scheduler_config, 'T_mult', 2),
                eta_min=getattr(scheduler_config, 'eta_min', 1e-6)
            )
        elif scheduler_config.name == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=getattr(scheduler_config, 'factor', 0.5),
                patience=getattr(scheduler_config, 'patience', 8),
                min_lr=getattr(scheduler_config, 'min_lr', 1e-7)
            )
        elif scheduler_config.name == 'StepLR':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=getattr(scheduler_config, 'step_size', 10),
                gamma=getattr(scheduler_config, 'gamma', 0.7)
            )
        else:
            self.scheduler = None
    
    def _setup_loss_function(self):
        """Setup enhanced loss function"""
        reg_config = self.config.regularization
        
        self.criterion = EnhancedLossFunction(
            num_classes=4,  # BCI IV-2a has 4 classes
            label_smoothing=reg_config.label_smoothing,
            use_focal=False  # Can be enabled if class imbalance is severe
        )
    
    def _setup_regularization(self):
        """Setup regularization techniques"""
        reg_config = self.config.regularization
        
        # Mixup/Cutmix augmentation
        self.mixup_cutmix = MixupCutmixLoss(
            mixup_alpha=reg_config.mixup_alpha,
            cutmix_alpha=reg_config.cutmix_alpha,
            prob=0.5
        )
        
        # Gradient clipping value
        self.gradient_clipping = reg_config.gradient_clipping
    
    def _setup_monitoring(self):
        """Setup monitoring components"""
        # Metrics tracker
        self.metrics = MetricsTracker()
        
        # Early stopping
        early_config = self.config.early_stopping
        self.early_stopping = EarlyStoppingMonitor(
            patience=early_config.patience,
            min_delta=early_config.min_delta,
            mode=early_config.mode,
            restore_best_weights=early_config.restore_best_weights,
            monitor_metrics=[early_config.monitor]
        )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch with enhanced regularization"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        gradient_norms = []
        
        for batch_idx, (data, targets, _) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Apply mixup/cutmix augmentation
            data, targets_a, targets_b, lam = self.mixup_cutmix(data, targets)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(data)
            loss = self.criterion(outputs, targets_a, targets_b, lam)
            
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clipping > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
                gradient_norms.append(grad_norm.item())
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions (use original targets for metrics)
            with torch.no_grad():
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        return {
            'train_loss': avg_loss,
            'train_acc': accuracy,
            'train_f1': f1,
            'gradient_norm': avg_grad_norm
        }
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets, _ in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return {
            'val_loss': avg_loss,
            'val_acc': accuracy, 
            'val_f1': f1,
            'predictions': all_preds,
            'targets': all_targets
        }
    
    def train(self, train_loader, val_loader) -> Dict[str, Any]:
        """Main training loop"""
        
        logging.info(f"Starting training for {self.config.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            epoch_metrics['epoch_time'] = time.time() - epoch_start
            
            # Update metrics tracker
            self.metrics.update(epoch_metrics, epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_acc'])
                else:
                    self.scheduler.step()
            
            # Logging
            if self.config.verbose:
                logging.info(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs}: "
                    f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    f"Train Acc: {train_metrics['train_acc']:.2%}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Val Acc: {val_metrics['val_acc']:.2%}, "
                    f"LR: {epoch_metrics['learning_rate']:.6f}"
                )
                
                # Warning for low validation accuracy
                if val_metrics['val_acc'] < 0.35:  # Below random + margin
                    logging.warning(f"Low validation accuracy ({val_metrics['val_acc']:.2%}) at epoch {epoch + 1}")
            
            # Early stopping check
            if self.early_stopping(epoch_metrics, self.model):
                logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f} seconds")
        
        return {
            'training_time': total_time,
            'epochs_trained': epoch + 1,
            'best_metrics': self.metrics.best_metrics,
            'final_metrics': self.metrics.get_current_metrics(),
            'history': self.metrics.history
        }