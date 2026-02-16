"""
Minimal Boilerplate Trainer Example

This is the simplest possible trainer to demonstrate Optuna integration.
Replace this with your actual training logic.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any


def simple_trainer(
    config: Dict[str, Any],
    save_dir: Path,
    data_dir: Path,
    experiment_tag: str = "sweep",
    cluster: bool = False,
    resume_ckpt: str = None,
    debug: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Minimal training function for demonstration.
    
    This function simulates a training process with synthetic data.
    Replace this with your actual training pipeline.
    
    Args:
        config: Configuration dictionary with model and training parameters
        save_dir: Directory to save outputs
        data_dir: Directory containing data
        experiment_tag: Tag for experiment tracking
        cluster: Whether running on cluster
        resume_ckpt: Path to checkpoint to resume from
        debug: Debug mode flag
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing training metrics
    """
    print(f"[Trainer] Starting training with config:")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Num layers: {config.model.num_layers}")
    print(f"  Learning rate: {config.training.lr}")
    print(f"  Dropout: {config.training.dropout}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Save dir: {save_dir}")
    
    # Simulate training process
    num_epochs = config.training.num_epochs
    val_losses = []
    train_losses = []
    
    # Simulate epoch-by-epoch training
    for epoch in range(num_epochs):
        # Simulate training loss (decreasing with some noise)
        train_loss = 1.0 - (epoch / num_epochs) * 0.5 + np.random.normal(0, 0.1)
        train_loss = max(0.1, train_loss)  # Keep positive
        
        # Simulate validation loss (related to hyperparameters)
        # Better hyperparameters = lower loss
        base_val_loss = 0.5
        
        # Penalize extreme hidden dims
        hidden_penalty = abs(config.model.hidden_dim - 128) / 1000
        
        # Penalize too many or too few layers
        layer_penalty = abs(config.model.num_layers - 4) / 20
        
        # Penalize extreme learning rates
        lr_penalty = abs(np.log10(config.training.lr) + 3) / 10
        
        # Penalize extreme dropout
        dropout_penalty = abs(config.training.dropout - 0.15) / 5
        
        val_loss = (base_val_loss + hidden_penalty + layer_penalty + 
                   lr_penalty + dropout_penalty + np.random.normal(0, 0.05))
        val_loss = max(0.1, val_loss)  # Keep positive
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Save some dummy outputs
    save_dir.mkdir(parents=True, exist_ok=True)
    results_file = save_dir / "results.txt"
    with open(results_file, 'w') as f:
        f.write(f"Training completed\n")
        f.write(f"Final val loss: {val_losses[-1]:.4f}\n")
        f.write(f"Final train loss: {train_losses[-1]:.4f}\n")
    
    print(f"[Trainer] Training complete! Results saved to {save_dir}")
    
    # Return metrics
    return {
        "val_loss": np.array(val_losses),
        "train_loss": np.array(train_losses),
        "final_val_loss": val_losses[-1],
        "final_train_loss": train_losses[-1],
    }
