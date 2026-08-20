import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def plot_training_losses(checkpoint_dir):
    """Plot training losses for all models in the checkpoint directory."""
    # Find all loss files
    loss_dir = os.path.join(checkpoint_dir, "losses")
    loss_files = glob(os.path.join(loss_dir, "*_losses.npy"))
    
    if not loss_files:
        print("No loss files found in", loss_dir)
        return
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot each model's losses
    for loss_file in loss_files:
        # Get model name from filename
        model_name = os.path.basename(loss_file).replace("_losses.npy", "")
        
        # Load losses
        losses = np.load(loss_file)
        epochs = np.arange(1, len(losses) + 1)
        
        # Plot losses
        plt.plot(epochs, losses, label=model_name, alpha=0.7)
        
        # Print final loss
        print(f"{model_name} final loss: {losses[-1]:.4f}")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses Across Different Models")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(checkpoint_dir, "training_losses.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved training loss plot to {plot_path}")

if __name__ == "__main__":
    from src.model import config
    plot_training_losses(config.CONFIG["checkpoint_dir"])