import os
import json
import torch
from datetime import datetime
from itertools import product

from src.model import config
from src.model.config import PROJECT_ROOT
from src.model.trainer import train
from src.model.predictandsave import predict_and_save_inline

# Define results directory at project root
HYPERPARAMS_DIR = os.path.join(PROJECT_ROOT, "HyperParams")

def get_experiment_name(cfg_weight, samples_per_epoch, num_epochs, diff_steps):
    """Create a descriptive name for the experiment files."""
    return f"cfg{cfg_weight}_samp{samples_per_epoch}_ep{num_epochs}_steps{diff_steps}"

def run_experiment(
    cfg_weight: float,
    samples_per_epoch: int,
    num_epochs: int,
    diff_steps: int,
    base_dir: str,
    device: torch.device
):
    """Run a single experiment with given hyperparameters."""
    
    # Create descriptive name and directory for this experiment
    exp_name = get_experiment_name(cfg_weight, samples_per_epoch, num_epochs, diff_steps)
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create prediction output directory inside experiment directory
    pred_dir = os.path.join(exp_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    # Backup original config
    original_config = config.CONFIG.copy()
    
    try:
        # Update CONFIG with experiment settings
        config.CONFIG.update({
            "classifier_free_guidance_weight": cfg_weight,
            "samples_per_epoch": samples_per_epoch,
            "n_epochs": num_epochs,
            "num_diffusion_timesteps": diff_steps,
            "checkpoint_dir": exp_dir,
            "prediction_output_dir": pred_dir,
            "save_name": "model",  # Simple name since each experiment has its own directory
            # Add more reasonable SDE parameters
            "sde_bmin": 0.1,  # Keep minimum noise the same
            "sde_bmax": 2.0,  # Reduce maximum noise significantly
        })
        
        # Debug prints to verify config
        print("\nVerifying experiment configuration:")
        print(f"CFG weight: {config.CONFIG['classifier_free_guidance_weight']}")
        print(f"Samples per epoch: {config.CONFIG['samples_per_epoch']}")
        print(f"Number of epochs: {config.CONFIG['n_epochs']}")
        print(f"Diffusion steps: {config.CONFIG['num_diffusion_timesteps']}")
        print(f"SDE noise range: [{config.CONFIG['sde_bmin']}, {config.CONFIG['sde_bmax']}]")
        print(f"Checkpoint dir: {config.CONFIG['checkpoint_dir']}")
        
        # Save experiment config
        config_path = os.path.join(exp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.CONFIG, f, indent=2)
        
        # Train model - this will save checkpoints in exp_dir
        print(f"\nStarting training for experiment: {exp_name}")
        train()  # Uses global CONFIG
        
        # Find the final checkpoint
        final_checkpoint = os.path.join(exp_dir, "model_ep{}.pth".format(num_epochs))
        
        # Generate predictions
        print("\nGenerating predictions for experiment")
        predict_and_save_inline(
            checkpoint_path=final_checkpoint,
            history_len=config.CONFIG["history_len"],
            predict_len=config.CONFIG["predict_len"],
            input_dim=config.CONFIG.get("input_dim", 1),
            num_diffusion_timesteps=diff_steps,
        )
        
        print("\nExperiment completed successfully!")
        return True, exp_name, exp_dir
        
    except Exception as e:
        print(f"\nError in experiment: {str(e)}")
        # Save error log in experiment directory
        with open(os.path.join(exp_dir, "error.log"), 'w') as f:
            f.write(f"Error occurred: {str(e)}")
        return False, exp_name, exp_dir
    
    finally:
        # Restore original config
        config.CONFIG.clear()
        config.CONFIG.update(original_config)

def main():
    # Define hyperparameter ranges to test
    # Focus on parameters that affect the model's ability to predict normalized values
    cfg_weights = [.25, 0.5, 1.0, 3.0, 8]  # Reduced range, focus on middle values
    samples_per_epochs = [500]  # More samples per epoch for better statistics
    num_epochs_list = [600]  # Longer training to ensure convergence
    diff_steps_list = [500]  # Standard number of steps
    
    # Create base results directory
    os.makedirs(HYPERPARAMS_DIR, exist_ok=True)
    print(f"\nResults will be saved in: {HYPERPARAMS_DIR}")
    
    # Save sweep configuration at root level
    sweep_config = {
        "cfg_weights": cfg_weights,
        "samples_per_epochs": samples_per_epochs,
        "num_epochs_list": num_epochs_list,
        "diff_steps_list": diff_steps_list,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results_path": HYPERPARAMS_DIR
    }
    with open(os.path.join(HYPERPARAMS_DIR, "sweep_config.json"), 'w') as f:
        json.dump(sweep_config, f, indent=2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Run all combinations
    total_experiments = len(list(product(cfg_weights, samples_per_epochs, num_epochs_list, diff_steps_list)))
    completed = 0
    successful = 0
    experiment_results = []
    
    print(f"\nStarting hyperparameter sweep with {total_experiments} combinations")
    print("Note: Focusing on CFG weight variations with increased samples/epoch")
    
    for cfg_w, samp, epochs, steps in product(cfg_weights, samples_per_epochs, num_epochs_list, diff_steps_list):
        completed += 1
        print(f"\nExperiment {completed}/{total_experiments}")
        print(f"Parameters: CFG={cfg_w}, Samples={samp}, Epochs={epochs}, Steps={steps}")
        
        success, exp_name, exp_dir = run_experiment(
            cfg_weight=cfg_w,
            samples_per_epoch=samp,
            num_epochs=epochs,
            diff_steps=steps,
            base_dir=HYPERPARAMS_DIR,
            device=device
        )
        
        experiment_results.append({
            "name": exp_name,
            "directory": exp_dir,
            "success": success,
            "parameters": {
                "cfg_weight": cfg_w,
                "samples_per_epoch": samp,
                "num_epochs": epochs,
                "diff_steps": steps
            }
        })
        
        if success:
            successful += 1
    
    # Save summary at root level
    summary = {
        "total_experiments": total_experiments,
        "successful_experiments": successful,
        "failed_experiments": total_experiments - successful,
        "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results_path": HYPERPARAMS_DIR,
        "experiment_results": experiment_results
    }
    with open(os.path.join(HYPERPARAMS_DIR, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nHyperparameter sweep completed!")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_experiments - successful}")
    print(f"Results saved in: {HYPERPARAMS_DIR}")

if __name__ == "__main__":
    main() 
