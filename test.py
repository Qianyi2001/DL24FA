import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import create_wall_dataloader, WallDataset
from models import JEPAModel
from evaluator import ProbingEvaluator
from schedulers import Scheduler, LRSchedule
from torch.optim import AdamW
import os
from torch.cuda.amp import GradScaler, autocast

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device, batch_size=64):
    """Load training and validation datasets."""
    # Main Training Dataset
    train_data_path = "/scratch/DL24FA/train"
    train_dataset = WallDataset(
        data_path=train_data_path,
        probing=False,
        device=device
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Probing Datasets
    probe_train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/train",
        probing=True,
        device=device,
        batch_size=batch_size,
        train=True
    )

    val_normal_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/val",
        probing=True,
        device=device,
        batch_size=batch_size,
        train=False
    )

    val_wall_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_wall/val",
        probing=True,
        device=device,
        batch_size=batch_size,
        train=False
    )

    return train_loader, probe_train_loader, {"normal": val_normal_loader, "wall": val_wall_loader}
def main():
    # Set device
    device = get_device()
    checkpoint_path = "checkpoints/epoch_0_batch100_jepa.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = JEPAModel(latent_dim=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_loader, probe_train_loader, val_loaders = load_data(device)

    # Evaluate
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_loader,
        probe_val_ds=val_loaders,
    )
    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober)

    for key, loss in avg_losses.items():
        print(f"{key} validation loss: {loss:.4f}")


if __name__ == "__main__":
    main()