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

def check_for_collapse(embeddings: torch.Tensor, eps: float = 1e-8):
    if embeddings.dim() == 3:
        B, T, D = embeddings.shape
        flat_emb = embeddings.view(B * T, D)
    else:
        flat_emb = embeddings
    var = flat_emb.var(dim=0)
    mean_var = var.mean().item()
    print(f"Check collapse: avg var={mean_var:.6f}, min var={var.min().item():.6f}, max var={var.max().item():.6f}")
    if mean_var < eps:
        print("Warning: Potential collapse detected.")
    return mean_var

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

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

def train_model(model, train_loader, optimizer, scheduler, device, epochs=30):
    """Train the JEPA model."""
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        total_batches = len(train_loader)  # 获取总批次数
        scaler = GradScaler()
        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            # Training step
            loss_dict = model.training_step(batch.states, batch.actions)
            loss = loss_dict['loss']

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.adjust_learning_rate(epoch)

            epoch_loss += loss.item()

            # 每30个批次打印一次累计损失和进度
            if batch_idx % 30 == 0 or batch_idx == total_batches:
                print(f"Batch [{batch_idx}/{total_batches}], Cumulative Loss: {epoch_loss:.4f}")

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        model.update_target_encoder(momentum=0.98)  # EMA target encoder update

    print(f"Model checkpoint saved")
    save_checkpoint(model, optimizer, epoch=epoch, filepath=f"checkpoints/epoch_{epoch}_jepa.pth")


def save_checkpoint(model, optimizer, epoch, filepath):
    """Save model checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, filepath)

def main():
    # Set device
    device = get_device()
    os.makedirs("checkpoints", exist_ok=True)

    # Load data
    train_loader, probe_train_loader, val_loaders = load_data(device)

    # Initialize model
    model = JEPAModel(latent_dim=256).to(device)
    
    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = Scheduler(
        schedule=LRSchedule.Cosine,
        base_lr=0.002,
        data_loader=train_loader,
        epochs=30,
        optimizer=optimizer
    )

    # Train the model
    train_model(model, train_loader, optimizer, scheduler, device, epochs=30)

    # Save final checkpoint
    save_checkpoint(model, optimizer, epoch=30, filepath="jepa_model_checkpoint.pth")

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
