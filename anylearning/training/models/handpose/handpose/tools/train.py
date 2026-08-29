import torch
import torch.utils
import yaml
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
from handpose.datasets import HandPoseDataset
from handpose.models.mlp import MLP



# The user can pin a run to the CPU (see anylearning/training/device_utils.py).
# Read from the environment rather than imported, so this vendored package keeps
# no dependency on the application around it.
def _anylearning_device():
    """"cuda", "mps" or "cpu": what this run should train on.

    Mirrors `anylearning.training.device_utils.device_type()`; keep the two in
    step. CUDA first because a machine with both is a machine with a real GPU.
    """
    import os

    if os.environ.get("ANYLEARNING_TRAINING_DEVICE", "").strip().lower() == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 -- a torch without the backend is an answer
        pass
    return "cpu"

def create_dataset(annotation_path, normalize=False):
    """
    Create dataset from annotation paths:
    """
    ds = HandPoseDataset(annotation_path, normalize=normalize)

    return ds


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def prepare_batches(dataset, batch_size=32, shuffle=True):
    """
    Create batches from dataset
    """
    torch.manual_seed(42)
    # Get all landmarks and labels
    all_landmarks = []
    all_labels = []

    for landmarks, label in dataset:
        # Skip not detected image
        if (landmarks == 0).all():
            continue
        all_landmarks.append(landmarks)
        all_labels.append(label)

    # Convert to tensors
    landmarks_tensor = torch.stack(all_landmarks)
    labels_tensor = torch.tensor(all_labels)

    # Shuffle the data
    if shuffle:
        indices = torch.randperm(len(all_landmarks))
        landmarks_tensor = landmarks_tensor[indices]
        labels_tensor = labels_tensor[indices]

    # Create batches
    batches = []
    for i in range(0, len(landmarks_tensor), batch_size):
        batch_landmarks = landmarks_tensor[i : i + batch_size]
        batch_labels = labels_tensor[i : i + batch_size]
        batches.append((batch_landmarks, batch_labels))

    return batches


def load_checkpoint(filepath, model, optimizer, scheduler=None):
    # Read on the CPU so a GPU-trained checkpoint also loads on a CPU-only machine
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    last_train_loss = checkpoint["train_loss"]
    last_val_loss = checkpoint["val_loss"]
    return model, optimizer, scheduler, start_epoch, last_train_loss, last_val_loss


def train(config_path: str, logger):
    config = load_config(config_path)

    # Create save_dir up front. torch.save() will not create parent directories,
    # so without this the run trains to completion and then dies on the first
    # checkpoint. It only worked because BaseTrainer.prepare_folders() happened
    # to have made the directory already -- the function was not self-sufficient.
    os.makedirs(config["save_dir"], exist_ok=True)

    # Absent in configs written before this existed, and those models were
    # trained on raw coordinates -- so the default has to stay False.
    normalize = bool(config["data"].get("normalize_landmarks", False))
    train_ds = create_dataset(config["data"]["train"]["annotation_path"], normalize)
    val_ds = create_dataset(config["data"]["val"]["annotation_path"], normalize)

    train_batches = prepare_batches(train_ds, config["data"]["train"]["batch_size"])
    val_batches = prepare_batches(val_ds, config["data"]["val"]["batch_size"])

    start_epoch = 0
    last_train_loss = None
    last_val_loss = None
    model = MLP(config)
    device = torch.device(_anylearning_device())
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["schedule"]["optimizer"]["lr"],
        weight_decay=config["schedule"]["optimizer"]["weight_decay"],
        betas=config["schedule"]["optimizer"]["betas"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["schedule"]["lr_schedule"]["T_max"],
        eta_min=float(config["schedule"]["lr_schedule"]["eta_min"]),
    )

    if "load_model" in config["schedule"].keys():
        print("Start training from last checkpoint.")
        model, optimizer, scheduler, start_epoch, last_train_loss, last_val_loss = (
            load_checkpoint(
                config["schedule"]["load_model"], model, optimizer, scheduler
            )
        )

    best_val_acc = 0
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, start_epoch + config["schedule"]["epochs"]):
        model.train()

        train_loss = 0 if last_train_loss is None else last_train_loss
        train_correct = 0
        train_total = 0

        for batch_landmarks, batch_labels in train_batches:
            batch_landmarks = batch_landmarks.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(batch_landmarks)
            loss = criterion(outputs, batch_labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()

            # Validation
            model.eval()
            val_loss = 0 if last_val_loss is None else last_val_loss
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_landmarks, batch_labels in val_batches:
                    batch_landmarks = batch_landmarks.to(device)
                    batch_labels = batch_labels.to(device)

                    outputs = model(batch_landmarks)
                    loss = criterion(outputs, batch_labels)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()

        # Calculate metrics
        train_loss = train_loss / len(train_batches)
        train_acc = train_correct / train_total
        val_loss = val_loss / len(val_batches)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch + 1}/{start_epoch + config['schedule']['epochs']}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | ")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        # Define checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "train_loss": train_loss,
        }

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(config["save_dir"], "model_best.ckpt"))
        # Log metrics
        eval_results = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        logger.log_metrics(eval_results, epoch)
        # Update scheduler
        scheduler.step()
        # Save last checkpoint
        torch.save(checkpoint, os.path.join(config["save_dir"], "model_last.ckpt"))
