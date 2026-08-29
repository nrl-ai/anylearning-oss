import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Import the logging class
from anylearning import settings
from anylearning.training import precision
from anylearning.training.logging import ConsoleLogsWriter, TrainingLogsWriter


# AverageMeter for tracking average loss


# The user can pin a run to the CPU (see anylearning/training/device_utils.py).
# Read from the environment rather than imported, so this vendored package keeps
# no dependency on the application around it.
def _anylearning_device():
    """ "cuda", "mps" or "cpu": what this run should train on.

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


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_transformations(img_size, is_train=True, augmentation=None):
    """Build the transform pipeline.

    `augmentation` comes from the training config and carries the user's
    choices; None keeps what this trainer always did, so inference and older
    projects are unaffected.
    """
    if not is_train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    settings = augmentation or {"horizontal_flip": True}
    steps = [transforms.Resize((img_size, img_size))]
    if settings.get("horizontal_flip", True):
        steps.append(transforms.RandomHorizontalFlip())
    if settings.get("vertical_flip"):
        steps.append(transforms.RandomVerticalFlip())
    if settings.get("rotation_degrees"):
        steps.append(transforms.RandomRotation(settings["rotation_degrees"]))
    if settings.get("color_jitter"):
        steps.append(
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        )
    steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(steps)


def get_dataloader(
    data_dir,
    img_size,
    batch_size,
    num_workers,
    is_train=True,
    device="",
    augmentation=None,
):
    transform = get_transformations(img_size, is_train, augmentation)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # The config used to ask for 0 workers, which decodes and resizes every
    # image on the training process's own thread. Measured on 1,072 images at
    # 224px: 65 img/s with 0 workers against 259 with 8, a 4x difference on the
    # same GPU. The right number is a property of the machine, so it comes from
    # settings with an automatic default rather than from the YAML.
    # MPS counts as a GPU here: the arithmetic leaves the CPU either way, so
    # the loader is what has to keep up.
    workers = settings.resolve_num_workers(
        num_workers, on_gpu=str(device) in ("cuda", "mps")
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=workers,
        pin_memory=settings.resolve_pin_memory(device),
        # Without this every epoch pays worker startup again, which dominates
        # on the small datasets this app is usually pointed at.
        persistent_workers=settings.resolve_persistent_workers(workers),
    )
    return dataloader


def load_or_create_model(config):
    model_arch = config["model"]["arch"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"].get("pretrained", "DEFAULT")

    resume_from = config["model"].get("resume_from", None)
    if resume_from:
        # Read on the CPU so a GPU-trained checkpoint also loads on a CPU-only
        # machine. The caller moves the model to the training device afterwards.
        loaded_model = torch.load(resume_from, map_location="cpu", weights_only=False)
        print(f"Continue training from checkpoint {resume_from}")
        # check architecture match
        if model_arch != loaded_model.architecture:
            raise ValueError(
                f"Model architecture '{model_arch}' does not match the pretrained model architecture '{loaded_model.architecture}'."
            )
        if num_classes != loaded_model.fc.out_features:
            warnings.warn(
                f"Number of classes in the current dataset ({num_classes}) does not match the number of classes in the pretrained model ({loaded_model.fc.out_features}). Reintialize the output layer",
                stacklevel=2,
            )
            loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)
        return loaded_model

    if model_arch in models.__dict__:
        model = getattr(models, model_arch)(weights=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.architecture = model_arch
    else:
        raise ValueError(f"Model architecture '{model_arch}' is not supported.")

    return model


def train_fn(config_path="cfg_classification.yaml", logger: TrainingLogsWriter = None):
    # The signature makes logger optional, so honour that instead of blowing up
    # on the first logger.write() call.
    logger = logger or ConsoleLogsWriter()
    config = load_config(config_path)

    # Set device and seeds
    device = torch.device(_anylearning_device())
    torch.manual_seed(config.get("seed", 67))
    # cudnn algorithm search, thread counts -- whatever the performance mode
    # calls for on this machine.
    settings.apply_torch_runtime(device)

    # Load data
    train_loader = get_dataloader(
        config["data"]["train_dir"],
        config["data"]["img_size"],
        config["training"]["batch_size"],
        config["data"]["num_workers"],
        is_train=True,
        device=device,
        # Training only. Augmenting validation would measure a different task
        # from the one being trained.
        augmentation=config["data"].get("augmentation"),
    )
    val_loader = get_dataloader(
        config["data"]["val_dir"],
        config["data"]["img_size"],
        config["training"]["batch_size"],
        config["data"]["num_workers"],
        is_train=False,
        device=device,
    )

    # Create a new model
    model = load_or_create_model(config)

    model = model.to(device)

    # Gradient checkpointing
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # Optimizer and scheduler
    optim_params = config["training"]["optim"]
    optimizer = getattr(optim, optim_params["name"])(
        model.parameters(),
        lr=optim_params["lr"],
        weight_decay=optim_params["weight_decay"],
        betas=optim_params["betas"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=float(config["training"]["min_lr"]),
    )

    # Mixed precision, decided per machine rather than per config file -- see
    # anylearning/training/precision.py. The scaler is a passthrough unless
    # fp16 is what this GPU ended up using, so the loop below is the same code
    # in every mode.
    plan = precision.from_config(config, device=device)
    logger.write(plan.describe())
    scaler = plan.scaler()

    # Training loop
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        logger.write(f"Starting epoch {epoch + 1}/{config['training']['epochs']}.")

        # Training phase
        model.train()
        train_loss_meter = AverageMeter()
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with plan.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                # Gradient clipping
                if config["training"].get("clip_grad_norm", None):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["clip_grad_norm"]
                    )

                scaler.step(optimizer)
                scaler.update()

                # Update average loss
                train_loss_meter.update(loss.item(), inputs.size(0))
                pbar.set_postfix(avg_loss=train_loss_meter.avg)
                pbar.update(1)

        # Validation phase
        model.eval()
        val_loss_meter = AverageMeter()
        corrects = 0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Validating", unit="batch") as pbar:
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    with plan.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss_meter.update(loss.item(), inputs.size(0))
                    corrects += (torch.argmax(outputs, dim=1) == labels).sum().item()
                    pbar.set_postfix(avg_loss=val_loss_meter.avg)
                    pbar.update(1)

        val_acc = corrects / len(val_loader.dataset)
        logger.write(f"Epoch {epoch + 1} complete. Validation Accuracy: {val_acc:.4f}")

        # Log metrics
        epoch_metrics = {
            "Epoch": epoch + 1,
            "Training Loss": train_loss_meter.avg,
            "Validation Loss": val_loss_meter.avg,
            "Validation Accuracy": val_acc,
        }
        logger.write_metrics(epoch_metrics)
        print(epoch_metrics)
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, save_dir / "best_model.pth")
        torch.save(model, save_dir / "last_model.pth")

        # Step scheduler
        scheduler.step()

    logger.write("Training complete")
