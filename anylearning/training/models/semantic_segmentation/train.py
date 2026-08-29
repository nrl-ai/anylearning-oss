from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from anylearning import settings
from anylearning.training import precision
from anylearning.training.logging import ConsoleLogsWriter, TrainingLogsWriter

# Import the logging class
from .dataset import SegmentationDataset

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


def get_transformations(img_size, mean, std, is_train=True, augmentation=None):
    """Image-only transforms. Anything spatial is applied in the dataset, where
    the mask can be moved with the image -- see SegmentationDataset._augment."""
    if is_train:
        settings = augmentation if augmentation is not None else {"color_jitter": True}
        steps = [transforms.Resize((img_size, img_size))]
        if settings.get("color_jitter"):
            steps.append(
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                )
            )
        steps += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        transform = transforms.Compose(steps)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    return transform


def get_dataloader(
    data_dir,
    class_name2id,
    img_size,
    mean,
    std,
    batch_size,
    num_workers,
    is_train=True,
    device="",
    augmentation=None,
):
    transform = get_transformations(img_size, mean, std, is_train, augmentation)
    dataset = SegmentationDataset(
        image_dir=data_dir,
        class_name2id=class_name2id,
        transform=transform,
        # Spatial augmentation needs the mask too, so it happens in the dataset
        # rather than in the image transform above.
        augmentation=augmentation if is_train else None,
    )
    # Worker count comes from the machine's performance mode rather than the
    # YAML: the right number depends on the hardware and on whether training is
    # on a GPU. See anylearning/settings.py for the measurements.
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
        drop_last=True,
        pin_memory=settings.resolve_pin_memory(device),
        persistent_workers=settings.resolve_persistent_workers(workers),
    )
    return dataloader


def load_or_create_model(config, logger: TrainingLogsWriter = None):
    model_arch = config["model"]["arch"]
    num_classes = config["model"]["num_classes"] + 1  # add background class
    # Honour config["model"]["pretrained"] instead of hardcoding "imagenet".
    # The hardcoded value meant the config key was silently ignored, so there was
    # no way to build the model without reaching out for encoder weights -- which
    # also made any "offline" test config a lie. The shipped
    # configs/deeplabv3-semseg.yml still specifies imagenet, so behaviour there
    # is unchanged.
    pretrained = config["model"].get("pretrained", "imagenet")

    model = smp.DeepLabV3Plus(
        encoder_name=model_arch,
        encoder_weights=pretrained,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )

    resume_from = config["model"].get("resume_from", None)
    if resume_from:
        # Read on the CPU so a GPU-trained checkpoint also loads on a CPU-only
        # machine. The caller moves the model to the training device afterwards.
        # weights_only=False is explicit: these checkpoints are whole pickled
        # modules written by this trainer, and the default flips to True in a
        # future torch release.
        last_model = torch.load(resume_from, map_location="cpu", weights_only=False)
        if logger:
            logger.write(f"Continue training from checkpoint {resume_from}")
        print(f"Continue training from checkpoint {resume_from}")
        # load pretrained weights
        load_status = model.load_state_dict(last_model.state_dict(), strict=False)
        print(f"Loaded pretrained weights: {load_status}")

    # Update BatchNorm layers to handle small batches
    # for module in model.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         module.momentum = 0.1
    #         module.track_running_stats = False  # Disable running stats for small batches

    return model


def calculate_iou(
    y_pred, y_true, num_classes, include_background=True, background_class=0
):
    # Move tensors to CPU first
    y_pred = y_pred.cpu().numpy()
    y_true = y_true.cpu().numpy()

    iou_scores = []

    for c in range(num_classes):
        if not include_background and c == background_class:
            continue  # Skip background class if not included

        true_class = y_true == c
        pred_class = y_pred == c

        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()

        iou = intersection / union if union != 0 else 0
        iou_scores.append(iou)

    mean_iou = np.mean(iou_scores) if iou_scores else 0  # Avoid division by zero
    return mean_iou


def train_fn(config_path, logger: TrainingLogsWriter = None):
    # The signature makes logger optional, so honour that instead of blowing up
    # on the first logger.write() call.
    logger = logger or ConsoleLogsWriter()
    config = load_config(config_path)

    # Set device and seeds
    device = torch.device(_anylearning_device())
    settings.apply_torch_runtime(device)
    labels_set = config["data"]["label_set"]
    class_name2id = {v["name"]: v["id"] for v in labels_set}
    torch.manual_seed(config.get("seed", 67))
    verbose_steps = config["training"].get("verbose_steps", 20)

    # Load data
    train_loader = get_dataloader(
        config["data"]["train_dir"],
        class_name2id,
        config["data"]["img_size"],
        config["data"]["normalize"]["mean"],
        config["data"]["normalize"]["std"],
        config["training"]["batch_size"],
        config["data"]["num_workers"],
        is_train=True,
        device=device,
        augmentation=config["data"].get("augmentation"),
    )
    val_loader = get_dataloader(
        config["data"]["val_dir"],
        class_name2id,
        config["data"]["img_size"],
        config["data"]["normalize"]["mean"],
        config["data"]["normalize"]["std"],
        config["training"]["batch_size"],
        config["data"]["num_workers"],
        is_train=False,
        device=device,
    )

    # Create a new model
    model = load_or_create_model(config, logger)

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

    # Initialize with warm-up lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = optim_params["lr"] * 0.1

    # Create scheduler with full lr
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(
            config["training"]["epochs"] - 1, 1
        ),  # Subtract 1 to account for warm-up epoch
        eta_min=float(config["training"]["min_lr"]),
    )

    # Mixed precision, decided per machine -- see
    # anylearning/training/precision.py. Matches the classification trainer.
    plan = precision.from_config(config, device=device)
    logger.write(plan.describe())
    scaler = plan.scaler()

    # Training loop
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    best_iou = 0
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["training"]["epochs"]):
        logger.write(
            f"===== Starting epoch {epoch + 1}/{config['training']['epochs']} ====="
        )

        # Set full learning rate after warm-up epoch
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group["lr"] = optim_params["lr"]

        # Training phase
        model.train()
        train_loss_meter = AverageMeter()
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
            for iteration, (inputs, labels) in enumerate(train_loader):
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
                pbar.set_postfix(
                    epoch=epoch + 1,
                    train_avg_loss=train_loss_meter.avg,
                    lr=f"{optimizer.param_groups[0]['lr']:.6f}",
                )
                pbar.update(1)

                if iteration % verbose_steps == 0:
                    logger.write(
                        f"Epoch {epoch + 1}, Step {iteration + 1}. Training Loss: "
                        f"{train_loss_meter.avg:.4f}. Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
                    )

        # Validation phase
        model.eval()
        val_loss_meter = AverageMeter()
        val_iou_meter = AverageMeter()
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Validating", unit="batch") as pbar:
                for iteration, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    with plan.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    val_loss_meter.update(loss.item(), inputs.size(0))

                    outputs_discrete = torch.argmax(outputs, dim=1)
                    val_iou = calculate_iou(
                        outputs_discrete,
                        labels,
                        num_classes=len(labels_set) + 1,
                        include_background=False,
                    )

                    val_iou_meter.update(val_iou, inputs.size(0))
                    pbar.set_postfix(
                        epoch=epoch + 1,
                        val_avg_loss=val_loss_meter.avg,
                        val_iou=val_iou_meter.avg,
                    )
                    pbar.update(1)

                    if iteration % verbose_steps == 0:
                        logger.write(
                            f"Epoch {epoch + 1}, Step {iteration + 1}. Validation Loss: "
                            f"{val_loss_meter.avg:.4f}. Validation IoU: {val_iou_meter.avg:.4f}"
                        )

        logger.write(
            f"===== Epoch {epoch + 1} complete. Average IoU score: {val_iou_meter.avg:.4f} ====="
        )

        # Log metrics
        epoch_metrics = {
            "Epoch": epoch + 1,
            "Training Loss": train_loss_meter.avg,
            "Validation Loss": val_loss_meter.avg,
            "Validation IoU": val_iou_meter.avg,
        }
        logger.write_metrics(epoch_metrics)
        print(epoch_metrics)
        # Save best model
        if val_iou_meter.avg > best_iou:
            best_iou = val_iou_meter.avg
            torch.save(model, save_dir / "best_model.pth")
        torch.save(model, save_dir / "last_model.pth")

        # Step scheduler
        scheduler.step()

    logger.write("===== Training complete =====")
