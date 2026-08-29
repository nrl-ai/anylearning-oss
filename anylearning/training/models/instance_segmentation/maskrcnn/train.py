import numpy as np
import torch
import torch.optim as optim
import yaml

import detectron2.utils.comm as comm
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.config.config import CfgNode as CN
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.data.samplers import InferenceSampler
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage

import gc
import os
import pathlib
import pickle
import time
from anylearning.training import precision as anylearning_precision
from anylearning.training.logging import TrainingLogsWriter

from ...utils import AverageMeter



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


def _anylearning_cuda_available():
    return _anylearning_device() == "cuda"

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def bundled_checkpoint(url: str) -> str:
    """The local path of an already-downloaded checkpoint, or the URL itself.

    Handing detectron2 a URL is what makes it lock. iopath routes a URL through
    its *caching* handler, which takes a portalocker lock on `<file>.lock`
    beside the cached copy before it will even read it -- and that is the whole
    reason installed copies needed a writable mirror of the weights directory.
    A plain local path goes through `NativePathHandler`, which never locks.

    We ship these files, so the download never happens and the lock protects
    nothing. Resolving the path ourselves removes it: no lock, no mirror, and
    on Windows no 426 MB of copies in the user's data root, since hardlinks to
    a Program Files source are denied to standard users.

    Falls back to the URL when the file is not where it should be, so a build
    without bundled weights behaves exactly as before and downloads it.
    """
    import os
    from urllib.parse import urlparse

    cache = os.environ.get("FVCORE_CACHE")
    if not cache:
        return url
    candidate = pathlib.Path(cache) / urlparse(url).path.lstrip("/")
    return str(candidate) if candidate.is_file() else url


def set_model_config_and_weights(backbone: str, detectron2_cfg: CfgNode):
    if backbone == "resnet50":
        weights = "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_50_FPN_400ep_LSJ/42019571/model_final_14d201.pkl"
        detectron2_cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        detectron2_cfg.MODEL.WEIGHTS = bundled_checkpoint(weights)
    elif backbone == "resnet101":
        weights = "https://dl.fbaipublicfiles.com/detectron2/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ/42073830/model_final_f96b26.pkl"
        detectron2_cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        )
        detectron2_cfg.MODEL.WEIGHTS = bundled_checkpoint(weights)
    else:
        raise ValueError(f"Backbone {backbone} not supported")


def get_detectron2_config(config: dict, logger: TrainingLogsWriter = None):
    OUTPUT_DIR = config["save_dir"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detectron2_cfg = get_cfg()
    set_model_config_and_weights(config["model"]["backbone"], detectron2_cfg)
    detectron2_cfg.DATASETS.TRAIN = ("train_ds",)
    detectron2_cfg.DATASETS.TEST = ("val_ds",)
    detectron2_cfg.OUTPUT_DIR = OUTPUT_DIR
    detectron2_cfg.DATALOADER.NUM_WORKERS = config["data"]["num_workers"]
    detectron2_cfg.INPUT.MAX_SIZE_TRAIN = config["data"]["img_size"]
    detectron2_cfg.INPUT.MAX_SIZE_TEST = config["data"]["img_size"]

    # CUDA or the CPU, never Metal. detectron2 ships its own kernels -- ROIAlign,
    # NMS, the mask head's samplers -- and they are written for CUDA and the CPU
    # only. Asking for "mps" does not raise a missing-operator error, which would
    # at least be legible: it runs, and then the Metal command buffer fails with
    #
    #     Error: command buffer exited with error status.
    #     Internal Error (0000000e:Internal Error)
    #
    # after which the process produces no model and no traceback. Measured on an
    # M1: every attempt ended that way, with and without
    # PYTORCH_ENABLE_MPS_FALLBACK. So this trainer stays on the CPU on a Mac,
    # and says so, rather than failing in a way nobody can act on.
    device = _anylearning_device()
    if device == "mps":
        # Phrased as another "Training device:" line on purpose. That is the
        # line the log is read for, by people and by smoke_test_training's
        # device assertion, and the last one wins -- so the run reports where it
        # actually happened rather than where it was sent.
        log_and_print(
            logger,
            "Training device: CPU -- Apple's GPU cannot run Mask R-CNN. "
            "detectron2's own operators are written for CUDA and the CPU.",
        )
        device = "cpu"
    detectron2_cfg.MODEL.DEVICE = device

    # Flipping is the augmentation detectron2 exposes as configuration, and its
    # dataset mapper carries boxes and masks along with the image -- which is
    # why only this one is offered. RANDOM_FLIP takes one mode, so a request for
    # both directions keeps the horizontal one: it is the safe default for
    # photographs, where "upside down" is usually a different scene.
    augmentation = config["data"].get("augmentation") or {}
    if augmentation:
        if augmentation.get("horizontal_flip"):
            detectron2_cfg.INPUT.RANDOM_FLIP = "horizontal"
        elif augmentation.get("vertical_flip"):
            detectron2_cfg.INPUT.RANDOM_FLIP = "vertical"
        else:
            detectron2_cfg.INPUT.RANDOM_FLIP = "none"

    detectron2_cfg.SOLVER.IMS_PER_BATCH = config["training"]["batch_size"]
    detectron2_cfg.SOLVER.BASE_LR = float(config["training"]["optim"]["lr"])
    detectron2_cfg.SOLVER.BASE_LR_END = float(config["training"]["min_lr"])

    detectron2_cfg.SOLVER.LR_SCHEDULER_NAME = config["training"]["scheduler"]
    detectron2_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES = config["model"]["num_classes"]
    detectron2_cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2

    detectron2_cfg.MODEL.ROI_BOX_HEAD.NORM = "FrozenBN"
    detectron2_cfg.MODEL.ROI_MASK_HEAD.NORM = "FrozenBN"
    detectron2_cfg.MODEL.FPN.NORM = "FrozenBN"

    return detectron2_cfg


def get_datasets(config: dict):
    register_coco_instances(
        "train_ds", {}, config["data"]["train_ann_file"], config["data"]["train_dir"]
    )
    register_coco_instances("val_ds", {}, config["data"]["val_ann_file"], config["data"]["val_dir"])

    train_ds = DatasetCatalog.get("train_ds")
    val_ds = DatasetCatalog.get("val_ds")

    return train_ds, val_ds


def log_and_print(logger: TrainingLogsWriter, log_line: str):
    if logger:
        logger.write(log_line)
    print(log_line)


def create_model(
    detectron2_cfg: CfgNode, resume_from: str = None, logger: TrainingLogsWriter = None
):
    if resume_from:
        if os.path.exists(resume_from):
            log_and_print(logger, f"Resuming from checkpoint {resume_from}")
            detectron2_cfg.MODEL.WEIGHTS = resume_from
        else:
            raise FileNotFoundError(f"Checkpoint file {resume_from} not found")

    model = build_model(detectron2_cfg)
    return model


def evaluate_average_precision(model, data_loader, evaluator):
    res = inference_on_dataset(model, data_loader, evaluator)
    return res


#: How many float16 overflows in a row stop the run. The scaler halves its
#: scale on each one, so a handful is normal at the start of training and a
#: sustained run of them is not something a smaller scale will fix.
MAX_CONSECUTIVE_OVERFLOWS = 20


def compute_loss(model, data, require_finite=True):
    loss_dict = model(data)
    loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
    losses = sum(loss_dict.values())
    # require_finite=False under float16: an overflow there is what the gradient
    # scaler exists to absorb, and the caller counts them rather than stopping
    # on the first one. In every other mode a non-finite loss is a real
    # divergence and stopping is the right answer.
    if require_finite:
        assert torch.isfinite(losses).all(), loss_dict

    return losses, loss_dict_reduced


def train_fn(config_path, logger: TrainingLogsWriter = None):
    config = load_config(config_path)

    detectron2_cfg = get_detectron2_config(config, logger)

    train_ds, val_ds = get_datasets(config)

    # Custom config
    detectron2_cfg.CUSTOM = CN()
    detectron2_cfg.CUSTOM.SEED = config.get("seed", 67)
    torch.manual_seed(detectron2_cfg.CUSTOM.SEED)
    detectron2_cfg.CUSTOM.VERBOSE_STEPS = config["training"].get("verbose_steps", 20)
    detectron2_cfg.CUSTOM.EPOCHS = config["training"]["epochs"]
    detectron2_cfg.CUSTOM.N_TRAIN_ITERS_1_EPOCH = int(
        np.ceil(len(train_ds) / detectron2_cfg.SOLVER.IMS_PER_BATCH)
    )
    detectron2_cfg.CUSTOM.N_VALID_ITERS_1_EPOCH = int(
        np.ceil(len(val_ds) / detectron2_cfg.SOLVER.IMS_PER_BATCH)
    )

    detectron2_cfg.SOLVER.MAX_ITER = (
        detectron2_cfg.CUSTOM.EPOCHS * detectron2_cfg.CUSTOM.N_TRAIN_ITERS_1_EPOCH
    )
    detectron2_cfg.SOLVER.WARMUP_ITERS = detectron2_cfg.CUSTOM.N_VALID_ITERS_1_EPOCH
    detectron2_cfg.TEST.EVAL_PERIOD = detectron2_cfg.CUSTOM.N_TRAIN_ITERS_1_EPOCH
    detectron2_cfg.SOLVER.CHECKPOINT_PERIOD = detectron2_cfg.CUSTOM.N_TRAIN_ITERS_1_EPOCH

    print(
        "Number of train iterations in 1 epoch:",
        detectron2_cfg.CUSTOM.N_TRAIN_ITERS_1_EPOCH,
    )

    # Save detectron2 config
    with open(os.path.join(config["save_dir"], config["detectron2_cfg_file"]), "wb") as f:
        pickle.dump(detectron2_cfg, f)

    # Load data
    train_loader = build_detection_train_loader(detectron2_cfg)
    val_loader = build_detection_train_loader(
        val_ds,
        mapper=DatasetMapper(
            detectron2_cfg,
            instance_mask_format=detectron2_cfg.INPUT.MASK_FORMAT,
            is_train=True,
        ),
        sampler=InferenceSampler(size=len(val_ds)),
        total_batch_size=detectron2_cfg.SOLVER.IMS_PER_BATCH,
    )

    val_loader_for_metrics = build_detection_test_loader(detectron2_cfg, "val_ds")

    val_evaluator = COCOEvaluator(
        "val_ds",
        ("segm",),
        False,
        output_dir=os.path.join(detectron2_cfg.OUTPUT_DIR, "val_inference"),
    )

    # Create a new model
    model = create_model(
        detectron2_cfg, resume_from=config["training"].get("resume_from"), logger=logger
    )

    # Optimizer and scheduler
    optim_params = config["training"]["optim"]
    optimizer = getattr(optim, optim_params["name"])(
        model.parameters(),
        lr=optim_params["lr"],
        weight_decay=optim_params["weight_decay"],
        betas=optim_params["betas"],
    )

    # Create scheduler with full lr
    scheduler = build_lr_scheduler(detectron2_cfg, optimizer)

    # Mixed precision. detectron2 has `SOLVER.AMP.ENABLED`, but it is read by
    # its `AMPTrainer`, and this loop is hand-written -- so the flag would have
    # been set and silently done nothing. The autocast goes where the forward
    # pass is instead.
    #
    # Mask R-CNN is the model here that most wants it: it is the largest, the
    # only one that runs out of memory on a laptop GPU, and half of what it
    # spends per iteration is convolution.
    plan = anylearning_precision.from_config(config, device=_anylearning_device())
    log_and_print(logger, plan.describe())
    scaler = plan.scaler()

    checkpointer = DetectionCheckpointer(
        model, detectron2_cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    checkpointer.resume_or_load(detectron2_cfg.MODEL.WEIGHTS, resume=False)

    start_iter = 1
    best_val_mAP = 0
    overflows = 0
    skipped_steps = 0
    steps_taken = 0

    train_loss_score = AverageMeter()
    train_times = AverageMeter()

    with EventStorage(0) as storage:
        for train_data, train_iteration in zip(
            train_loader, range(start_iter, detectron2_cfg.SOLVER.MAX_ITER + 1)
        ):
            model.train()

            optimizer.zero_grad()

            # do training
            storage.step()

            _start_time = time.time()

            epoch = int(np.ceil(train_iteration / detectron2_cfg.CUSTOM.N_TRAIN_ITERS_1_EPOCH))

            with plan.autocast():
                train_total_loss, train_detail_loss = compute_loss(
                    model, train_data, require_finite=not plan.tolerates_overflow
                )

            # A non-finite loss is the visible case, and the rarer one.
            if not torch.isfinite(train_total_loss).all():
                overflows += 1
                if overflows > MAX_CONSECUTIVE_OVERFLOWS:
                    raise RuntimeError(
                        f"The loss has been infinite for {overflows} iterations "
                        "in a row. Mixed precision cannot recover from that; "
                        "lower the learning rate, or set "
                        "ANYLEARNING_MIXED_PRECISION=off to train in float32."
                    )
            else:
                overflows = 0

            scaler.scale(train_total_loss).backward()

            # Watching the scale, not the loss. Measured on an RTX 2070, this
            # loop's losses stayed finite for every single iteration while the
            # *scaled gradients* overflowed -- and an overflow there makes
            # `scaler.step` discard the update silently. Four of the first
            # seven updates went that way, and the run finished at mAP@0.5 0.13
            # against float32's 0.31: a real regression that nothing in the log
            # would have explained.
            #
            # The scaler halves its scale on exactly the iterations it skips,
            # so comparing the scale across `update()` is the one reliable
            # signal that the optimiser did nothing.
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < scale_before:
                skipped_steps += 1
            scheduler.step()

            train_times.update(time.time() - _start_time)
            steps_taken += 1

            train_loss_score.update(train_total_loss.detach().cpu().item(), train_loader.batch_size)

            if train_iteration % detectron2_cfg.CUSTOM.VERBOSE_STEPS == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                log_and_print(
                    logger,
                    f"Iteration {train_iteration}, Epoch {epoch},  Train Total Loss: "
                    f"{train_loss_score.avg:.4f}. Learning Rate: {current_lr:.6f}. "
                    f"Average time per iter: {train_times.avg:.4f}s",
                )

            del train_data, train_total_loss, train_detail_loss
            if _anylearning_cuda_available():
                torch.cuda.empty_cache()
            gc.collect()

            if train_iteration % detectron2_cfg.CUSTOM.N_TRAIN_ITERS_1_EPOCH == 0:
                # Say it in the log the user reads. A run that trains on a
                # fraction of its updates looks exactly like a run that trained
                # badly, and this is the only place the difference is visible.
                if skipped_steps:
                    log_and_print(
                        logger,
                        f"Mixed precision skipped {skipped_steps} of "
                        f"{steps_taken} optimiser steps so far (gradient "
                        f"overflow; scale now {scaler.get_scale():.0f}). Set "
                        "ANYLEARNING_MIXED_PRECISION=off if the model does not "
                        "reach the accuracy you expect.",
                    )

                valid_loss_score = AverageMeter()

                for val_data in val_loader:
                    with torch.no_grad(), plan.autocast():
                        # Same tolerance as the training path. Without it an
                        # overflow during *validation* raises an AssertionError
                        # out of compute_loss and ends a run that was training
                        # perfectly well -- and validation is where the model is
                        # scored, so the failure would land after the work.
                        val_total_loss, val_detail_loss = compute_loss(
                            model, val_data, require_finite=not plan.tolerates_overflow
                        )

                        valid_loss_score.update(
                            val_total_loss.detach().cpu().item(), val_loader.batch_size
                        )
                        del val_data, val_total_loss, val_detail_loss
                        if _anylearning_cuda_available():
                            torch.cuda.empty_cache()
                        gc.collect()

                # compute APs
                val_ap_detailed = evaluate_average_precision(
                    model, val_loader_for_metrics, val_evaluator
                )
                val_ap = (
                    val_ap_detailed["segm"]["AP"] * 0.01
                    if not np.isnan(val_ap_detailed["segm"]["AP"])
                    else 0
                )
                val_ap_05 = (
                    val_ap_detailed["segm"]["AP50"] * 0.01
                    if not np.isnan(val_ap_detailed["segm"]["AP50"])
                    else 0
                )
                # log to file and to std out
                log_line = f"===== Epoch: {epoch} completed =====\n"
                log_line += (
                    f"Train total loss:{train_loss_score.avg:.4f} - "
                    f"Valid total loss: {valid_loss_score.avg:.4f}. \n"
                )
                log_line += (
                    f"Valid mAP@0.5: {val_ap_05:.4f} - Valid mAP@0.5:0.95: {val_ap:.4f}. \n\n"
                )

                log_and_print(logger, log_line)
                metric_dict = {
                    "Training Loss": train_loss_score.avg,
                    "Validation Loss": valid_loss_score.avg,
                    "Validation mAP@0.5": val_ap_05,
                    "Validation mAP@0.5:0.95": val_ap,
                }

                if logger:
                    logger.write_metrics(metric_dict)
                print(metric_dict)

                # save last model
                torch.save(model, os.path.join(detectron2_cfg.OUTPUT_DIR, "last_model.pth"))

                if val_ap > best_val_mAP:
                    best_val_mAP = val_ap
                    # save best model
                    torch.save(model, os.path.join(detectron2_cfg.OUTPUT_DIR, "best_model.pth"))

                # reset training logging (end of epoch)
                train_loss_score = AverageMeter()
