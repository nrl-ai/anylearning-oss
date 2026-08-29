# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import warnings

import pytorch_lightning as pl
import torch

from anylearning import settings as anylearning_settings
from anylearning.training import precision as anylearning_precision
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)
from pytorch_lightning.callbacks import TQDMProgressBar



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


def _anylearning_precision(configured):
    """Lightning's name for the precision this machine should train in.

    NanoDet's own configs carry `device.precision: 32  # set to 16 to use AMP
    training`, which is a decision made when the file was written rather than
    where the run happens. `precision.resolve()` makes it where the run
    happens: bf16 on an Ampere card, fp16 on an older one, and fp32 on a CPU,
    where 16-bit autocast is not a speed-up but a different, slower path.

    A config that explicitly asks for something other than 32 is left alone --
    it is the only way left to pin a run to one precision from the file.
    """
    if configured not in (32, "32", "32-true", None):
        return configured

    plan = anylearning_precision.resolve()
    if plan.dtype is torch.bfloat16:
        return "bf16-mixed"
    if plan.dtype is torch.float16:
        return "16-mixed"
    return configured if configured is not None else 32

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args


class TrainArgs:
    config: str = ""
    local_rank: int = -1
    seed: int = None


def main(args, logger=None):
    load_config(cfg, args.config)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError(
            "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
            "but got {} and {}".format(
                cfg.model.arch.head.num_classes, len(cfg.class_names)
            )
        )
    local_rank = int(args.local_rank)
    # cudnn is CUDA's, and enabling it on a machine without one is a no-op that
    # only reads as if the run were on a GPU.
    if _anylearning_cuda_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)

    if logger is None:
        logger = NanoDetLightningLogger(cfg.save_dir)
        logger.dump_cfg(cfg)

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "test")

    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    # workers_per_gpu is 1 in the shipped configs. The right number depends on
    # the machine and on whether there is a GPU to keep fed, so it comes from
    # the app's performance mode; see anylearning/settings.py.
    # MPS counts as a GPU here: the arithmetic leaves the CPU either way, so
    # the loader is what has to keep up.
    _device = _anylearning_device()
    _on_gpu = _device != "cpu"
    # Page-locked host memory is a CUDA transfer optimisation. It was hardcoded
    # True, which on any other backend is a copy nobody benefits from -- and on
    # MPS torch warns about it once per worker.
    _pin_memory = _device == "cuda"
    _workers = anylearning_settings.resolve_num_workers(
        cfg.device.workers_per_gpu, on_gpu=_on_gpu
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=_workers,
        persistent_workers=anylearning_settings.resolve_persistent_workers(_workers),
        pin_memory=_pin_memory,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=_workers,
        persistent_workers=anylearning_settings.resolve_persistent_workers(_workers),
        pin_memory=_pin_memory,
        collate_fn=naive_collate,
        drop_last=False,
    )

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    if "load_model" in cfg.schedule:
        ckpt = torch.load(
            cfg.schedule.load_model, map_location="cpu", weights_only=False
        )
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))

    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )
    # "auto", not None: Lightning 1.9 accepted None for devices/strategy, but
    # Lightning 2 rejects it outright with "You selected an invalid strategy
    # name: strategy=None".
    #
    # The backend comes from the environment, not from gpu_ids: gpu_ids can only
    # say "which CUDA device", and there is no number that means Metal. The
    # gpu_ids == -1 sentinel is still honoured for configs written before this.
    if _device == "mps":
        # Apple's Metal backend. One device by definition: there is one GPU and
        # it is part of the chip, so gpu_ids means nothing here.
        logger.info("Using Apple Metal (MPS) training")
        accelerator, devices, strategy, precision = (
            "mps",
            1,
            "auto",
            _anylearning_precision(cfg.device.precision),
        )
    elif _device == "cpu" or str(cfg.device.gpu_ids) == "-1":
        logger.info("Using CPU training")
        accelerator, devices, strategy, precision = (
            "cpu",
            "auto",
            "auto",
            _anylearning_precision(cfg.device.precision),
        )
    else:
        accelerator, devices, strategy, precision = (
            "gpu",
            cfg.device.gpu_ids,
            "auto",
            _anylearning_precision(cfg.device.precision),
        )

    # Both lines on purpose. Lightning's vocabulary ("bf16-mixed") is what was
    # actually passed to the Trainer and is worth having in the log, but the
    # other four trainers write "Mixed precision: ..." and this one did not, so
    # object detection was the one project type whose log could not be checked
    # the same way as the rest.
    logger.info(f"Precision: {precision}")
    logger.info(anylearning_precision.resolve().describe())

    # Guard on the list case explicitly. devices is now "auto" for CPU runs, and
    # len("auto") is 4, which would otherwise select DDP on a single-CPU machine.
    if isinstance(devices, (list, tuple)) and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True) and accelerator == "gpu",
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
        precision=precision,
    )

    trainer.fit(task, train_dataloader, val_dataloader, ckpt_path=model_resume_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
