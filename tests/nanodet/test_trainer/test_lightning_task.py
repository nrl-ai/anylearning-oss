import tempfile
from unittest.mock import Mock

import numpy as np
import torch
from nanodet.trainer.task import TrainingTask
from nanodet.util import NanoDetLightningLogger, cfg, load_config

from anylearning.training.training_job import release_log_files_under


class DummyRunner:
    def __init__(self, task):
        self.task = task

    def test(self, save_dir):
        # Redirect cfg.save_dir into the temp directory. The config ships a
        # relative "workspace/nanodet_m", and now that on_train_epoch_end really
        # runs it writes checkpoints there -- i.e. into the repository root of
        # whoever ran the tests.
        self.task.cfg.defrost()
        self.task.cfg.save_dir = save_dir
        self.task.cfg.freeze()

        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.local_rank = 0
        trainer.use_ddp = False
        trainer.loggers = [NanoDetLightningLogger(save_dir)]
        trainer.num_val_batches = [1]

        # A real evaluator is needed now that the epoch-end hooks receive the
        # accumulated detections: previously the test passed an empty list, so
        # the evaluation branch was never entered at all.
        evaluator = Mock()
        evaluator.evaluate.return_value = {self.task.cfg.evaluator.save_key: 0.5}
        evaluator.results2json.return_value = {}
        self.task.evaluator = evaluator

        # on_test_epoch_end reads cfg.test_mode, which the legacy config omits.
        # The CfgNode is frozen after load_config, so unfreeze to add it.
        if "test_mode" not in self.task.cfg:
            self.task.cfg.defrost()
            self.task.cfg.test_mode = "val"
            self.task.cfg.freeze()

        optimizer = self.task.configure_optimizers()["optimizer"]

        trainer.optimizers = [optimizer]
        self.task._trainer = trainer

        self.task.on_train_start()
        assert self.task.current_epoch == 0

        dummy_batch = {
            "img": torch.randn((2, 3, 32, 32)),
            "img_info": {
                "height": torch.randn(2),
                "width": torch.randn(2),
                "id": torch.from_numpy(np.array([0, 1])),
            },
            "gt_bboxes": [
                np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                np.array(
                    [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32
                ),
            ],
            "gt_bboxes_ignore": [
                np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
                np.array(
                    [[7.0, 8.0, 9.0, 10.0], [7.0, 8.0, 9.0, 10.0]], dtype=np.float32
                ),
            ],
            "gt_labels": [np.array([1]), np.array([1, 2])],
            "warp_matrix": [np.eye(3), np.eye(3)],
        }

        def func(*args, **kwargs):
            pass

        self.task.scalar_summary = func
        self.task.training_step(dummy_batch, 0)

        self.task.optimizer_step(optimizer=optimizer)
        # on_*_epoch_end, not the removed *_epoch_end: Lightning 2.0 dropped the
        # latter, and calling them here would silently hit the inherited no-op
        # stubs instead of NanoDet's real logic.
        self.task.on_train_epoch_end()

        self.task.on_validation_epoch_start()
        self.task.validation_step(dummy_batch, 0)
        assert self.task.validation_step_outputs, "validation_step did not accumulate"
        self.task.on_validation_epoch_end()

        self.task.on_test_epoch_start()
        self.task.test_step(dummy_batch, 0)
        assert self.task.test_step_outputs, "test_step did not accumulate"
        self.task.on_test_epoch_end()

        # The accumulators must reset, or every epoch re-evaluates stale results.
        self.task.on_validation_epoch_start()
        assert self.task.validation_step_outputs == []


def test_lightning_training_task():
    load_config(
        cfg,
        "./anylearning/training/models/nanodet/config/legacy_v0.x_configs/nanodet-m.yml",
    )
    task = TrainingTask(cfg)
    runner = DummyRunner(task)
    # Own the temp directory for the duration of the run. The previous
    # `tempfile.TemporaryDirectory().name` dropped its only reference
    # immediately, so the directory could be removed while the logger wrote to it.
    with tempfile.TemporaryDirectory() as save_dir:
        try:
            runner.test(save_dir)
        finally:
            # NanoDetLightningLogger attaches a FileHandler to logs.txt inside
            # save_dir and never closes it. Windows refuses to delete a file a
            # handle still holds open, so TemporaryDirectory's own cleanup died
            # with "WinError 32: the process cannot access the file" -- after
            # the test had already passed. Released with the same helper the
            # training job uses before it removes a run's folder.
            release_log_files_under(save_dir)
