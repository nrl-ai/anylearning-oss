from torch.utils.tensorboard import SummaryWriter

import tempfile
from nanodet.util import NanoDetLightningLogger, cfg, load_config


def test_logger():
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = NanoDetLightningLogger(tmp_dir)

        writer = logger.experiment
        assert isinstance(writer, SummaryWriter)

        logger.info("test")

        logger.log_hyperparams({"lr": 1})

        logger.log_metrics({"mAP": 30.1}, 1)

        load_config(
            cfg,
            "./anylearning/training/models/nanodet/config/legacy_v0.x_configs/nanodet-m.yml",
        )
        logger.dump_cfg(cfg)

        logger.finalize(None)

        # Windows cannot delete a file that is still open, and the temporary
        # directory goes away at the end of this block.
        for handler in list(logger.logger.handlers):
            handler.close()
            logger.logger.removeHandler(handler)
