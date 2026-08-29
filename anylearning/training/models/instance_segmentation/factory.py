import os
import pickle

import numpy as np
import yaml

# Import from the module that defines it, not via instseg_trainer. The trainer
# imports this factory, so going back through it made the two modules circular:
# importing factory first raised ImportError, and it only ever worked because
# something happened to import the trainer first.
from anylearning.training.logging import TrainingLogsWriter

from .maskrcnn.inference import inference_fn as maskrcnn_inference_fn
from .maskrcnn.inference import rebuild_inference_config
from .maskrcnn.train import train_fn as maskrcnn_train_fn


class InstanceSegmentationModelFactory:
    def load_config(self, config_path: str):
        with open(config_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def train(self, config_path: str, logger: TrainingLogsWriter = None):
        config = self.load_config(config_path)
        if config["model"]["arch"] == "maskrcnn":
            maskrcnn_train_fn(config_path, logger)
        else:
            raise ValueError(f"Unsupported model architecture: {config['model']['arch']}")

    def inference(self, config: dict, model_path: str, image: np.ndarray):
        formatted_predictions, visualization_image = None, None
        if config["model"]["arch"] == "maskrcnn":
            detectron2_cfg = self.load_inference_config(config, model_path)
            class_names = [item["name"] for item in config["data"]["label_set"]]
            formatted_predictions, visualization_image = maskrcnn_inference_fn(
                detectron2_cfg, model_path, class_names, image
            )
        else:
            raise ValueError(f"Unsupported model architecture: {config['model']['arch']}")

        return formatted_predictions, visualization_image

    @staticmethod
    def load_inference_config(config: dict, model_path: str):
        """The detectron2 config for a trained model, wherever it survived.

        Training pickles it into the run's output folder, and that folder is
        deleted when the run finishes -- so this used to raise

            No such file or directory: .../training/<n>/training_output/detectron2_cfg.pkl

        which reached the user as a 500 from the Try dialog, on every instance
        segmentation model they had. A run keeps a copy beside the model now;
        models trained before that have no copy anywhere, and are rebuilt from
        the stored YAML instead.
        """
        name = config["detectron2_cfg_file"]
        candidates = [
            # Beside the model, which is what a registered model now carries.
            os.path.join(os.path.dirname(model_path), name),
            # The run's own folder: still there while a run is in progress, and
            # for anyone using --development, which keeps it.
            os.path.join(config["save_dir"], name),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                with open(candidate, "rb") as f:
                    return pickle.load(f)
        return rebuild_inference_config(config)
