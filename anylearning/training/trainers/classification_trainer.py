import json
import os
import pathlib
import shutil
import traceback

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from sqlalchemy.orm import Session

from anylearning import config as anylearning_config
from anylearning.database import DataItem, TrainingParams, db_manager
from anylearning.training import augmentation
from anylearning.training.device_utils import (
    get_model_device,
    load_torch_model,
    load_torch_model_for_export,
)
from anylearning.training.logging import TrainingLogsWriter
from anylearning.training.models.classification.train import (
    get_transformations,
    train_fn,
)
from anylearning.training.trainers.base_trainer import BaseTrainer
from anylearning.utils.resources import resource_path

PACKAGE_NAME = "anylearning"
CONFIG_DIR = "training/configs/"


class ClassificationTrainer(BaseTrainer):
    # torchvision transforms on a single image: every one of these is exact,
    # because there are no boxes or masks to keep in step.
    AUGMENTATIONS = (
        augmentation.HORIZONTAL_FLIP,
        augmentation.VERTICAL_FLIP,
        augmentation.ROTATION,
        augmentation.COLOR_JITTER,
    )

    def __init__(
        self,
        training_folder: str,
        logger: TrainingLogsWriter,
        project_id: int,
        training_params: TrainingParams,
    ):
        print("Training folder:", training_folder)
        super().__init__(training_folder, logger, project_id, training_params)
        self.project_folder = pathlib.Path(anylearning_config.PROJECTS_ROOT) / str(
            self.project_id
        )

    def prepare_data(self):
        """
        Prepare the data for the training job.
        """

        print("training params:", self.training_params)

        subsets = ["train", "val", "test"]
        engine = db_manager.get_project_engine(self.project_id)
        session = Session(bind=engine)
        try:
            data_items = session.query(DataItem).all()

            for i, item in enumerate(data_items):
                current_item = session.query(DataItem).get(item.id)
                if current_item is None:
                    self.logger.write(
                        f"Warning: DataItem with id {item.id} not found. Skipping."
                    )
                    continue

                image_path = current_item.path
                image_path = self.project_folder / "data" / image_path
                image_path.parent.mkdir(parents=True, exist_ok=True)
                subset_path = self.data_folder / subsets[current_item.subset]
                subset_path.mkdir(parents=True, exist_ok=True)

                if item.class_id != -1:
                    class_path = subset_path / str(item.class_id)
                    class_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy(image_path, class_path)

                if i % 50 == 0:
                    self.logger.write(
                        f"Exported data item {i + 1} of {len(data_items)}"
                    )
            self.logger.write(f"Exported {len(data_items)} data items.")

            # Create every split directory, even the empty ones. Only the splits
            # that happened to contain an item were created above, so a project
            # imported with just train and test data left data/val missing and
            # training died on a bare FileNotFoundError from deep inside the
            # dataloader.
            for subset in subsets:
                (self.data_folder / subset).mkdir(parents=True, exist_ok=True)

            # Fail here, with something the user can act on, rather than letting
            # ImageFolder raise "Found no valid file" three layers down.
            for required in ("train", "val"):
                split_dir = self.data_folder / required
                has_images = any(
                    entry.is_dir() and any(entry.iterdir())
                    for entry in split_dir.iterdir()
                )
                if not has_images:
                    raise ValueError(
                        f"The {required} split has no labelled images. "
                        f"Assign images to it in the Dataset tab before training "
                        f"(train and val are both required; test is optional)."
                    )

            self.logger.write("Data exported successfully.")
            with open(self.training_folder / "labels.json", "w") as f:
                json.dump(self.labels, f)
            self.logger.write("Labels exported successfully.")
        finally:
            session.close()

    def prepare_config(self):
        self.config_path = self.training_folder / "cfg_classification.yml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.training_folder / "labels.json", "r") as f:
            labels = json.load(f)

        # We start with a very lightweight NanoDet config
        model_arch = self.training_params.model_architecture

        base_config_path = resource_path(
            PACKAGE_NAME, os.path.join(CONFIG_DIR, f"torchvision-cls-{model_arch}.yml")
        )
        if not os.path.exists(base_config_path):
            raise ValueError(f"Model architecture '{model_arch}' is not supported.")

        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        config["save_dir"] = str(self.output_folder)

        labels = sorted(labels, key=lambda x: x["id"])
        config["data"]["class_names"] = [x["name"] for x in labels]
        config["data"]["img_size"] = self.resolve_image_size(config["data"]["img_size"])

        config["data"]["augmentation"] = self.resolve_augmentation()
        config["data"]["train_dir"] = str(self.training_folder / "data" / "train")
        config["data"]["val_dir"] = str(self.training_folder / "data" / "val")
        config["data"]["test_dir"] = str(self.training_folder / "data" / "test")

        config["model"]["num_classes"] = len(labels)

        resume_from = self.resolve_pretrained_model_path()
        if resume_from:
            config["model"]["resume_from"] = resume_from

        config["training"]["batch_size"] = self.training_params.batch_size
        config["training"]["optim"]["lr"] = self.training_params.learning_rate
        config["training"]["epochs"] = self.training_params.epochs

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Return the content of the config file
        with open(self.config_path, "r") as f:
            return f.read()

    def train(self):
        try:
            train_fn(config_path=self.config_path, logger=self.logger)
        except Exception as e:
            self.logger.write(
                f"Error during training: {str(e)} {traceback.format_exc()}"
            )
            raise RuntimeError(f"Training process failed due to Error: {str(e)}") from e

    def get_model_path(self):
        print("Getting model path")
        # Empty strings, not None: best_model.pth is only written when validation
        # improves, so a run whose accuracy never rises above 0 legitimately
        # produces last_model.pth alone. os.path.exists(None) raises TypeError,
        # which used to crash that fallback instead of taking it.
        model_best_path = ""
        model_last_path = ""

        for root, _, files in os.walk(self.output_folder):
            for file in files:
                if file == "best_model.pth":
                    model_best_path = os.path.join(root, file)
                elif file == "last_model.pth":
                    model_last_path = os.path.join(root, file)

        if not os.path.exists(model_best_path) and not os.path.exists(model_last_path):
            print("No model found in training output")
            return False, None

        if not os.path.exists(model_best_path):
            print(
                "model_best.ckpt not found in training output, using model_last.ckpt instead"
            )
            model_best_path = model_last_path

        return True, model_best_path

    @staticmethod
    def run_inference(config_data: str, model_path: str, image: np.ndarray):
        """Run object detection inference on an image
        Args:
            config_data: str, the content of the config file
            model_path: str, the path to the model file
            image: np.ndarray, the image to run inference on
        Returns:
            results: list, the results of the inference
            visualization_image: np.ndarray, the visuali
            zed image
        """
        config = yaml.safe_load(config_data)
        model = load_torch_model(model_path)
        transform = get_transformations(config["data"]["img_size"], is_train=False)
        # BGR to RGB before PIL sees it. `run_inference` receives the image in
        # OpenCV's order -- that is what the detection and instance-segmentation
        # trainers want, and what `routers/model.py` hands over -- but
        # `Image.fromarray` interprets an array as RGB, so red and blue were
        # swapped for this trainer only. Training reads its images through PIL
        # from disk, correctly, so the model was trained on RGB and asked to
        # predict on BGR.
        #
        # It cost a wrong answer: on six colour photographs through the real
        # endpoint, one flipped its top-1 label (probability moved 0.31). It was
        # invisible for as long as it was because both sample datasets here are
        # greyscale, where the swap is a no-op.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        inp = transform(pil_image)
        inp = inp.unsqueeze(0).to(get_model_device(model))
        with torch.no_grad():
            output = torch.softmax(model(inp)[0], dim=-1)

        # Make a dictionary (class name -> prediction)
        predictions = {}
        for i, class_name in enumerate(config["data"]["class_names"]):
            predictions[class_name] = output[i].item()

        # Sort the predictions by confidence
        predictions = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        )

        # display top 5 predictions (if number of classes is less than 5, display all)
        top_k = min(5, len(predictions))
        top_k_predictions = {k: predictions[k] for k in list(predictions)[:top_k]}
        result = {f"top_{top_k}_class_probability": top_k_predictions}
        return result, image

    def export_onnx(self):
        # Return the content of the config file
        with open(self.config_path, "r") as f:
            config_data = f.read()
        config = yaml.safe_load(config_data)
        ret, model_path = self.get_model_path()
        if not ret:
            return None
        onnx_path = os.path.join(self.output_folder, "exported_model.onnx")
        torch_model = load_torch_model_for_export(model_path)
        img_size = config["data"]["img_size"]
        dummy_input = torch.rand(1, 3, img_size, img_size).to(
            get_model_device(torch_model)
        )
        # The TorchScript exporter, not dynamo. dynamo routes through
        # onnxscript, whose @script decorator calls inspect.getsource() to
        # parse a function's AST -- and a Nuitka-compiled binary has no
        # source, so it fails with "Decorator script does not work on
        # dynamically compiled function". Export runs after training and
        # gates model registration, so that discarded finished runs.
        torch.onnx.export(torch_model, dummy_input, onnx_path, dynamo=False)
        return onnx_path
