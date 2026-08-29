import json
import os
import pathlib
import re
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
    get_device,
    get_model_device,
    load_torch_model,
    load_torch_model_for_export,
)
from anylearning.training.logging import TrainingLogsWriter
from anylearning.training.models.semantic_segmentation import (
    get_transformations,
    train_fn,
)
from anylearning.training.trainers.base_trainer import BaseTrainer
from anylearning.utils.resources import resource_path

PACKAGE_NAME = "anylearning"
CONFIG_DIR = "training/configs/"


class SemSegTrainer(BaseTrainer):
    # Spatial augmentation is applied to the image and its mask together, in
    # the dataset -- see semantic_segmentation/dataset.py. Doing it through the
    # image transform alone would move the image and leave the labels behind.
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
        super().__init__(training_folder, logger, project_id, training_params)
        self.device = get_device()

    def prepare_data(self):
        """Prepare the data for training"""
        subsets = ["train", "val", "test"]
        engine = db_manager.get_project_engine(self.project_id)

        with Session(engine) as session:
            data_items = session.query(DataItem).all()

            for i, item in enumerate(data_items):
                current_item = session.query(DataItem).get(item.id)
                if current_item is None:
                    self.logger.write(
                        f"Warning: DataItem with id {item.id} not found. Skipping."
                    )
                    continue

                image_path = current_item.path
                image_path = (
                    pathlib.Path(anylearning_config.PROJECTS_ROOT)
                    / str(self.project_id)
                    / "data"
                    / image_path
                )
                subset_path = self.data_folder / subsets[current_item.subset]
                subset_path.mkdir(parents=True, exist_ok=True)

                # Copy image
                shutil.copy(image_path, subset_path)

                # Save annotation
                annotation = current_item.annotation
                annotation_path = os.path.join(
                    subset_path,
                    re.sub(r"\.\w+$", ".json", os.path.basename(image_path)),
                )

                with open(annotation_path, "w") as f:
                    if annotation:
                        json.dump(annotation["data"], f)
                    else:
                        f.write("")

                if i % 50 == 0:
                    self.logger.write(
                        f"Exported data item {i + 1} of {len(data_items)}"
                    )

            with open(self.training_folder / "labels.json", "w") as f:
                json.dump(self.labels, f)

    def prepare_config(self):
        """Prepare and save the training configuration"""
        self.config_path = self.training_folder / "cfg_semantic_segmentation.yml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.training_folder / "labels.json", "r") as f:
            labels = json.load(f)

        base_config_path = resource_path(
            PACKAGE_NAME, os.path.join(CONFIG_DIR, "deeplabv3-semseg.yml")
        )
        if not os.path.exists(base_config_path):
            raise ValueError(f"Config file not found at {base_config_path}")

        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        model_arch = self.training_params.model_architecture
        config["model"]["arch"] = model_arch
        config["save_dir"] = str(self.output_folder)

        # id starts from 1 (0 is background)
        labels = [{**x, "id": i + 1} for i, x in enumerate(labels)]
        config["data"]["label_set"] = labels

        config["data"]["img_size"] = self.resolve_image_size(config["data"]["img_size"])

        config["data"]["augmentation"] = self.resolve_augmentation()
        config["data"]["train_dir"] = str(self.training_folder / "data" / "train")
        config["data"]["val_dir"] = str(self.training_folder / "data" / "val")
        config["data"]["test_dir"] = str(self.training_folder / "data" / "test")

        config["model"]["num_classes"] = len(labels)

        resume_from = self.resolve_pretrained_model_path()
        if resume_from:
            config["model"]["resume_from"] = resume_from

        config["training"]["batch_size"] = max(
            2, self.training_params.batch_size
        )  # batch size == 1 will raise BatchNorm error
        config["training"]["optim"]["lr"] = self.training_params.learning_rate
        config["training"]["epochs"] = self.training_params.epochs

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Return the content of the config file
        with open(self.config_path, "r") as f:
            return f.read()

    def train(self):
        print(
            "Training session id:",
            self.logger.training_session_id,
            "Project id:",
            self.logger.project_id,
        )
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
        # improves, so a run whose IoU never rises above 0 legitimately produces
        # last_model.pth alone. os.path.exists(None) raises TypeError, which used
        # to crash that fallback instead of taking it.
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
    def hex_to_rgb(hex_color: str) -> tuple:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in range(0, len(hex_color), 2))

    @staticmethod
    def run_inference(config_data: str, model_path: str, image: np.ndarray):
        """Run semantic segmentation inference on an image"""
        config = yaml.safe_load(config_data)
        label_set = config["data"]["label_set"]
        label_id_to_name = {v["id"]: v["name"] for v in label_set}
        label_id_to_color = {
            v["id"]: SemSegTrainer.hex_to_rgb(v["color"]) for v in label_set
        }

        model = load_torch_model(model_path)
        transform = get_transformations(
            config["data"]["img_size"],
            mean=config["data"]["normalize"]["mean"],
            std=config["data"]["normalize"]["std"],
            is_train=False,
        )
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
            pred_mask = model(inp)
            pred_mask = torch.argmax(pred_mask, dim=1)[0]

        # Visualize
        np_image = np.array(pil_image)
        pred_mask = pred_mask.cpu().numpy()
        pred_mask = pred_mask.astype(np.uint8)
        pred_mask = cv2.resize(
            pred_mask,
            (np_image.shape[1], np_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        visualization_image = np.zeros_like(np_image)
        for label_id, color in label_id_to_color.items():
            visualization_image[pred_mask == label_id] = color

        # Blend with original image
        alpha = 0.5
        visualization_image = cv2.addWeighted(
            np_image, alpha, visualization_image, 1 - alpha, 0
        )

        # formatted predictions, to polygon format
        formatted_predictions = []
        for label_id, name in label_id_to_name.items():
            mask = (pred_mask == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            all_polygons = []
            for contour in contours:
                if cv2.contourArea(contour) > 5:
                    all_polygons.append(contour.flatten().tolist())
            if all_polygons:
                formatted_predictions.append(
                    {
                        "id": label_id,
                        "label": name,
                        "color": label_id_to_color[label_id],
                        "points": all_polygons,
                        "type": "polygon",
                    }
                )

        return formatted_predictions, visualization_image

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
