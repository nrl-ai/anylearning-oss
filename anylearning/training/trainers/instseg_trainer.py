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
from detectron2.export import TracingAdapter
from PIL import Image
from sqlalchemy.orm import Session

from anylearning import config as anylearning_config
from anylearning.database import DataItem, TrainingParams, db_manager
from anylearning.training import augmentation
from anylearning.training.device_utils import (
    get_model_device,
    load_torch_model_for_export,
)
from anylearning.training.logging import TrainingLogsWriter
from anylearning.training.models.instance_segmentation.factory import (
    InstanceSegmentationModelFactory,
)
from anylearning.training.trainers.base_trainer import BaseTrainer
from anylearning.utils.resources import resource_path

PACKAGE_NAME = "anylearning"
CONFIG_DIR = "training/configs/"


class AnnotationStore:
    def __init__(self, categories: list):
        self.images = []
        self.annotations = []
        self.categories = categories

        self.cate_name_to_id = {x["name"]: x["id"] for x in categories}

        self.img_path_to_id = {}

    def add_annotation(self, img_path, ann_dict):
        # get im width and height without loading image
        image_name = os.path.basename(img_path)
        if img_path not in self.img_path_to_id:
            self.img_path_to_id[img_path] = len(self.img_path_to_id) + 1
            # add image if not already added
            image_id = self.img_path_to_id[img_path]
            width, height = Image.open(img_path).size
            self.images.append(
                {
                    "id": image_id,
                    "file_name": image_name,
                    "height": height,
                    "width": width,
                }
            )

        # add annotations
        if ann_dict is not None:
            for ann in ann_dict["data"]:
                # float32, not whatever the JSON happened to decode to.
                # cv2.contourArea accepts only CV_32F or CV_32S, so a polygon
                # with one fractional coordinate -- which the canvas can
                # produce -- aborts the whole run with
                #   (-215:Assertion failed) npoints >= 0 && (depth == CV_32F ...
                points = np.array(ann["points"], dtype=np.float32)
                area = cv2.contourArea(points)
                bbox = cv2.boundingRect(points)
                if area == 0:  # not a valid polygon
                    continue
                self.annotations.append(
                    {
                        "id": len(self.annotations) + 1,
                        "image_id": image_id,
                        "category_id": self.cate_name_to_id[ann["categories"][0]],
                        "segmentation": [points.flatten().tolist()],
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0,
                    }
                )

    def to_coco_format(self):
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }


class InstSegTrainer(BaseTrainer):
    # detectron2 builds its own augmentation list; flipping is the part it
    # exposes as configuration, and it carries boxes and masks along with it.
    # Rotation would need a custom dataset mapper, so it is not offered rather
    # than accepted and ignored.
    AUGMENTATIONS = (
        augmentation.HORIZONTAL_FLIP,
        augmentation.VERTICAL_FLIP,
    )

    def __init__(
        self,
        training_folder: str,
        logger: TrainingLogsWriter,
        project_id: int,
        training_params: TrainingParams,
    ):
        super().__init__(training_folder, logger, project_id, training_params)

    def prepare_data(self):
        """Prepare the data for training"""
        subsets = ["train", "val", "test"]
        engine = db_manager.get_project_engine(self.project_id)

        categories = [
            {"id": i + 1, "name": x["name"]} for i, x in enumerate(self.labels)
        ]
        # annotations for each set
        annotation_stores = {subset: AnnotationStore(categories) for subset in subsets}

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
                os.path.join(
                    subset_path,
                    re.sub(r"\.\w+$", ".json", os.path.basename(image_path)),
                )
                # print(image_path)
                # print(annotation)
                annotation_stores[subsets[current_item.subset]].add_annotation(
                    image_path, annotation
                )

                if i % 50 == 0:
                    self.logger.write(
                        f"Exported data item {i + 1} of {len(data_items)}"
                    )

            for subset in subsets:
                with open(
                    self.training_folder / f"coco_{subset}_annotations.json", "w"
                ) as f:
                    json.dump(annotation_stores[subset].to_coco_format(), f)

            # save labels
            with open(self.training_folder / "labels.json", "w") as f:
                json.dump(self.labels, f)

    def prepare_config(self):
        """Prepare and save the training configuration"""
        self.config_path = self.training_folder / "cfg_instance_segmentation.yml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.training_folder / "labels.json", "r") as f:
            labels = json.load(f)

        base_config_path = resource_path(
            PACKAGE_NAME, os.path.join(CONFIG_DIR, "maskrcnn-instseg.yml")
        )
        if not os.path.exists(base_config_path):
            raise ValueError(f"Config file not found at {base_config_path}")

        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        # "maskrcnn-resnet50" -- architecture and backbone in one field, which
        # is what config.MODEL_VARIANTS offers. A value without the backbone
        # half used to raise IndexError several lines into prepare_config,
        # after the whole dataset had been exported, and the run ended with
        # "list index out of range" -- which names neither the setting nor the
        # value that was wrong.
        model_arch_and_backbone = self.training_params.model_architecture or ""
        architecture, _, backbone = model_arch_and_backbone.partition("-")
        if not backbone:
            raise ValueError(
                f"Model architecture {model_arch_and_backbone!r} does not name a "
                "backbone. Instance segmentation expects something like "
                "'maskrcnn-resnet50'; see config.MODEL_VARIANTS."
            )
        config["model"]["arch"] = architecture
        config["model"]["backbone"] = backbone
        config["save_dir"] = str(self.output_folder)

        # id starts from 1 (0 is background)
        labels = [{**x, "id": i + 1} for i, x in enumerate(labels)]
        config["data"]["label_set"] = labels

        config["data"]["img_size"] = self.resolve_image_size(config["data"]["img_size"])

        config["data"]["augmentation"] = self.resolve_augmentation()
        config["data"]["train_dir"] = str(self.training_folder / "data" / "train")
        config["data"]["val_dir"] = str(self.training_folder / "data" / "val")
        config["data"]["test_dir"] = str(self.training_folder / "data" / "test")

        config["data"]["train_ann_file"] = str(
            self.training_folder / "coco_train_annotations.json"
        )
        config["data"]["val_ann_file"] = str(
            self.training_folder / "coco_val_annotations.json"
        )
        config["data"]["test_ann_file"] = str(
            self.training_folder / "coco_test_annotations.json"
        )

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
        print(
            "Training session id:",
            self.logger.training_session_id,
            "Project id:",
            self.logger.project_id,
        )
        try:
            factory = InstanceSegmentationModelFactory()
            factory.train(self.config_path, self.logger)
        except Exception as e:
            self.logger.write(
                f"Error during training: {str(e)} {traceback.format_exc()}"
            )
            raise RuntimeError(f"Training process failed due to Error: {str(e)}") from e

    def get_model_path(self):
        print("Getting model path")
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
        config_data = yaml.safe_load(config_data)
        try:
            factory = InstanceSegmentationModelFactory()
            formatted_predictions, visualization_image = factory.inference(
                config_data, model_path, image
            )
        except Exception as e:
            raise RuntimeError(f"Training process failed due to Error: {str(e)}") from e

        return formatted_predictions, visualization_image

    def companion_files(self) -> list:
        """The pickled detectron2 config, which inference cannot work without.

        `factory.load_inference_config` can rebuild an equivalent one from the
        stored YAML, and does for models trained before this existed -- but the
        real thing is exact, so it is kept.
        """
        return [os.path.join(self.output_folder, "detectron2_cfg.pkl")]

    def export_onnx(self):
        ret, model_path = self.get_model_path()
        if not ret:
            return None
        onnx_path = os.path.join(self.output_folder, "exported_model.onnx")

        # load model
        torch_model = load_torch_model_for_export(model_path)

        im_torch = (
            torch.rand(3, 512, 512).float().to(get_model_device(torch_model))
        )  # random image with some shape
        inputs = [{"image": im_torch}]

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

        traceable_model = TracingAdapter(torch_model, inputs, inference)
        torch.onnx.export(
            traceable_model,
            (im_torch,),
            onnx_path,
            opset_version=16,
            input_names=["image"],
            output_names=["instances"],
            dynamic_axes={
                "image": {1: "height", 2: "width"},
                "instances": {0: "batch_size"},
            },
            # Stay on the TorchScript tracing exporter. torch 2.6 made dynamo
            # the default, and it cannot capture detectron2's TracingAdapter:
            # it rejects the call with
            #   inputs[0] is a <class 'tuple'>, but dynamic_shapes[0] is a
            #   <class 'dict'>
            # because it converts the name-keyed dynamic_axes above into a
            # structure that no longer lines up with the positional args.
            # Mask R-CNN's data-dependent control flow is what tracing exists
            # for, and detectron2 exports the same way upstream.
            #
            # This matters more than a warning: export runs *after* training,
            # and the model is only registered once it succeeds -- so a failure
            # here throws away a finished run.
            dynamo=False,
        )
        return onnx_path
