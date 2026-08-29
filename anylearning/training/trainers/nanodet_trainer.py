import numpy as np
import cv2
from anylearning.utils.resources import resource_path
import torch
import yaml

import json
import os
import pathlib
import shutil
import tempfile
import time
import traceback
from anylearning import config as anylearning_config
from anylearning.database import DataItem, TrainingParams, db_manager
from anylearning.training.device_utils import cuda_available, get_device
from anylearning.training.logging import NanoDetLogger, TrainingLogsWriter
from anylearning.training.models.nanodet.tools.train import TrainArgs
from anylearning.training.models.nanodet.tools.train import main as nanodet_train
from anylearning.training import augmentation
from anylearning.training.trainers.base_trainer import BaseTrainer
from anylearning.utils.converters import convert_anylearning_to_yolo
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.export_onnx import convert_onnx
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, load_model_weight
from sqlalchemy.orm import Session

# We start with a very lightweight NanoDet config
CONFIG_TEMPLATES = {
    "lightweight": resource_path("anylearning", "training/configs/nanodet-m-0.5x.yml"),
    "medium": resource_path("anylearning", "training/configs/nanodet-plus-m_416.yml"),
    "large": resource_path(
        "anylearning", "training/configs/nanodet-plus-m-1.5x_416.yml"
    ),
}


class Predictor:
    def __init__(self, cfg, model_path, device=None):
        self.cfg = cfg
        # Weights are always read onto the CPU below, then moved to the target
        # device, so a GPU-trained checkpoint also loads on a CPU-only machine.
        self.device = torch.device(device) if device is not None else get_device()
        model = build_model(cfg.model)
        ckpt = torch.load(
            model_path, map_location=lambda storage, loc: storage, weights_only=False
        )
        load_model_weight(model, ckpt, None)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(self.device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        with torch.no_grad():
            results = self.model.inference(meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = self.model.head.show_result(
            meta["raw_img"][0], dets, class_names, score_thres=score_thres
        )
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


class NanoDetTrainer(BaseTrainer):
    # NanoDet warps the boxes with the image, so these are safe. Its flip
    # matrix only ever mirrors horizontally, so there is no vertical flip to
    # offer.
    AUGMENTATIONS = (
        augmentation.HORIZONTAL_FLIP,
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

    def prepare_data(self):
        """
        Prepare the data for the training job.
        """
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
                image_path.parent.mkdir(parents=True, exist_ok=True)
                subset_path = self.data_folder / subsets[current_item.subset]
                subset_path.mkdir(parents=True, exist_ok=True)
                shutil.copy(image_path, subset_path)
                annotation = current_item.annotation
                image_filename = os.path.basename(image_path)
                annotation_filename = os.path.splitext(image_filename)[0] + ".txt"
                annotation_path = os.path.join(subset_path, annotation_filename)
                with open(annotation_path, "w") as f:
                    if not annotation:
                        f.write("")
                    else:
                        image = cv2.imread(str(image_path))
                        image_size = image.shape[1], image.shape[0]
                        f.write(
                            convert_anylearning_to_yolo(
                                annotation["data"], self.labels, image_size
                            )
                        )

                if i % 50 == 0:
                    self.logger.write(
                        f"Exported data item {i + 1} of {len(data_items)}"
                    )
            self.logger.write(f"Exported {len(data_items)} data items.")
            self.logger.write("Data exported successfully.")
            with open(self.training_folder / "labels.json", "w") as f:
                json.dump(self.labels, f)
            self.logger.write("Labels exported successfully.")

    def prepare_config(self):
        self.config_path = self.training_folder / "nanodet.yml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.training_folder / "labels.json", "r") as f:
            labels = json.load(f)
        sorted_labels = sorted(labels, key=lambda x: x["id"])
        class_names = [label["name"] for label in sorted_labels]

        base_config_path = CONFIG_TEMPLATES[self.training_params.model_size]
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        resume_from = self.resolve_pretrained_model_path()
        if resume_from:
            config["schedule"]["load_model"] = resume_from

        config["save_dir"] = str(self.output_folder)
        config["class_names"] = class_names
        config["data"]["train"]["class_names"] = class_names
        config["data"]["val"]["class_names"] = class_names

        config["data"]["train"]["img_path"] = str(self.data_folder / "train")
        config["data"]["val"]["img_path"] = str(self.data_folder / "val")

        config["data"]["train"]["ann_path"] = str(self.data_folder / "train")
        config["data"]["val"]["ann_path"] = str(self.data_folder / "val")

        config["device"]["batchsize_per_gpu"] = self.training_params.batch_size
        config["device"]["workers_per_gpu"] = max(1, os.cpu_count() // 2)
        config["schedule"]["optimizer"]["lr"] = self.training_params.learning_rate
        config["schedule"]["total_epochs"] = self.training_params.epochs

        image_size = self.resolve_image_size(config["data"]["train"]["input_size"][0])
        # NanoDet keeps its augmentation in the training pipeline, as
        # probabilities and ranges rather than switches. Only the train subset:
        # augmenting validation would make the metric measure a different task
        # from the one being trained.
        chosen = self.resolve_augmentation()
        pipeline = config["data"]["train"]["pipeline"]
        pipeline["flip"] = 0.5 if chosen["horizontal_flip"] else 0.0
        pipeline["rotation"] = chosen["rotation_degrees"]
        if chosen["color_jitter"]:
            pipeline["brightness"] = 0.2
            pipeline["contrast"] = [0.6, 1.4]
            pipeline["saturation"] = [0.5, 1.2]
        else:
            pipeline["brightness"] = 0.0
            pipeline["contrast"] = [1.0, 1.0]
            pipeline["saturation"] = [1.0, 1.0]
        for subset in ("train", "val"):
            config["data"][subset]["input_size"] = [image_size, image_size]

        # Which CUDA device, or -1 for "no CUDA device". Not which backend:
        # Metal has no id to put here, so NanoDet reads that from the
        # environment like the other four trainers do.
        #
        # -1 as an int rather than the string "-1". Both work -- yacs runs
        # literal_eval over every scalar it loads, so the string arrives as the
        # integer anyway -- but a value whose meaning survives only through a
        # coincidence in the config library is worth writing down correctly.
        if cuda_available():
            config["device"]["gpu_ids"] = [0]
        else:
            config["device"]["gpu_ids"] = -1

        if "aux_head" in config["model"]["arch"]:
            config["model"]["arch"]["aux_head"]["num_classes"] = len(class_names)
        config["model"]["arch"]["head"]["num_classes"] = len(class_names)

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Return the content of the config file
        config_content = yaml.dump(config)
        return config_content

    def train(self):
        # Run the training command as a subprocess
        train_configs = TrainArgs()
        train_configs.config = str(self.config_path)
        try:
            nanodet_logger = NanoDetLogger(
                writer=self.logger, save_dir=str(self.output_folder)
            )
            nanodet_train(train_configs, logger=nanodet_logger)
        except Exception as e:
            self.logger.write(
                f"Error during training: {str(e)} {traceback.format_exc()}"
            )
            raise RuntimeError(f"Training process failed due to Error: {str(e)}") from e

    def export_onnx(self):
        ret, model_path = self.get_model_path()
        if not ret:
            return None
        onnx_path = os.path.join(self.output_folder, "exported_model.onnx")
        convert_onnx(str(self.config_path), str(model_path), onnx_path)
        return onnx_path

    def get_model_path(self):
        # Empty strings, not None: model_best.ckpt is only written when validation
        # improves, so a run that never improves legitimately produces
        # model_last.ckpt alone. os.path.exists(None) raises TypeError, which used
        # to crash that fallback instead of taking it.
        model_best_path = ""
        model_last_path = ""

        for root, _, files in os.walk(self.output_folder):
            for file in files:
                if file == "model_best.ckpt":
                    model_best_path = os.path.join(root, file)
                elif file == "model_last.ckpt":
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
            visualization_image: np.ndarray, the visualized image
        """

        # Save config to a temp file
        cfg_file = tempfile.mktemp()
        with open(cfg_file, "w") as f:
            f.write(config_data)
        load_config(cfg, cfg_file)

        # Use the GPU when one is available, fall back to the CPU otherwise
        predictor = Predictor(cfg, str(model_path))

        # Perform inference and filter results by confidence score
        threshold = 0.35
        meta, raw_results = predictor.inference(image)
        results = []
        for class_id, class_results in raw_results[0].items():
            filtered_boxes = []
            for box in class_results:
                if box[-1] >= threshold:  # Filter boxes with score >= 0.05
                    filtered_boxes.append(box)
            if filtered_boxes:
                results.append({class_id: filtered_boxes})
        # Every class, merged into one mapping -- this used to be
        # `[dict(results[0])]`, which kept the first class and silently threw
        # the rest away. A helmet-and-jacket detector returned five jackets and
        # no helmets, from both the endpoint and the drawn visualisation, and
        # nothing in the log said a class had been dropped. Most detection
        # projects have more than one class, so most of them were affected.
        merged = {}
        for entry in results:
            merged.update(entry)
        results = [merged] if merged else [{}]

        # Visualize results
        visualization_image = predictor.visualize(
            results[0], meta, cfg.class_names, threshold
        )

        return results, visualization_image
