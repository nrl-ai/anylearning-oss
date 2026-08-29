import json
import os
import pathlib
import shutil
import traceback

import cv2
import numpy as np
import onnx
import torch
import yaml
from handpose.drawing import draw_hand_landmarks
from handpose.models.mlp import MLP
from sqlalchemy.orm import Session

from anylearning import config as anylearning_config
from anylearning.database import DataItem, TrainingParams, db_manager
from anylearning.training import handpose_landmarks
from anylearning.training.device_utils import get_model_device
from anylearning.training.logging import MLPLogger, TrainingLogsWriter
from anylearning.training.models.handpose.handpose.tools.train import train
from anylearning.training.models.handpose.handpose.utils import normalize_landmarks
from anylearning.training.trainers.base_trainer import BaseTrainer
from anylearning.utils.resources import resource_path

HANDPOSE_CONFIG_TEMPLATE = {
    "lightweight": resource_path(
        "anylearning", "training/configs/mlp-s.yaml"
    ),  # Running on all machine
    "medium": resource_path(
        "anylearning", "training/configs/mlp-m.yaml"
    ),  # Running on good CPU: i7, i9 or M
    "large": resource_path(
        "anylearning", "training/configs/mlp-l.yaml"
    ),  # Running on GPU
}


class Predictor:
    def __init__(self, cfg, model, image, device="cpu"):
        self.cfg = cfg
        self.model = model
        self.image = image
        self.device = device

    def predict(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # In a child process: mediapipe aborts rather than raising on some
        # machines, and this runs inside the API process, so a prediction must
        # not be able to take the application down. See handpose_landmarks.
        hands = handpose_landmarks.detect(np.array(img))
        if hands:
            landmarks_list = [
                [point.x, point.y, point.z] for point in hands[0].landmarks
            ]
            # From the *stored* config, so a model trained before this existed
            # is still fed what it was trained on. Getting this wrong does not
            # fail, it just predicts nonsense.
            if bool((self.cfg.get("data") or {}).get("normalize_landmarks", False)):
                landmarks_list = normalize_landmarks(landmarks_list)
        else:
            return [
                {"prediction": "Cannot classify handpose because no landmarks detected"}
            ], None

        landmarks_tensor = torch.Tensor(landmarks_list)
        landmarks_tensor = landmarks_tensor.reshape(1, -1).to(
            get_model_device(self.model)
        )

        with torch.no_grad():
            output = self.model(landmarks_tensor)[0]
        predicted = torch.argmax(output).item()
        label_str = self.cfg["class_names"][predicted]

        return [{"prediction": label_str}], hands

    def visualize(self, image, hands):
        MARGIN = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

        annotated_image = np.copy(image)

        for hand in hands:
            hand_landmarks = hand.landmarks

            # Draw the hand landmarks. mediapipe 1.0 removed the `solutions`
            # package (drawing_utils / drawing_styles / hands) along with the
            # landmark protobuf round-trip this used to need; detection already
            # uses the Tasks API, so only the drawing moved in-house.
            draw_hand_landmarks(annotated_image, hand_landmarks)

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(
                annotated_image,
                f"{hand.handedness}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

        return annotated_image


def _has_landmarks(annotation) -> bool:
    """True when the annotation carries the 21 landmarks the MLP expects.

    HandPoseDataset reads annotation["data"]["landmarks"] and indexes it by
    "0".."20". Anything else -- an empty list, a missing key -- is a record it
    cannot use, and it fails with a TypeError that names neither the file nor
    the reason.
    """
    if not isinstance(annotation, dict):
        return False
    data = annotation.get("data")
    if not isinstance(data, dict):
        return False
    landmarks = data.get("landmarks")
    return isinstance(landmarks, dict) and len(landmarks) > 0


class HandposeClassificationTrainer(BaseTrainer):
    # Trains on hand landmark vectors, not on pixels: flipping or jittering the
    # image it came from would not change a single input value.
    AUGMENTATIONS = ()

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
            skipped: list[str] = []

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

                annotation = current_item.annotation
                image_filename = os.path.basename(image_path)
                annotation_filename = os.path.splitext(image_filename)[0] + ".json"
                annotation_path = os.path.join(subset_path, annotation_filename)
                if item.labeled != 0:
                    # An item can be flagged as labelled while carrying no
                    # landmarks -- MediaPipe finds no hand in the image and the
                    # annotation is stored as {"data": []}. Exporting that
                    # produces a file HandPoseDataset then crashes on, with
                    # "list indices must be integers or slices, not str" and no
                    # indication of which image is at fault. One such item in
                    # 1,329 failed a whole training run.
                    if not _has_landmarks(annotation):
                        skipped.append(image_filename)
                        continue
                    shutil.copy(image_path, subset_path)
                    with open(annotation_path, "w") as f:
                        json.dump(annotation, f)

                if i % 50 == 0:
                    self.logger.write(
                        f"Exported data item {i + 1} of {len(data_items)}"
                    )
            self.logger.write(f"Exported {len(data_items) - len(skipped)} data items.")
            if skipped:
                shown = ", ".join(skipped[:5])
                more = f" and {len(skipped) - 5} more" if len(skipped) > 5 else ""
                self.logger.write(
                    f"Skipped {len(skipped)} labelled item(s) with no hand landmarks: "
                    f"{shown}{more}. Re-label or remove them to include them."
                )
            self.logger.write("Data exported successfully.")
            with open(self.training_folder / "labels.json", "w") as f:
                json.dump(self.labels, f)
            self.logger.write("Labels exported successfully.")

    def prepare_config(self):
        self.config_path = self.training_folder / "mlp.yml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.training_folder / "labels.json", "r") as f:
            labels = json.load(f)
        sorted_labels = sorted(labels, key=lambda x: x["id"])
        class_names = [label["name"] for label in sorted_labels]

        base_config_path = HANDPOSE_CONFIG_TEMPLATE[self.training_params.model_size]
        with open(base_config_path, "r") as f:
            config = yaml.safe_load(f)

        resume_from = self.resolve_pretrained_model_path()
        if resume_from:
            config["schedule"]["load_model"] = resume_from

        # Prepare save directory
        config["save_dir"] = str(self.output_folder)

        # Prepare data
        config["class_names"] = class_names
        config["data"]["train"]["class_names"] = class_names
        config["data"]["val"]["class_names"] = class_names

        config["data"]["train"]["annotation_path"] = str(self.data_folder / "train")
        config["data"]["val"]["annotation_path"] = str(self.data_folder / "val")
        config["data"]["train"]["batch_size"] = self.training_params.batch_size
        config["data"]["val"]["batch_size"] = self.training_params.batch_size
        config["data"]["test"]["batch_size"] = self.training_params.batch_size

        # Prepare training parameters
        config["models"]["arch"]["head"]["output_units"] = len(class_names)
        # Landmarks relative to the hand rather than to the image: 77.1% ->
        # 84.0% on the 26-letter ASL set. Written into the run's own config,
        # which is what inference reads back, so models trained before this
        # keep being fed raw coordinates.
        config["data"]["normalize_landmarks"] = True
        config["schedule"]["optimizer"]["lr"] = self.training_params.learning_rate
        config["schedule"]["epochs"] = self.training_params.epochs
        # The cosine schedule has to end when the run does. T_max stayed at the
        # template's 300 whatever the user asked for, so a 100-epoch run
        # finished a third of the way down the curve, still at a learning rate
        # meant for the middle of training -- and a run longer than 300 epochs
        # came back *up* the other side of the cosine, unlearning as it went.
        config["schedule"]["lr_schedule"]["T_max"] = max(
            1, int(self.training_params.epochs)
        )

        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Return the content of the config file
        with open(self.config_path, "r") as f:
            return f.read()

    def train(self):
        try:
            mlp_logger = MLPLogger(writer=self.logger, save_dir=str(self.output_folder))
            train(str(self.config_path), logger=mlp_logger)
        except Exception as e:
            self.logger.write(
                f"Error during training: {str(e)} {traceback.format_exc()}"
            )
            raise RuntimeError(f"Training process failed due to Error: {str(e)}") from e

    def export_onnx(self):
        with open(str(self.config_path), "r") as f:
            config = yaml.safe_load(f)

        ret, model_path = self.get_model_path()
        if not ret:
            return None
        onnx_path = os.path.join(self.output_folder, "exported_model.onnx")

        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )
        model_state_dict = checkpoint["model_state_dict"]

        # Init model
        model = MLP(config)
        model.load_state_dict(model_state_dict)
        model.eval()

        # Create dummy input
        dummy_input = torch.autograd.Variable(torch.randn(1, 63))

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            # TorchScript exporter, not dynamo -- dynamo needs onnxscript, which
            # reads function source and so cannot work in a compiled binary.
            dynamo=False,
        )

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"Model exported to ONNX successfully: {onnx_path}")

        return onnx_path

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
        config = yaml.safe_load(config_data)
        checkpoint = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=False
        )

        model = MLP(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        predictor = Predictor(config, model, image)

        prediction, hands = predictor.predict()
        if hands:
            visualized_image = predictor.visualize(image, hands)
            return prediction, visualized_image
        else:
            return prediction, image

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
