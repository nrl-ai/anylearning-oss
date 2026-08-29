import cv2
import torch

import detectron2.data.transforms as T
from anylearning.training.device_utils import get_model_device, load_torch_model
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import ColorMode, Visualizer

from .train import set_model_config_and_weights


def rebuild_inference_config(config: dict):
    """The detectron2 config for a model whose pickled one is gone.

    Training pickles its config into the run's output folder, and that folder is
    deleted when the run finishes -- so every attempt to try an instance
    segmentation model answered 500 with

        No such file or directory: .../training/<n>/training_output/detectron2_cfg.pkl

    A run keeps a copy beside the model now, but models trained before that
    cannot get theirs back. Everything inference actually reads is in
    the run's stored YAML anyway -- the backbone, the class count and the image
    size -- so it is rebuilt from that rather than telling the user to train
    again.

    The thresholds match get_detectron2_config: a model behaves the same way
    whichever path produced its config, which is the whole point.
    """
    cfg = get_cfg()
    set_model_config_and_weights(config["model"]["backbone"], cfg)
    # No dataset is registered at inference time, so the Predictor must not go
    # looking for metadata under a name that does not exist.
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.INPUT.MAX_SIZE_TEST = config["data"]["img_size"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config["model"]["num_classes"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.MODEL.ROI_BOX_HEAD.NORM = "FrozenBN"
    cfg.MODEL.ROI_MASK_HEAD.NORM = "FrozenBN"
    cfg.MODEL.FPN.NORM = "FrozenBN"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def resolve_config_device(cfg):
    """Return the device named by a detectron2 config, downgraded when unusable.

    ``MODEL.DEVICE`` is pickled during training, so a config produced on a GPU
    machine still asks for CUDA when it is later loaded on a CPU-only one.

    "mps" is downgraded unconditionally, not merely when Metal is absent:
    detectron2's own kernels are CPU and CUDA, and a Mask R-CNN whose weights
    are on Metal ends in a Metal internal error rather than a wrong answer. No
    config we write should say mps -- this catches one that arrived from
    somewhere else.
    """
    device = str(cfg.MODEL.DEVICE)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if device.startswith("mps"):
        return "cpu"
    return device


class Predictor:
    def __init__(self, cfg, model):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = model
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # Follow the device the weights were actually loaded on: the config was
            # pickled during training and may name a device this machine lacks.
            image = image.to(get_model_device(self.model))

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions


def inference_fn(detectron2_cfg, model_path, class_names, image):
    model = load_torch_model(model_path, device=resolve_config_device(detectron2_cfg))

    # create a metadata to map the class id to the class name
    id_map = {i + 1: i for i in range(len(class_names))}
    metadata = Metadata.set(
        dict(thing_classes=class_names, thing_dataset_id_to_contiguous_id=id_map)
    )

    predictor = Predictor(detectron2_cfg, model)

    image_bgr = image[:, :, ::-1]
    outputs = predictor(image_bgr)
    instances = outputs["instances"]

    v = Visualizer(
        image,
        metadata=metadata,
        font_size_scale=2.0,
        instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )

    out_pred = v.draw_instance_predictions(instances.to("cpu"))
    formatted_predictions = dict()
    formatted_predictions["boxes"] = instances.pred_boxes.tensor.cpu().numpy().tolist()
    formatted_predictions["scores"] = instances.scores.cpu().numpy().tolist()
    formatted_predictions["labels"] = [
        class_names[i] for i in instances.pred_classes.cpu().numpy().tolist()
    ]
    formatted_predictions["masks"] = []
    # convert masks to polygons
    for mask in instances.pred_masks.cpu().numpy().astype("uint8"):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        formatted_predictions["masks"].append([contour.flatten().tolist() for contour in contours])
    visualization_image = out_pred.get_image()

    return formatted_predictions, visualization_image
