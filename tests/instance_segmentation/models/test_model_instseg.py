import numpy as np
import pytest
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling.backbone import resnet
from detectron2.structures import Boxes, Instances, PolygonMasks
from detectron2.utils.events import EventStorage

from anylearning.training.models.instance_segmentation.maskrcnn.inference import (
    Predictor,
)
from anylearning.training.models.instance_segmentation.maskrcnn.train import (
    compute_loss,
    create_model,
)


@pytest.fixture
def sample_detectron2_config():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cpu"
    return cfg


def test_model(sample_detectron2_config):
    model = create_model(sample_detectron2_config)
    # check if model is Resnet50
    assert isinstance(model.backbone.bottom_up, resnet.ResNet)
    assert len(model.backbone.bottom_up.res5) == 3

    # forward pass
    model.train()
    sample_data = [
        {
            "image": torch.tensor(
                [[[0, 1, 2, 3, 0, 1, 2], [1, 2, 0, 1, 2, 3, 0], [2, 3, 1, 0, 1, 2, 3]]],
                dtype=torch.uint8,
            ),
            "instances": Instances(
                image_size=(224, 224),
                gt_boxes=Boxes(torch.tensor([[366.0, 226.0, 382.0, 243.0]])),
                gt_classes=torch.tensor([0]),
                gt_masks=PolygonMasks(
                    [[[366, 226, 382, 226, 382, 243, 366, 243]]]
                ),  # Sample polygon
            ),
        },
        {
            "image": torch.tensor(
                [[[0, 1, 2, 3, 0, 1, 2], [1, 2, 0, 1, 2, 3, 0], [2, 3, 1, 0, 1, 2, 3]]],
                dtype=torch.uint8,
            ),
            "instances": Instances(
                image_size=(224, 224),
                gt_boxes=Boxes(
                    torch.tensor(
                        [[305.0, 264.0, 353.0, 288.0], [355.0, 280.0, 367.0, 283.0]]
                    )
                ),
                gt_classes=torch.tensor([0, 0]),
                gt_masks=PolygonMasks(
                    [
                        [
                            [305, 264, 353, 264, 353, 288, 305, 288]
                        ],  # First instance polygon
                        [
                            [355, 280, 367, 280, 367, 283, 355, 283]
                        ],  # Second instance polygon
                    ]
                ),
            ),
        },
    ]
    # EventStorage is entered for its side effect -- detectron2's loss path writes
    # to the ambient storage -- so the handle itself is unused.
    with EventStorage(0):
        train_total_loss, train_detail_loss = compute_loss(model, sample_data)
    print(train_total_loss, train_detail_loss)
    assert train_total_loss.item() > 0
    print(train_detail_loss)

    # Check if specific keys are in train_detail_loss
    expected_keys = {
        "loss_cls",
        "loss_box_reg",
        "loss_mask",
        "loss_rpn_cls",
        "loss_rpn_loc",
    }
    assert expected_keys.issubset(train_detail_loss.keys()), (
        "Missing keys in train_detail_loss"
    )


def test_model_inference(sample_detectron2_config):
    model = create_model(sample_detectron2_config)
    predictor = Predictor(sample_detectron2_config, model)
    input = np.random.random((224, 224, 3))
    output = predictor(input)
    assert output["instances"] is not None
