from unittest.mock import Mock, patch

import pytest
from PIL import Image

from anylearning.auto_labeling.inference_model import InferenceModel
from anylearning.auto_labeling.label_spaces import COCO_80, COCO_91
from anylearning.inference import (
    InferenceResult,
    InferenceShape,
    ModelCapabilities,
    ModelTask,
    Point,
    ShapeType,
)


class FakeSession:
    def __init__(self, task=ModelTask.DETECTION):
        self.capabilities = ModelCapabilities(
            model_id="test-detector",
            model_revision="sha256:test",
            tasks=(task,),
        )
        self.loaded = False
        self.unloaded = False
        self.request = None

    def load(self):
        self.loaded = True

    def predict(self, request, image):
        self.request = request
        assert image.size == (64, 48)
        return InferenceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            model_id=request.model_id,
            model_revision=request.model_revision,
            shapes=(
                InferenceShape(
                    type=ShapeType.RECTANGLE,
                    points=(Point(x=1.25, y=2.5), Point(x=40.75, y=30.5)),
                    label="dog",
                    score=0.95,
                    attributes={"class_id": 16},
                ),
            ),
        )

    def unload(self):
        self.unloaded = True


def detector_config(tmp_path):
    return {
        "type": "inference",
        "name": "test-detector",
        "display_name": "Test detector",
        "backend": "dfine_onnx",
        "config_file": str(tmp_path / "config.yaml"),
        "inference_config": {
            "name": "test-detector",
            "model_path": "model.onnx",
            "label_space": "coco80",
        },
    }


def test_coco_label_spaces_match_exported_tensor_layouts():
    assert len(COCO_80) == 80
    assert COCO_80[16] == "dog"
    assert len(COCO_91) == 91
    assert COCO_91[0] is None
    assert COCO_91[18] == "dog"


def test_generic_detector_uses_shared_session_and_project_class_filter(tmp_path):
    session = FakeSession()
    registry = Mock()
    registry.create_session.return_value = session
    with patch(
        "anylearning.auto_labeling.inference_model.get_default_registry",
        return_value=registry,
    ):
        model = InferenceModel(detector_config(tmp_path), None)

    result = model.predict_shapes(
        Image.new("RGB", (64, 48)),
        allowed_labels=("dog", "custom"),
        parameters={"confidence": 0.8},
    )

    assert session.loaded
    registry.create_session.assert_called_once()
    assert session.request.output_shape is ShapeType.RECTANGLE
    assert session.request.parameters == {"confidence": 0.8, "class_ids": (16,)}
    assert session.request.source_id.startswith("image-sha256:")
    assert len(result.shapes) == 1
    assert result.shapes[0].label == "dog"
    assert result.shapes[0].score == 0.95
    assert result.shapes[0].points[0].x == 1.25
    assert result.model_id == "test-detector"
    assert result.model_revision == "sha256:test"

    model.unload()
    assert session.unloaded


def test_detector_rejects_prompts_and_projects_without_matching_classes(tmp_path):
    session = FakeSession()
    registry = Mock()
    registry.create_session.return_value = session
    with patch(
        "anylearning.auto_labeling.inference_model.get_default_registry",
        return_value=registry,
    ):
        model = InferenceModel(detector_config(tmp_path), None)

    with pytest.raises(ValueError, match="no classes matching"):
        model.predict_shapes(Image.new("RGB", (64, 48)), allowed_labels=("helmet",))

    model.set_auto_labeling_marks([{"type": "point", "data": [10, 20], "label": 1}])
    with pytest.raises(ValueError, match="does not accept prompts"):
        model.predict_shapes(Image.new("RGB", (64, 48)), allowed_labels=("dog",))


@pytest.mark.parametrize("label", [-1, 2, "1", None])
def test_prompt_labels_are_strictly_binary(label):
    with pytest.raises(ValueError, match="must be 0 or 1"):
        InferenceModel._prompts([{"type": "point", "data": [10, 20], "label": label}])
