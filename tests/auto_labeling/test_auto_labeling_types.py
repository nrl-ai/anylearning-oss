from anylearning.auto_labeling.types import AutoLabelingMode, AutoLabelingResult


def test_auto_labeling_result_init():
    # Test with default replace=True
    result = AutoLabelingResult(shapes=[])
    assert result.shapes == []
    assert result.replace is True

    # Test with explicit replace=False
    result = AutoLabelingResult(shapes=[], replace=False)
    assert result.shapes == []
    assert result.replace is False

    # Test with some shapes
    shapes = ["shape1", "shape2"]
    result = AutoLabelingResult(shapes=shapes)
    assert result.shapes == shapes
    assert result.replace is True


def test_auto_labeling_mode_constants():
    assert AutoLabelingMode.OBJECT == "AUTOLABEL_OBJECT"
    assert AutoLabelingMode.ADD == "AUTOLABEL_ADD"
    assert AutoLabelingMode.REMOVE == "AUTOLABEL_REMOVE"
    assert AutoLabelingMode.POINT == "point"
    assert AutoLabelingMode.RECTANGLE == "rectangle"


def test_auto_labeling_mode_init():
    mode = AutoLabelingMode("AUTOLABEL_ADD", "point")
    assert mode.edit_mode == "AUTOLABEL_ADD"
    assert mode.shape_type == "point"


def test_auto_labeling_mode_get_default():
    default_mode = AutoLabelingMode.get_default_mode()
    assert isinstance(default_mode, AutoLabelingMode)
    assert default_mode.edit_mode == AutoLabelingMode.ADD
    assert default_mode.shape_type == AutoLabelingMode.POINT


def test_auto_labeling_mode_equality():
    mode1 = AutoLabelingMode("AUTOLABEL_ADD", "point")
    mode2 = AutoLabelingMode("AUTOLABEL_ADD", "point")
    mode3 = AutoLabelingMode("AUTOLABEL_REMOVE", "point")
    mode4 = AutoLabelingMode("AUTOLABEL_ADD", "rectangle")

    assert mode1 == mode2
    assert mode1 != mode3
    assert mode1 != mode4
    assert mode1 != "not a mode"


def test_auto_labeling_mode_none():
    assert AutoLabelingMode.NONE.edit_mode is None
    assert AutoLabelingMode.NONE.shape_type is None
