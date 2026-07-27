import json

import pytest

from llm_labeler.validation import AnnotationValidationError, validate_annotation


def test_valid_annotation_is_normalized():
    annotations = [
        {"aspect": "runtime_performance", "class": "positive"},
        {"aspect": "compile_time", "class": "negative"},
    ]
    raw = json.dumps({"annotations": annotations})
    assert validate_annotation(raw) == {"annotations": annotations}


def test_empty_annotations_is_valid():
    assert validate_annotation('{"annotations":[]}') == {"annotations": []}


@pytest.mark.parametrize(
    "payload",
    [
        {"annotations": [{"aspect": "speed", "class": "positive"}]},
        {"annotations": [{"aspect": "runtime_performance", "class": "mixed"}]},
        {"annotations": [{"aspect": "safety", "class": "not_mentioned"}]},
        {
            "annotations": [
                {"aspect": "runtime_performance", "class": "positive", "reason": "x"}
            ]
        },
        {"annotations": ["runtime_performance"]},
        {"labels": []},
    ],
)
def test_rejects_non_whitelisted_shapes(payload):
    with pytest.raises(AnnotationValidationError):
        validate_annotation(json.dumps(payload))


def test_rejects_duplicate_aspect():
    raw = json.dumps(
        {
            "annotations": [
                {"aspect": "safety", "class": "positive"},
                {"aspect": "safety", "class": "neutral"},
            ]
        }
    )
    with pytest.raises(AnnotationValidationError, match="重复"):
        validate_annotation(raw)
