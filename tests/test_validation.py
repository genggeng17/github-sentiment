import json

import pytest

from llm_labeler.validation import AnnotationValidationError, validate_annotation


def test_valid_annotation_is_normalized():
    raw = json.dumps({"annotations": [{"aspect": "performance", "class": "positive"}]})
    assert validate_annotation(raw) == {
        "annotations": [{"aspect": "performance", "class": "positive"}]
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"annotations": [{"aspect": "speed", "class": "positive"}]},
        {"annotations": [{"aspect": "performance", "class": "mixed"}]},
        {"annotations": [{"aspect": "performance", "class": "positive", "reason": "x"}]},
        {"annotations": ["performance"]},
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
