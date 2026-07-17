from __future__ import annotations

import json
from typing import Any

ASPECTS = frozenset(
    {
        "ownership",
        "type_system",
        "safety",
        "performance",
        "learning_curve",
        "compile_time",
        "error_message",
        "debugging",
        "maintainability",
        "readability",
        "extensibility",
        "api_design",
        "package_manager",
        "libraries",
        "framework_support",
        "community",
    }
)
CLASSES = frozenset({"positive", "neutral", "negative"})


class AnnotationValidationError(ValueError):
    pass


def validate_annotation(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnnotationValidationError(f"不是合法 JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"annotations"}:
        raise AnnotationValidationError("根对象必须且只能包含 annotations")
    annotations = payload["annotations"]
    if not isinstance(annotations, list):
        raise AnnotationValidationError("annotations 必须为数组")
    seen: set[str] = set()
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(annotations):
        if not isinstance(item, dict) or set(item) != {"aspect", "class"}:
            raise AnnotationValidationError(f"annotations[{index}] 字段必须为 aspect 和 class")
        aspect = item["aspect"]
        sentiment_class = item["class"]
        if aspect not in ASPECTS:
            raise AnnotationValidationError(f"未知 aspect: {aspect}")
        if sentiment_class not in CLASSES:
            raise AnnotationValidationError(f"未知 class: {sentiment_class}")
        if aspect in seen:
            raise AnnotationValidationError(f"aspect 重复: {aspect}")
        seen.add(aspect)
        normalized.append({"aspect": aspect, "class": sentiment_class})
    return {"annotations": normalized}
