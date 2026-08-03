from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ASPECTS = frozenset(
    {
        "ownership",
        "type_system",
        "safety",
        "runtime_performance",
        "learning_curve",
        "compile_time",
        "diagnostics_debugging",
        "tooling_documentation",
        "readability_maintainability",
        "api_extensibility",
        "package_manager",
        "libraries_frameworks",
        "community",
    }
)
CLASSES = frozenset({"negative", "neutral", "positive"})


class AnnotationValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BatchAnnotation:
    raw_response: str | None
    parsed_result: dict[str, Any] | None
    error_message: str | None


def _validate_annotation_payload(payload: Any) -> dict[str, Any]:
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


def validate_annotation(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnnotationValidationError(f"不是合法 JSON: {exc.msg}") from exc
    return _validate_annotation_payload(payload)


def validate_batch_annotations(
    raw: str,
    expected_corpus_ids: list[int],
) -> dict[int, BatchAnnotation]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnnotationValidationError(f"不是合法 JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"results"}:
        raise AnnotationValidationError("批量响应根对象必须且只能包含 results")
    if not isinstance(payload["results"], list):
        raise AnnotationValidationError("results 必须为数组")

    expected = set(expected_corpus_ids)
    items_by_id: dict[int, Any] = {}
    duplicate_ids: set[int] = set()
    for item in payload["results"]:
        if not isinstance(item, dict) or set(item) != {"corpus_id", "annotations"}:
            continue
        corpus_id = item["corpus_id"]
        if type(corpus_id) is not int or corpus_id not in expected:
            continue
        if corpus_id in items_by_id:
            duplicate_ids.add(corpus_id)
            continue
        items_by_id[corpus_id] = item

    results: dict[int, BatchAnnotation] = {}
    for corpus_id in expected_corpus_ids:
        item = items_by_id.get(corpus_id)
        if item is None:
            results[corpus_id] = BatchAnnotation(
                raw_response=raw,
                parsed_result=None,
                error_message=f"批量响应缺少 corpus_id={corpus_id}",
            )
            continue
        item_raw = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        if corpus_id in duplicate_ids:
            results[corpus_id] = BatchAnnotation(
                raw_response=item_raw,
                parsed_result=None,
                error_message=f"批量响应中 corpus_id={corpus_id} 重复",
            )
            continue
        try:
            parsed = _validate_annotation_payload({"annotations": item["annotations"]})
        except AnnotationValidationError as exc:
            results[corpus_id] = BatchAnnotation(
                raw_response=item_raw,
                parsed_result=None,
                error_message=str(exc),
            )
        else:
            results[corpus_id] = BatchAnnotation(
                raw_response=item_raw,
                parsed_result=parsed,
                error_message=None,
            )
    return results
