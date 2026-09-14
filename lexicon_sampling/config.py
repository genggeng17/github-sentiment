"""解析配额，合并公共参数与各方面覆盖参数，并验证筛选条件。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from llm_labeler.validation import ASPECTS

SOURCE_TYPES = {"issue", "pull_request", "issue_comment", "pr_issue_comment", "pr_review_comment"}


def parse_quotas(values: list[str]) -> dict[str, int]:
    quotas = {}
    for value in values:
        aspect, separator, count = value.partition("=")
        if not separator or aspect not in ASPECTS or aspect in quotas:
            raise ValueError(f"无效或重复的方面配额: {value}；格式为 aspect=条数")
        try:
            number = int(count)
        except ValueError as exc:
            raise ValueError(f"配额必须是正整数: {value}") from exc
        if number <= 0:
            raise ValueError(f"配额必须是正整数: {value}")
        quotas[aspect] = number
    if not quotas:
        raise ValueError("至少指定一个方面配额")
    return quotas


def _positive(value: Any, name: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


@dataclass(frozen=True)
class AspectFilters:
    """一个方面的有效筛选条件；词典内的配置优先于命令行公共参数。"""

    min_target_chars: int = 1
    max_model_input_chars: int | None = 4000
    per_repository_limit: int | None = None
    source_types: list[str] | None = None
    repository_names: list[str] | None = None

    def with_overrides(self, aspect: str, overrides: dict[str, Any]) -> AspectFilters:
        values = asdict(self)
        if not isinstance(overrides, dict) or set(overrides) - set(values):
            raise ValueError(f"{aspect} 包含未知筛选参数")
        values.update(overrides)
        resolved = AspectFilters(**values)
        resolved.validate(aspect)
        return resolved

    def validate(self, aspect: str) -> None:
        _positive(self.min_target_chars, "min_target_chars")
        _positive(self.max_model_input_chars, "max_model_input_chars", nullable=True)
        _positive(self.per_repository_limit, "per_repository_limit", nullable=True)
        if self.source_types is not None and (
            not isinstance(self.source_types, list)
            or not self.source_types
            or any(source not in SOURCE_TYPES for source in self.source_types)
        ):
            raise ValueError(f"{aspect} 的 source_types 无效")
        if self.repository_names is not None and (
            not isinstance(self.repository_names, list)
            or not self.repository_names
            or any(not isinstance(name, str) for name in self.repository_names)
        ):
            raise ValueError(f"{aspect} 的 repository_names 无效")

    def accepts(self, row: dict[str, Any], repository_name: str) -> bool:
        """只检查长度、来源、仓库；词典是否命中由 matcher 决定。"""
        if len(row["clean_text"].strip()) < self.min_target_chars:
            return False
        if self.max_model_input_chars and len(row["model_input"]) > self.max_model_input_chars:
            return False
        if self.source_types and row["source_type"] not in self.source_types:
            return False
        return not self.repository_names or repository_name in self.repository_names


def validate_request(name: str, seed: str, quotas: dict[str, int]) -> None:
    if not name.strip() or len(name) > 100 or len(str(seed)) > 100:
        raise ValueError("集合名和种子不能为空或超过100字符")
    if not quotas or any(aspect not in ASPECTS for aspect in quotas):
        raise ValueError("必须指定有效方面配额")
    for aspect, quota in quotas.items():
        _positive(quota, aspect)


def resolve_filters(
    definition: dict[str, Any],
    quotas: dict[str, int],
    defaults: AspectFilters,
    repositories: dict[int, str],
) -> dict[str, AspectFilters]:
    """把每方面的最终参数算好，扫描时不再反复合并配置。"""
    filters = {}
    for aspect in quotas:
        overrides = definition["aspects"][aspect].get("filters", {})
        resolved = defaults.with_overrides(aspect, overrides)
        if resolved.repository_names and set(resolved.repository_names) - set(
            repositories.values()
        ):
            raise ValueError(f"{aspect} 指定了不存在或未启用的仓库")
        filters[aspect] = resolved
    return filters


def scan_character_limit(filters: dict[str, AspectFilters]) -> int | None:
    """数据库扫描取最宽松的上限，各方面再按自己的条件精筛。"""
    ceilings = [conditions.max_model_input_chars for conditions in filters.values()]
    return None if None in ceilings else max(ceilings)


def configuration_snapshot(
    definition: dict[str, Any],
    quotas: dict[str, int],
    filters: dict[str, AspectFilters],
    repositories: dict[int, str],
    matcher_version: str,
) -> dict[str, Any]:
    """冻结实际使用的配置，保证同名集合不能悄悄换词典或配额。"""
    encoded = json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "method": "lexicon",
        "matcher_version": matcher_version,
        "lexicon": definition,
        "lexicon_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "quotas": quotas,
        "filters": {aspect: asdict(values) for aspect, values in filters.items()},
        # 只保存稳定身份，不把仓库更新时间等无关字段纳入配置比较。
        "repository_snapshot": [
            {"id": key, "full_name": value} for key, value in sorted(repositories.items())
        ],
        "overlap_policy": "shared_candidates_unique_union",
    }
