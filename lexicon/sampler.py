"""词典集主流程：准备配置 → 扫描筛选 → 合并去重 → 保存集合。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from storage import Storage
from storage.models import utcnow

from .config import (
    AspectFilters,
    configuration_snapshot,
    resolve_filters,
    scan_character_limit,
    validate_request,
)
from .matcher import MATCHER_VERSION, LexiconMatcher
from .selection import AspectQuotaSelector, Candidate, SelectionResult, merge_selections

logger = logging.getLogger(__name__)


class LexiconSampler:
    """只组织流程；不在这里解释正则或实现堆排序，不直接执行 SQL。"""

    def __init__(self, storage: Storage):
        self.storage = storage

    def build(
        self,
        name: str,
        *,
        lexicon_path: str,
        quotas: dict[str, int],
        cleaning_version: str,
        seed: str = "0",
        max_model_input_chars: int | None = 4000,
        min_target_chars: int = 1,
        per_repository_limit: int | None = None,
        source_types: list[str] | None = None,
        repository_names: list[str] | None = None,
    ) -> dict[str, Any]:
        # 1. 合并并验证参数。所有配置错误都在创建集合前发现。
        validate_request(name, seed, quotas)
        if cleaning_version not in self.storage.available_cleaning_versions():
            raise ValueError(f"数据库中不存在清洗版本 {cleaning_version!r}")
        definition = json.loads(Path(lexicon_path).read_text(encoding="utf-8-sig"))
        matcher = LexiconMatcher(definition, quotas)
        repositories = {
            row["id"]: row["full_name"] for row in self.storage.list_repositories(enabled_only=True)
        }
        defaults = AspectFilters(
            min_target_chars=min_target_chars,
            max_model_input_chars=max_model_input_chars,
            per_repository_limit=per_repository_limit,
            source_types=source_types,
            repository_names=repository_names,
        )
        filters = resolve_filters(definition, quotas, defaults, repositories)
        config = configuration_snapshot(definition, quotas, filters, repositories, MATCHER_VERSION)
        scan_limit = scan_character_limit(filters)

        # 2. 同名且已完成的集合直接复用；不重新随机选择成员。
        sample_id, completed = self.storage.prepare_sample_set(
            name.strip(),
            sum(quotas.values()),
            str(seed),
            cleaning_version,
            scan_limit,
            selection_config=config,
        )
        if completed is not None:
            return {**completed, "sample_set_id": sample_id, "reused": True}

        stats = self._initial_stats(sample_id, name, seed, cleaning_version, config)
        selectors = {
            aspect: AspectQuotaSelector(aspect, quota, seed, filters[aspect].per_repository_limit)
            for aspect, quota in quotas.items()
        }
        try:
            # 3. 扫描全体合格语料，每个方面独立积累自己的配额候选。
            self._scan_candidates(
                matcher,
                filters,
                selectors,
                repositories,
                cleaning_version,
                scan_limit,
                stats,
            )
            # 4. 合并时只去掉重复正文，不丢失各方面的配额归属。
            selection = merge_selections(selectors)
            self._record_selection_stats(stats, selectors, selection)
            self._save_selection(sample_id, selection)
            stats["selected"] = len(selection.candidates)
            stats["overlap_selections"] = selection.overlap_count
            self.storage.finish_sample_set(sample_id, "completed", stats)
            return stats
        except Exception:
            # 半成品不能送去标注；下次以同一配置重试时由 Storage 清理。
            self.storage.finish_sample_set(sample_id, "failed", stats)
            raise

    def _scan_candidates(
        self,
        matcher: LexiconMatcher,
        filters: dict[str, AspectFilters],
        selectors: dict[str, AspectQuotaSelector],
        repositories: dict[int, str],
        cleaning_version: str,
        scan_limit: int | None,
        stats: dict[str, Any],
    ) -> None:
        watermark = self.storage.corpus_high_watermark()
        stats["corpus_high_watermark"] = watermark
        batches = self.storage.iter_sample_candidates(
            tuple(repositories),
            cleaning_version=cleaning_version,
            max_model_input_chars=scan_limit,
            include_text=True,
            max_corpus_id=watermark,
        )
        for batch in batches:
            for row in batch:
                stats["scanned"] += 1
                matches = matcher.match(row["clean_text"])
                if not matches:
                    continue
                candidate = Candidate.from_row(row, matches)
                for aspect in matches:
                    if filters[aspect].accepts(row, repositories[row["repository_id"]]):
                        selectors[aspect].consider(candidate)
                        stats["aspects"][aspect]["eligible"] = selectors[aspect].eligible_count
            if stats["scanned"] % 50000 < len(batch):
                logger.info("词典检索已扫描 %d 条", stats["scanned"])

    def _save_selection(self, sample_id: int, selection: SelectionResult) -> None:
        """将内存结果转成两种表记录：去重后的集合成员、各方面命中证据。"""
        items = []
        hits = []
        repository_ranks: dict[int, int] = {}
        for corpus_id, candidate in sorted(selection.candidates.items()):
            repository_id = candidate.repository_id
            repository_ranks[repository_id] = repository_ranks.get(repository_id, 0) + 1
            items.append(
                {
                    "sample_set_id": sample_id,
                    "corpus_id": corpus_id,
                    "repository_id": repository_id,
                    "source_type": candidate.source_type,
                    "sample_rank": repository_ranks[repository_id],
                    "selected_at": utcnow(),
                }
            )
            for aspect, evidence in candidate.matches.items():
                hits.append(
                    {
                        "sample_set_id": sample_id,
                        "corpus_id": corpus_id,
                        "candidate_aspect": aspect,
                        "selected_for_quota": corpus_id in selection.aspect_memberships[aspect],
                        "content_hash": candidate.content_hash,
                        "input_sha256": candidate.input_sha256,
                        "field_sha256": candidate.field_sha256,
                        "evidence": evidence,
                    }
                )
        self.storage.insert_sample_items(items)
        self.storage.insert_lexicon_hits(hits)

    @staticmethod
    def _initial_stats(
        sample_id: int,
        name: str,
        seed: str,
        cleaning_version: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "sample_set_id": sample_id,
            "name": name.strip(),
            "selection_method": "lexicon",
            "selection_config": config,
            "cleaning_version": cleaning_version,
            "seed": str(seed),
            "scanned": 0,
            "selected": 0,
            "reused": False,
            "aspects": {
                aspect: {"requested": quota, "eligible": 0}
                for aspect, quota in config["quotas"].items()
            },
        }

    @staticmethod
    def _record_selection_stats(
        stats: dict[str, Any],
        selectors: dict[str, AspectQuotaSelector],
        selection: SelectionResult,
    ) -> None:
        for aspect, selector in selectors.items():
            count = len(selection.aspect_memberships[aspect])
            stats["aspects"][aspect].update(selected=count, shortage=selector.quota - count)
