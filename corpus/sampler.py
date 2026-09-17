from __future__ import annotations

import hashlib
import heapq
import logging
import time
from typing import Any

from storage import Storage
from storage.models import utcnow

from .cleaning import CLEANING_VERSION

logger = logging.getLogger(__name__)


class CorpusSampler:
    def __init__(self, storage: Storage):
        self.storage = storage

    @staticmethod
    def _score(seed: str, repository_id: int, row: dict[str, Any]) -> int:
        material = (
            f"{seed}\0{repository_id}\0{row['source_type']}\0"
            f"{row['id']}\0{row['content_hash']}"
        )
        return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest(), "big")

    def _select_repositories(
        self,
        repository_ids: tuple[int, ...],
        limit: int,
        seed: str,
        cleaning_version: str,
        max_model_input_chars: int | None,
    ) -> tuple[dict[int, int], dict[int, list[dict[str, Any]]]]:
        eligible = {repository_id: 0 for repository_id in repository_ids}
        selected: dict[int, list[tuple[int, int, dict[str, Any]]]] = {
            repository_id: [] for repository_id in repository_ids
        }
        scanned = 0
        started_at = time.monotonic()
        for batch in self.storage.iter_sample_candidates(
            repository_ids,
            cleaning_version=cleaning_version,
            max_model_input_chars=max_model_input_chars,
        ):
            for row in batch:
                repository_id = row["repository_id"]
                eligible[repository_id] += 1
                scanned += 1
                score = self._score(seed, repository_id, row)
                entry = (-score, -row["id"], row)
                repository_selection = selected[repository_id]
                if len(repository_selection) < limit:
                    heapq.heappush(repository_selection, entry)
                    continue
                worst_score = -repository_selection[0][0]
                worst_id = -repository_selection[0][1]
                if (score, row["id"]) < (worst_score, worst_id):
                    heapq.heapreplace(repository_selection, entry)
            if scanned and scanned % 50000 < len(batch):
                elapsed = max(time.monotonic() - started_at, 0.001)
                logger.info(
                    "采样扫描进度: 已检查 %d 条候选，速度 %.0f 条/秒",
                    scanned,
                    scanned / elapsed,
                )
        elapsed = max(time.monotonic() - started_at, 0.001)
        logger.info(
            "候选扫描完成: 共 %d 条，耗时 %.1f 秒，平均 %.0f 条/秒",
            scanned,
            elapsed,
            scanned / elapsed,
        )
        ordered = {
            repository_id: sorted(
                (row for _, _, row in repository_selection),
                key=lambda row: (
                    self._score(seed, repository_id, row),
                    row["id"],
                ),
            )
            for repository_id, repository_selection in selected.items()
        }
        return eligible, ordered

    def build(
        self,
        name: str,
        *,
        per_repository_limit: int = 5000,
        seed: str = "0",
        cleaning_version: str = CLEANING_VERSION,
        max_model_input_chars: int | None = None,
    ) -> dict[str, Any]:
        name = name.strip()
        seed = str(seed)
        cleaning_version = cleaning_version.strip()
        if not name:
            raise ValueError("采样集名称不能为空")
        if not cleaning_version:
            raise ValueError("清洗版本不能为空")
        if per_repository_limit <= 0:
            raise ValueError("每仓库采样上限必须大于 0")
        if max_model_input_chars is not None and max_model_input_chars <= 0:
            raise ValueError("候选语料字符长度上限必须大于 0")
        available_versions = self.storage.available_cleaning_versions()
        if cleaning_version not in available_versions:
            available = ", ".join(available_versions) or "（无）"
            raise ValueError(
                f"数据库中不存在清洗版本 {cleaning_version!r}；可用版本: {available}"
            )

        sample_set_id, completed_stats = self.storage.prepare_sample_set(
            name,
            per_repository_limit,
            seed,
            cleaning_version,
            max_model_input_chars,
        )
        if completed_stats is not None:
            return {
                **completed_stats,
                "sample_set_id": sample_set_id,
                "reused": True,
            }

        stats: dict[str, Any] = {
            "sample_set_id": sample_set_id,
            "name": name,
            "per_repository_limit": per_repository_limit,
            "seed": seed,
            "max_model_input_chars": max_model_input_chars,
            "cleaning_version": cleaning_version,
            "repositories": {},
            "eligible": 0,
            "selected": 0,
            "reused": False,
        }
        try:
            repositories = self.storage.list_repositories(enabled_only=True)
            repository_ids = tuple(repository["id"] for repository in repositories)
            eligible_by_repository, selected_by_repository = (
                self._select_repositories(
                    repository_ids,
                    per_repository_limit,
                    seed,
                    cleaning_version,
                    max_model_input_chars,
                )
            )
            all_rows: list[dict[str, Any]] = []
            for repository in repositories:
                repository_id = repository["id"]
                eligible = eligible_by_repository[repository_id]
                selected = selected_by_repository[repository_id]
                rows = [
                    {
                        "sample_set_id": sample_set_id,
                        "corpus_id": row["id"],
                        "repository_id": repository_id,
                        "source_type": row["source_type"],
                        "sample_rank": rank,
                        "selected_at": utcnow(),
                    }
                    for rank, row in enumerate(selected, start=1)
                ]
                all_rows.extend(rows)
                stats["repositories"][repository["full_name"]] = {
                    "eligible": eligible,
                    "selected": len(rows),
                }
                stats["eligible"] += eligible
                stats["selected"] += len(rows)
            self.storage.insert_sample_items(all_rows)
            logger.info("采样项写入完成: %d 条", len(all_rows))
            self.storage.finish_sample_set(sample_set_id, "completed", stats)
            return stats
        except Exception:
            self.storage.finish_sample_set(sample_set_id, "failed", stats)
            raise
