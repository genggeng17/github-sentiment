from __future__ import annotations

import hashlib
import heapq
from typing import Any

from corpus_builder import CLEANING_VERSION
from storage import Storage
from storage.models import utcnow


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

    def _select_repository(
        self,
        repository_id: int,
        limit: int,
        seed: str,
    ) -> tuple[int, list[dict[str, Any]]]:
        eligible = 0
        selected: list[tuple[int, int, dict[str, Any]]] = []
        for batch in self.storage.iter_sample_candidates(
            repository_id,
            cleaning_version=CLEANING_VERSION,
        ):
            for row in batch:
                eligible += 1
                score = self._score(seed, repository_id, row)
                entry = (-score, -row["id"], row)
                if len(selected) < limit:
                    heapq.heappush(selected, entry)
                    continue
                worst_score = -selected[0][0]
                worst_id = -selected[0][1]
                if (score, row["id"]) < (worst_score, worst_id):
                    heapq.heapreplace(selected, entry)
        ordered = sorted(
            (row for _, _, row in selected),
            key=lambda row: (self._score(seed, repository_id, row), row["id"]),
        )
        return eligible, ordered

    def build(
        self,
        name: str,
        *,
        per_repository_limit: int = 5000,
        seed: str = "0",
    ) -> dict[str, Any]:
        name = name.strip()
        seed = str(seed)
        if not name:
            raise ValueError("采样集名称不能为空")
        if per_repository_limit <= 0:
            raise ValueError("每仓库采样上限必须大于 0")

        sample_set_id, completed_stats = self.storage.prepare_sample_set(
            name,
            per_repository_limit,
            seed,
        )
        if completed_stats is not None:
            if completed_stats.get("cleaning_version") != CLEANING_VERSION:
                raise ValueError(
                    f"采样集 {name!r} 使用的清洗版本不是 {CLEANING_VERSION}；"
                    "请使用新的采样集名称"
                )
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
            "cleaning_version": CLEANING_VERSION,
            "repositories": {},
            "eligible": 0,
            "selected": 0,
            "reused": False,
        }
        try:
            repositories = self.storage.list_repositories(enabled_only=True)
            for repository in repositories:
                repository_id = repository["id"]
                eligible, selected = self._select_repository(
                    repository_id,
                    per_repository_limit,
                    seed,
                )
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
                self.storage.insert_sample_items(rows)
                stats["repositories"][repository["full_name"]] = {
                    "eligible": eligible,
                    "selected": len(rows),
                }
                stats["eligible"] += eligible
                stats["selected"] += len(rows)
            self.storage.finish_sample_set(sample_set_id, "completed", stats)
            return stats
        except Exception:
            self.storage.finish_sample_set(sample_set_id, "failed", stats)
            raise
