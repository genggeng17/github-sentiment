"""按方面配额选候选；堆和排序细节集中在这里，不进入主流程。"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """内存中只保留身份、哈希和命中证据，不复制长篇正文。"""

    corpus_id: int
    repository_id: int
    source_type: str
    content_hash: str
    input_sha256: str
    field_sha256: str
    matches: dict[str, list[dict[str, Any]]]

    @classmethod
    def from_row(cls, row: dict[str, Any], matches: dict[str, list[dict[str, Any]]]) -> Candidate:
        return cls(
            corpus_id=row["id"],
            repository_id=row["repository_id"],
            source_type=row["source_type"],
            content_hash=row["content_hash"],
            input_sha256=hashlib.sha256(row["model_input"].encode()).hexdigest(),
            field_sha256=hashlib.sha256(row["clean_text"].encode()).hexdigest(),
            matches=matches,
        )


@dataclass(order=True)
class _RankedCandidate:
    # heapq 是最小堆。取负数后，堆顶恰好是当前最应该被淘汰的候选。
    negative_score: int
    negative_corpus_id: int
    candidate: Candidate = field(compare=False)


class AspectQuotaSelector:
    """独立管理一个方面的条数、仓库上限和确定性选择结果。"""

    def __init__(self, aspect: str, quota: int, seed: str, repository_limit: int | None):
        self.aspect = aspect
        self.quota = quota
        self.seed = seed
        self.repository_limit = repository_limit
        self.eligible_count = 0
        self._groups: dict[int | None, list[_RankedCandidate]] = {}

    def consider(self, candidate: Candidate) -> None:
        """接收已通过词典和过滤条件的候选，只保留有机会入选的记录。"""
        self.eligible_count += 1
        if self.repository_limit is None:
            group, capacity = None, self.quota
        else:
            group = candidate.repository_id
            capacity = min(self.repository_limit, self.quota)

        ranked = self._rank(candidate)
        heap = self._groups.setdefault(group, [])
        if len(heap) < capacity:
            heapq.heappush(heap, ranked)
        elif ranked > heap[0]:
            heapq.heapreplace(heap, ranked)

    def selected(self) -> list[Candidate]:
        """有仓库上限时，先取各仓库候选，再按同一排序取全局配额。"""
        candidates = (entry for heap in self._groups.values() for entry in heap)
        return [entry.candidate for entry in heapq.nlargest(self.quota, candidates)]

    def _rank(self, candidate: Candidate) -> _RankedCandidate:
        # 与原实现完全相同的哈希材料，重构后同种子仍选中相同 corpus_id。
        material = (
            f"{self.seed}\0{self.aspect}\0{candidate.repository_id}\0"
            f"{candidate.source_type}\0{candidate.corpus_id}\0{candidate.content_hash}"
        )
        score = int.from_bytes(hashlib.sha256(material.encode()).digest(), "big")
        return _RankedCandidate(-score, -candidate.corpus_id, candidate)


@dataclass
class SelectionResult:
    """候选正文只计一次，各方面的配额归属分别保留。"""

    candidates: dict[int, Candidate]
    aspect_memberships: dict[str, set[int]]

    @property
    def overlap_count(self) -> int:
        return sum(len(ids) for ids in self.aspect_memberships.values()) - len(self.candidates)


def merge_selections(selectors: dict[str, AspectQuotaSelector]) -> SelectionResult:
    candidates = {}
    memberships = {}
    for aspect, selector in selectors.items():
        selected = selector.selected()
        memberships[aspect] = {candidate.corpus_id for candidate in selected}
        candidates.update({candidate.corpus_id: candidate for candidate in selected})
    return SelectionResult(candidates, memberships)
