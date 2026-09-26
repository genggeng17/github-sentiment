"""Select confirmed aspect labels and prepare target text for topic discovery."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from corpus.cleaning import clean_text
from taxonomy import ASPECTS, CLASSES

logger = logging.getLogger(__name__)

TOPIC_PREPROCESSING_VERSION = "topic-text-v1"
TOPIC_SNAPSHOT_VERSION = "topic-snapshot-v1"

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_REMOVED = re.compile(
    r"\[(?:CODE_BLOCK|STACK_TRACE|TECHNICAL_OUTPUT)_REMOVED:[^]]*]", re.IGNORECASE
)
_MARKDOWN_IMAGE = re.compile(r"!\[[^]]*]\([^)]*\)")
_MARKDOWN_LINK = re.compile(r"\[([^]]+)]\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class TopicDocument:
    corpus_id: int
    source_type: str
    source_id: int
    parent_id: int | None
    content_hash: str
    repository: str
    created_at: datetime
    github_url: str
    text: str
    sentiments: dict[str, str]


def parse_sentiments(value: Any, corpus_id: int) -> dict[str, str]:
    if not isinstance(value, dict) or not isinstance(value.get("annotations"), list):
        raise ValueError(f"corpus_id={corpus_id} 的成功标注缺少 annotations 数组")
    result: dict[str, str] = {}
    for item in value["annotations"]:
        if not isinstance(item, dict):
            raise ValueError(f"corpus_id={corpus_id} 含非法标注元素")
        aspect, sentiment = item.get("aspect"), item.get("class")
        if aspect not in ASPECTS or sentiment not in CLASSES or aspect in result:
            raise ValueError(f"corpus_id={corpus_id} 含非法或重复的方面情感标签")
        result[aspect] = sentiment
    return result


def prepare_text(value: str, max_chars: int) -> tuple[str, bool]:
    """Reuse the corpus cleaner without changing saved labels or source text."""
    cleaned = clean_text(value)
    cleaned = _REMOVED.sub(" ", cleaned)
    cleaned = _MARKDOWN_IMAGE.sub(" ", cleaned)
    cleaned = _MARKDOWN_LINK.sub(r"\1", cleaned)
    cleaned = _URL.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned[:max_chars], len(cleaned) > max_chars


def select_aspect_members(
    documents: dict[int, TopicDocument],
    *,
    aspects: tuple[str, ...],
    max_per_aspect: int | None,
    seed: int,
) -> dict[str, list[int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for corpus_id, document in documents.items():
        for aspect in document.sentiments:
            if aspect in aspects:
                members[aspect].append(corpus_id)
    for aspect in aspects:
        ids = members[aspect]
        if max_per_aspect is not None and len(ids) > max_per_aspect:
            ids.sort(
                key=lambda corpus_id: hashlib.sha256(
                    f"{seed}:{aspect}:{corpus_id}".encode("ascii")
                ).digest()
            )
            del ids[max_per_aspect:]
        ids.sort()
    return dict(members)


def load_snapshot_documents(
    snapshot_dir: Path,
    *,
    config: dict[str, Any],
) -> tuple[dict[int, TopicDocument], dict[str, list[int]], dict[str, Any], dict[str, Any]]:
    manifest_path = snapshot_dir / "manifest.json"
    corpus_path = snapshot_dir / "documents.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        key: config[key]
        for key in (
            "taxonomy_version",
            "prompt_version",
            "model_name",
            "cleaning_version",
            "language",
            "max_chars",
            "min_chars",
        )
    }
    if (
        manifest.get("source") != expected
        or manifest.get("topic_preprocessing_version") != TOPIC_PREPROCESSING_VERSION
        or manifest.get("snapshot_format_version") != TOPIC_SNAPSHOT_VERSION
    ):
        raise ValueError("快照的数据版本或清洗参数与 fit 命令不一致")

    documents: dict[int, TopicDocument] = {}
    aspects = tuple(config["aspects"])
    limit = config["max_docs_per_aspect"]
    heaps: dict[str, list[tuple[int, int, int]]] = {aspect: [] for aspect in aspects}
    members: dict[str, list[int]] = {aspect: [] for aspect in aspects}
    retained: Counter[int] = Counter()
    seen_ids: set[int] = set()
    digest = hashlib.sha256()
    total = 0
    with corpus_path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            row = json.loads(line)
            corpus_id = row["corpus_id"]
            if corpus_id in seen_ids:
                raise ValueError(f"快照含重复 corpus_id={corpus_id}")
            seen_ids.add(corpus_id)
            total += 1
            chosen = []
            for aspect in aspects:
                if aspect not in row["sentiments"]:
                    continue
                if limit is None:
                    members[aspect].append(corpus_id)
                    chosen.append(aspect)
                    continue
                rank = int.from_bytes(
                    hashlib.sha256(
                        f"{config['seed']}:{aspect}:{corpus_id}".encode("ascii")
                    ).digest(),
                    "big",
                )
                heap = heaps[aspect]
                candidate = (-rank, -corpus_id, corpus_id)
                if len(heap) < limit:
                    heapq.heappush(heap, candidate)
                elif candidate > heap[0]:
                    evicted = heapq.heapreplace(heap, candidate)[2]
                    retained[evicted] -= 1
                    if retained[evicted] == 0:
                        del retained[evicted]
                        documents.pop(evicted)
                else:
                    continue
                chosen.append(aspect)
            if chosen:
                document = TopicDocument(
                    corpus_id=corpus_id,
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    parent_id=row["parent_id"],
                    content_hash=row["content_hash"],
                    repository=row["repository"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    github_url=row["github_url"],
                    text=row["text"],
                    sentiments=row["sentiments"],
                )
                documents[corpus_id] = document
                retained[corpus_id] += len(chosen)
            if total % 1000 == 0:
                logger.info("fit 快照读取进度：扫描 %d 条，内存保留 %d 条", total, len(documents))
    if digest.hexdigest() != manifest.get("documents_sha256") or total != manifest.get(
        "selection", {}
    ).get("usable_rows"):
        raise ValueError("快照文件损坏或不完整")
    if limit is not None:
        members = {aspect: sorted(item[2] for item in heaps[aspect]) for aspect in aspects}
    else:
        for ids in members.values():
            ids.sort()
    logger.info("fit 快照读取完成：扫描 %d 条，选中 %d 条不重复语料", total, len(documents))
    return documents, members, manifest["selection"], manifest
