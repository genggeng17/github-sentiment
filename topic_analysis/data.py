"""Select confirmed aspect labels and prepare target text for topic discovery."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from corpus.cleaning import clean_text
from taxonomy import ASPECTS, CLASSES

logger = logging.getLogger(__name__)

TOPIC_PREPROCESSING_VERSION = "topic-text-v1"

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


def load_documents(
    storage: Any,
    *,
    taxonomy_version: str,
    prompt_version: str,
    model_name: str,
    cleaning_version: str,
    language: str,
    max_chars: int,
    min_chars: int,
) -> tuple[dict[int, TopicDocument], dict[str, Any]]:
    documents: dict[int, TopicDocument] = {}
    stats: Counter[str] = Counter()
    for batch in storage.iter_confirmed_topic_documents(
        taxonomy_version=taxonomy_version,
        prompt_version=prompt_version,
        model_name=model_name,
        cleaning_version=cleaning_version,
        language=language,
    ):
        for row in batch:
            stats["successful_rows"] += 1
            sentiments = parse_sentiments(row["annotations"], row["corpus_id"])
            if not sentiments:
                stats["empty_aspect_rows"] += 1
                continue
            text, truncated = prepare_text(row["clean_text"], max_chars)
            if len(text) < min_chars:
                stats["too_short_rows"] += 1
                continue
            stats["truncated_rows"] += int(truncated)
            corpus_id = row["corpus_id"]
            if corpus_id in documents:
                raise RuntimeError(f"重复读取 corpus_id={corpus_id}")
            documents[corpus_id] = TopicDocument(
                corpus_id=corpus_id,
                source_type=row["source_type"],
                source_id=row["source_id"],
                parent_id=row["parent_id"],
                content_hash=row["content_hash"],
                repository=row["repository"],
                created_at=row["created_at"],
                github_url=row["github_url"],
                text=text,
                sentiments=sentiments,
            )
        logger.info(
            "语料读取进度：已读取 %d 条，保留 %d 条，空方面 %d 条，过短 %d 条",
            stats["successful_rows"],
            len(documents),
            stats["empty_aspect_rows"],
            stats["too_short_rows"],
        )
    stats["usable_rows"] = len(documents)
    return documents, dict(stats)
