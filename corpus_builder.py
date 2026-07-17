from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from storage import Storage
from storage.models import utcnow

CLEANING_VERSION = "clean-v1"

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_EXCESS_BLANKS = re.compile(r"\n{3,}")
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def clean_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _HTML_COMMENT.sub("", value)
    value = _ZERO_WIDTH.sub("", value)
    lines = [line.rstrip() for line in value.splitlines()]
    return _EXCESS_BLANKS.sub("\n\n", "\n".join(lines)).strip()


def detect_language(value: str) -> str:
    cjk = len(_CJK.findall(value))
    latin = len(_LATIN.findall(value))
    if cjk and cjk >= latin * 0.2:
        return "zh"
    if latin:
        return "en"
    return "unknown"


def make_corpus_row(source: dict[str, Any]) -> dict[str, Any]:
    source_type = source["source_type"]
    raw_title = source.get("title") or ""
    raw_body = source.get("body") or ""
    title = clean_text(raw_title)
    body = clean_text(raw_body)
    if source_type in {"issue", "pull_request"}:
        context = ""
        target = "\n\n".join(part for part in (title, body) if part)
        raw_context = ""
        raw_text = "\n\n".join(part for part in (raw_title, raw_body) if part)
    elif source_type == "issue_comment":
        context = f"Issue title: {title}"
        target = body
        raw_context = f"Issue title: {raw_title}"
        raw_text = raw_body
    elif source_type == "pr_issue_comment":
        context = f"Pull request title: {title}"
        target = body
        raw_context = f"Pull request title: {raw_title}"
        raw_text = raw_body
    elif source_type == "pr_review_comment":
        context = f"Pull request title: {title}"
        target = body
        raw_context = f"Pull request title: {raw_title}"
        raw_text = raw_body
    else:
        raise ValueError(f"未知语料来源: {source_type}")

    model_input = f"[CONTEXT]\n{context or '(none)'}\n\n[TARGET]\n{target}"
    version_material = f"{CLEANING_VERSION}\0{raw_context}\0{raw_text}"
    digest = hashlib.sha256(version_material.encode("utf-8")).hexdigest()
    return {
        "source_type": source_type,
        "source_id": source["source_id"],
        "parent_id": source.get("parent_id"),
        "raw_text": raw_text,
        "context_text": context,
        "target_text": target,
        "model_input": model_input,
        "clean_text": target,
        "language": detect_language(target),
        "content_hash": digest,
        "cleaning_version": CLEANING_VERSION,
        "duplicate_of_id": None,
        "source_updated_at": source["source_updated_at"],
        "updated_at": utcnow(),
    }


class CorpusBuilder:
    def __init__(self, storage: Storage):
        self.storage = storage

    def build(self, batch_size: int = 500) -> dict[str, int]:
        stats = {"read": 0, "written": 0, "duplicates": 0, "empty": 0}
        for candidates in self.storage.iter_corpus_candidates(batch_size):
            for source in candidates:
                stats["read"] += 1
                row = make_corpus_row(source)
                if not row["target_text"]:
                    stats["empty"] += 1
                    continue
                current_id = self.storage.get_corpus_id(
                    row["source_type"], row["source_id"], row["content_hash"]
                )
                duplicate_id = self.storage.find_corpus_by_hash(
                    row["content_hash"], exclude_id=current_id
                )
                row["duplicate_of_id"] = duplicate_id
                stats["duplicates"] += int(duplicate_id is not None)
                stats["written"] += self.storage.upsert_corpus([row])
        return stats
