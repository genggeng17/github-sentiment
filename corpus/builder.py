"""构造统一语料记录，并通过 Storage 完成去重和写入。"""

from __future__ import annotations

import hashlib
from typing import Any

from storage import Storage
from storage.models import utcnow

from .cleaning import CLEANING_VERSION, clean_text, detect_language, normalize_text


def make_corpus_row(source: dict[str, Any]) -> dict[str, Any]:
    source_type = source["source_type"]
    raw_title = source.get("title") or ""
    raw_body = source.get("body") or ""
    title = normalize_text(raw_title)
    body = normalize_text(raw_body)
    clean_title = clean_text(raw_title)
    clean_body = clean_text(raw_body)
    if source_type in {"issue", "pull_request"}:
        context = ""
        target = "\n\n".join(part for part in (title, body) if part)
        label_target = "\n\n".join(part for part in (clean_title, clean_body) if part)
        raw_context = ""
        raw_text = "\n\n".join(part for part in (raw_title, raw_body) if part)
    elif source_type == "issue_comment":
        context = f"Issue title: {title}"
        target = body
        label_context = f"Issue title: {clean_title}"
        label_target = clean_body
        raw_context = f"Issue title: {raw_title}"
        raw_text = raw_body
    elif source_type == "pr_issue_comment":
        context = f"Pull request title: {title}"
        target = body
        label_context = f"Pull request title: {clean_title}"
        label_target = clean_body
        raw_context = f"Pull request title: {raw_title}"
        raw_text = raw_body
    elif source_type == "pr_review_comment":
        context = f"Pull request title: {title}"
        target = body
        label_context = f"Pull request title: {clean_title}"
        label_target = clean_body
        raw_context = f"Pull request title: {raw_title}"
        raw_text = raw_body
    else:
        raise ValueError(f"未知语料来源: {source_type}")

    if source_type in {"issue", "pull_request"}:
        label_context = ""
    model_input = (
        f"[CONTEXT]\n{label_context or '(none)'}\n\n[TARGET]\n{label_target}"
    )
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
        "model_input_chars": len(model_input),
        "clean_text": label_target,
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
