from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any

from storage import Storage
from storage.models import utcnow

CLEANING_VERSION = "clean-v2"

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_EXCESS_BLANKS = re.compile(r"\n{3,}")
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_FENCE_START = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
_MARKDOWN_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}\s+")
_EMPTY_TEMPLATE_SECTION = re.compile(
    r"(?ms)^[ \t]{0,3}#{1,6}\s+[^\n]+\n+"
    r"[ \t]*(?:_No response_|No response provided\.?)[ \t]*"
    r"(?=\n[ \t]*\n^[ \t]{0,3}#{1,6}\s+|\Z)"
)
_DETAILS_WRAPPER_LINE = re.compile(
    r"^\s*(?:</?details\s*>|<summary(?:\s[^>]*)?>.*</summary>)\s*$",
    re.IGNORECASE,
)
_HTML_WRAPPER_ONLY_LINE = re.compile(
    r"^(?:\s*</?(?:span|div|body|html)(?:\s[^>]*)?>)+\s*$",
    re.IGNORECASE,
)
_STACK_TRACE_START = re.compile(
    r"^\s*(?:stack backtrace:|backtrace:|Traceback \(most recent call last\):)\s*$",
    re.IGNORECASE,
)
_TECHNICAL_LINE_PATTERNS = (
    # Timestamped application and CI logs.
    re.compile(
        r"^\s*(?:\d{4}-\d{2}-\d{2}[T ][0-9:.+-]+\s+)?"
        r"(?:\[[^\]]*\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b[^\]]*\]"
        r"|(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL)\b[: ]).*$",
        re.IGNORECASE,
    ),
    # Rust/Python/native stack frames and chained exceptions.
    re.compile(r"^\s*(?:\d+:\s+0x[0-9a-f]+\b|at\s+\S+.*:\d+|Caused by:)"),
    re.compile(r'^\s*File\s+"[^"]+",\s+line\s+\d+'),
    # rustc diagnostics and source-location gutters.
    re.compile(r"^\s*(?:error|warning)(?:\[[A-Z]\d+\])?:", re.IGNORECASE),
    re.compile(r"^\s*-->\s+\S+:\d+(?::\d+)?"),
    re.compile(r"^\s*(?:\d+\s*)?\|.*$"),
    re.compile(r"^\s*=\s*(?:note|help):", re.IGNORECASE),
    re.compile(r"^\s*[\^~_-]{3,}\s*$"),
    # Cargo/rustup progress output and common version appendices.
    re.compile(
        r"^\s*(?:Compiling|Checking|Finished|Downloading|Downloaded|Updating|Running|Fresh|Dirty)\b"
    ),
    re.compile(
        r"^\s*(?:rustc|cargo|release|commit-hash|commit-date|host|libgit2|libcurl|ssl|os)"
        r"\s*:?\s+\S+",
        re.IGNORECASE,
    ),
)


def normalize_text(value: str) -> str:
    """Apply lossless normalization while retaining all visible author content."""
    value = unicodedata.normalize("NFKC", value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _HTML_COMMENT.sub("", value)
    value = _ZERO_WIDTH.sub("", value)
    lines = [line.rstrip() for line in value.splitlines()]
    return _EXCESS_BLANKS.sub("\n\n", "\n".join(lines)).strip()


def _collapse_fenced_blocks(value: str) -> str:
    lines = value.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        match = _FENCE_START.match(lines[index])
        if match is None:
            output.append(lines[index])
            index += 1
            continue

        fence = match.group(1)
        closing = re.compile(
            rf"^[ \t]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$"
        )
        end = index + 1
        while end < len(lines) and closing.match(lines[end]) is None:
            end += 1
        content_end = end if end < len(lines) else len(lines)
        content_lines = content_end - index - 1
        output.append(f"[CODE_BLOCK_REMOVED: {max(0, content_lines)} lines]")
        index = end + 1 if end < len(lines) else len(lines)
    return "\n".join(output)


def _is_technical_line(line: str) -> bool:
    return any(pattern.match(line) is not None for pattern in _TECHNICAL_LINE_PATTERNS)


def _collapse_stack_traces(value: str) -> str:
    """Collapse explicitly introduced stack traces without consuming later prose."""
    lines = value.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if _STACK_TRACE_START.match(lines[index]) is None:
            output.append(lines[index])
            index += 1
            continue

        end = index + 1
        while end < len(lines):
            line = lines[end]
            if not line.strip() or _MARKDOWN_HEADING.match(line):
                break
            # Stack traces often contain indented symbol names that do not match a
            # stable language-specific pattern, so the explicit header is the boundary.
            end += 1
        output.append(f"[STACK_TRACE_REMOVED: {end - index} lines]")
        index = end
    return "\n".join(output)


def _collapse_technical_runs(value: str, minimum_lines: int = 3) -> str:
    """Collapse only long, contiguous runs of strongly technical output lines."""
    lines = value.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if not _is_technical_line(lines[index]):
            output.append(lines[index])
            index += 1
            continue

        end = index + 1
        while end < len(lines) and _is_technical_line(lines[end]):
            end += 1
        run_length = end - index
        if run_length >= minimum_lines:
            output.append(f"[TECHNICAL_OUTPUT_REMOVED: {run_length} lines]")
        else:
            output.extend(lines[index:end])
        index = end
    return "\n".join(output)


def _remove_standalone_wrappers(value: str) -> str:
    return "\n".join(
        line
        for line in value.splitlines()
        if _DETAILS_WRAPPER_LINE.match(line) is None
        and _HTML_WRAPPER_ONLY_LINE.match(line) is None
    )


def clean_text(value: str) -> str:
    """Build the denoised text sent to sentiment labelers.

    Natural-language prose and inline code are retained. Large technical blocks are
    represented by auditable placeholders instead of being silently deleted.
    """
    value = normalize_text(value)
    value = _EMPTY_TEMPLATE_SECTION.sub("", value)
    value = _collapse_fenced_blocks(value)
    value = _collapse_stack_traces(value)
    value = _collapse_technical_runs(value)
    value = _remove_standalone_wrappers(value)
    return _EXCESS_BLANKS.sub("\n\n", value).strip()


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
