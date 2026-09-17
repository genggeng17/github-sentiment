"""文本规范化、技术块清洗和语言检测；不访问数据库。"""

import re
import unicodedata

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
