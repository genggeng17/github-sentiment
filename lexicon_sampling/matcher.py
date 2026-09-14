"""词典匹配：只负责从 TARGET 找到候选方面和原文证据，不决定最终标签。"""

import re
from typing import Any

MATCHER_VERSION = "target-prose-regex-v1"


def prose_view(text: str) -> str:
    """用等长空白屏蔽代码块和引用，保证命中位置仍对应原文。"""
    result = []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        excluded = fence is not None or marker is not None or line.lstrip().startswith(">")
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
        result.append("".join("\n" if c == "\n" else " " for c in line) if excluded else line)
    return "".join(result)


class LexiconMatcher:
    def __init__(self, definition: dict[str, Any], aspects: dict[str, int]):
        if definition.get("schema_version") != 1 or not definition.get("version"):
            raise ValueError("词典需要 schema_version=1 和非空 version")
        self.rules = {}
        seen = set()
        for aspect in aspects:
            definition_for_aspect = definition.get("aspects", {}).get(aspect)
            if not isinstance(definition_for_aspect, dict):
                raise ValueError(f"词典缺少方面: {aspect}")
            compiled = []
            for rule in definition_for_aspect.get("rules", []):
                rule_id = rule.get("id")
                patterns = rule.get("all")
                if not isinstance(rule_id, str) or not rule_id or rule_id in seen:
                    raise ValueError(f"rule id 为空或重复: {rule_id}")
                if not isinstance(patterns, list) or not patterns:
                    raise ValueError(f"规则 {rule_id} 需要非空 all 正则列表")
                try:
                    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
                except (re.error, TypeError) as exc:
                    raise ValueError(f"规则 {rule_id} 的正则无效") from exc
                if any(regex.search("") for regex in regexes):
                    raise ValueError(f"规则 {rule_id} 不允许空匹配")
                compiled.append((rule_id, regexes))
                seen.add(rule_id)
            if not compiled:
                raise ValueError(f"方面 {aspect} 没有规则")
            self.rules[aspect] = compiled

    def match(self, text: str) -> dict[str, list[dict[str, Any]]]:
        view = prose_view(text)
        # 一条规则的所有表达式必须在同一段落命中。
        paragraphs = list(re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", view))
        hits = {}
        for aspect, rules in self.rules.items():
            evidence = []
            for rule_id, regexes in rules:
                for paragraph in paragraphs:
                    matches = [regex.search(paragraph.group()) for regex in regexes]
                    if not all(matches):
                        continue
                    for match in matches:
                        start = paragraph.start() + match.start()
                        end = paragraph.start() + match.end()
                        evidence.append(
                            {
                                "rule_id": rule_id,
                                "field": "clean_text",
                                "region": "target_prose",
                                "start": start,
                                "end": end,
                                "match": text[start:end],
                            }
                        )
                    break  # 每条规则保留第一组证据，避免明细无限增长。
            if evidence:
                hits[aspect] = evidence
        return hits
