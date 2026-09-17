"""内置全方面词典的检索边界；这些例子不是模型情感准确率测试。"""

import json
from importlib.resources import files

import pytest

from lexicon.matcher import LexiconMatcher
from taxonomy import ASPECTS


def definition(version):
    return json.loads(
        files("lexicon").joinpath("rules", f"rust-targeted-{version}.json").read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture
def matcher():
    return LexiconMatcher(definition("v2"), dict.fromkeys(ASPECTS, 1))


def test_full_taxonomy_and_original_rules_are_preserved():
    old, new = definition("v1"), definition("v2")
    assert set(new["aspects"]) == ASPECTS
    assert new["version"] != old["version"]
    for aspect, rules in old["aspects"].items():
        assert new["aspects"][aspect] == rules


@pytest.mark.parametrize(
    ("aspect", "english", "chinese"),
    [
        ("ownership", "This function takes ownership of the buffer.", "这里会转移所有权。"),
        ("type_system", "Type inference works well here.", "这里的类型推导很准确。"),
        ("safety", "This creates a data race.", "这里存在数据竞争。"),
        ("runtime_performance", "The new parser is faster.", "请求延迟降低了一半。"),
        ("learning_curve", "Learning Rust is confusing.", "Rust 很难入门。"),
        ("compile_time", "A clean build took 60 seconds.", "这次构建耗时一分钟。"),
        (
            "diagnostics_debugging",
            "The warning does not explain what to change.",
            "错误提示没有解释该怎么改。",
        ),
        (
            "tooling_documentation",
            "The tutorial does not cover configuring a custom backend.",
            "教程没有覆盖配置方式。",
        ),
        (
            "readability_maintainability",
            "The code is unnecessarily complex.",
            "这一改动提高了可读性。",
        ),
        ("api_extensibility", "This API is easy to use.", "这个接口易于扩展。"),
        ("package_manager", "Cargo dependency resolution is reliable.", "依赖解析是确定性的。"),
        ("libraries_frameworks", "The library is mature and stable.", "这个框架很成熟。"),
        (
            "community",
            "The maintainers review unanswered support requests every Friday.",
            "维护者会定期响应求助。",
        ),
    ],
)
def test_retrieves_english_and_chinese_evidence(matcher, aspect, english, chinese):
    for text in (english, chinese):
        hits = matcher.match(text)
        assert aspect in hits
        for evidence in hits[aspect]:
            assert text[evidence["start"] : evidence["end"]] == evidence["match"]
        assert matcher.match(f"> {text}\n\n```text\n{text}\n```\n") == {}


@pytest.mark.parametrize(
    ("text", "excluded"),
    [
        ("The certificate expired and authentication failed.", "safety"),
        ("The API returned status 404.", "api_extensibility"),
        ("Thanks! LGTM. Any updates on this PR?", "community"),
        ("The CI queue took 60 seconds.", "compile_time"),
        ("Compilation failed with E0308.", "compile_time"),
        ("The build took 60 seconds.", "runtime_performance"),
        ("The parser is faster.", "compile_time"),
        ("Run rustfmt. Update README.md. All tests pass.", "tooling_documentation"),
        ("The error occurred again. Fixed now.", "diagnostics_debugging"),
        ("The function was renamed.", "readability_maintainability"),
        ("The ownership of the GitHub organization changed.", "ownership"),
        ("If you take ownership of this code, you bear responsibility for it.", "ownership"),
        ("I will point to a related suggestion on that issue.", "diagnostics_debugging"),
        ("The name field is optional when the private key is unavailable.", "api_extensibility"),
        ("The API exists.\n\nAn unrelated task is easy to use.", "api_extensibility"),
    ],
)
def test_avoids_common_unrelated_triggers(matcher, text, excluded):
    assert excluded not in matcher.match(text)


def test_multiple_aspects_keep_separate_evidence(matcher):
    text = "中文🙂 The borrow checker prevents data races. A clean build took 60 seconds."
    hits = matcher.match(text)
    assert {"ownership", "safety", "compile_time"} <= set(hits)
    assert all(
        text[item["start"] : item["end"]] == item["match"]
        for evidence in hits.values()
        for item in evidence
    )
