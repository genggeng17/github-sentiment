from datetime import datetime

from corpus.builder import make_corpus_row
from corpus.cleaning import CLEANING_VERSION, clean_text, normalize_text


def source(source_type, **overrides):
    data = {
        "source_type": source_type,
        "source_id": 10,
        "parent_id": 2,
        "title": "Faster compiler",
        "body": "This is great!",
        "path": "src/main.rs",
        "source_updated_at": datetime(2025, 1, 1),
    }
    data.update(overrides)
    return data


def test_issue_uses_title_and_body_as_target():
    row = make_corpus_row(source("issue"))
    assert row["context_text"] == ""
    assert row["target_text"] == "Faster compiler\n\nThis is great!"
    assert "[TARGET]\nFaster compiler" in row["model_input"]


def test_comment_separates_context_from_target():
    row = make_corpus_row(source("issue_comment", body="Compilation is painfully slow."))
    assert row["context_text"] == "Issue title: Faster compiler"
    assert row["target_text"] == "Compilation is painfully slow."
    assert row["model_input"].index("[CONTEXT]") < row["model_input"].index("[TARGET]")


def test_review_comment_context_does_not_contain_path():
    row = make_corpus_row(source("pr_review_comment"))
    assert row["context_text"] == "Pull request title: Faster compiler"
    assert "src/main.rs" not in row["model_input"]
    assert row["target_text"] == "This is great!"


def test_review_file_path_does_not_affect_corpus_version():
    first = make_corpus_row(source("pr_review_comment", path="src/main.rs"))
    second = make_corpus_row(source("pr_review_comment", path="tests/test_main.rs"))
    assert first["content_hash"] == second["content_hash"]
    assert first["model_input"] == second["model_input"]


def test_normalization_preserves_code_but_removes_hidden_markup():
    value = "hello\u200b  \r\n<!-- bot -->\r\n\r\n\r\n```rust\nfn main() {}\n```"
    cleaned = normalize_text(value)
    assert "bot" not in cleaned
    assert "\u200b" not in cleaned
    assert "```rust" in cleaned
    assert "\n\n\n" not in cleaned


def test_cleaning_collapses_fenced_code_but_preserves_surrounding_sentiment():
    value = (
        "This API is extremely confusing.\n\n"
        "```rust\nfn main() {\n    panic!(\"boom\");\n}\n```\n\n"
        "The compiler message gives no useful hint."
    )
    cleaned = clean_text(value)
    assert "This API is extremely confusing." in cleaned
    assert "The compiler message gives no useful hint." in cleaned
    assert "[CODE_BLOCK_REMOVED: 3 lines]" in cleaned
    assert "panic!" not in cleaned


def test_cleaning_preserves_inline_code_and_short_diagnostic():
    value = "The `foo()` API is hard to use.\nerror[E0001]: example"
    cleaned = clean_text(value)
    assert cleaned == value


def test_cleaning_collapses_stack_trace_without_consuming_following_prose():
    value = (
        "The crash is easy to reproduce.\n\n"
        "stack backtrace:\n"
        "   0: example::first\n"
        "   1: example::second\n\n"
        "The debugging experience is frustrating."
    )
    cleaned = clean_text(value)
    assert "[STACK_TRACE_REMOVED: 3 lines]" in cleaned
    assert "example::first" not in cleaned
    assert "The debugging experience is frustrating." in cleaned


def test_cleaning_collapses_long_compiler_output_run():
    value = (
        "The suggestion is misleading.\n"
        "error[E0001]: example\n"
        " --> src/main.rs:1:1\n"
        "1 | broken()\n"
        "The documentation is otherwise excellent."
    )
    cleaned = clean_text(value)
    assert "[TECHNICAL_OUTPUT_REMOVED: 3 lines]" in cleaned
    assert "The suggestion is misleading." in cleaned
    assert "The documentation is otherwise excellent." in cleaned


def test_cleaning_removes_presentation_wrappers_without_removing_prose():
    value = (
        "<details>\n<summary>Long output</summary>\n"
        "```text\nerror\n```\n</details>\n"
        "The error message is confusing.\n</body>\n</html>"
    )
    cleaned = clean_text(value)
    assert "details" not in cleaned
    assert "summary" not in cleaned
    assert "</body>" not in cleaned
    assert "The error message is confusing." in cleaned


def test_corpus_keeps_full_target_but_sends_denoised_text_to_model():
    row = make_corpus_row(
        source(
            "issue",
            body="This is frustrating.\n\n```rust\nfn main() {}\n```",
        )
    )
    assert "fn main()" in row["target_text"]
    assert "fn main()" not in row["clean_text"]
    assert "[CODE_BLOCK_REMOVED: 1 lines]" in row["clean_text"]
    assert row["clean_text"] in row["model_input"]
    assert row["model_input_chars"] == len(row["model_input"])
    assert row["cleaning_version"] == CLEANING_VERSION == "clean-v2"
