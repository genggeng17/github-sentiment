from datetime import datetime

from corpus_builder import clean_text, make_corpus_row


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


def test_cleaning_preserves_code_but_removes_hidden_markup():
    value = "hello\u200b  \r\n<!-- bot -->\r\n\r\n\r\n```rust\nfn main() {}\n```"
    cleaned = clean_text(value)
    assert "bot" not in cleaned
    assert "\u200b" not in cleaned
    assert "```rust" in cleaned
    assert "\n\n\n" not in cleaned
