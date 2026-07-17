from datetime import datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from corpus_builder import CorpusBuilder
from storage import Storage
from storage.models import Base, Corpus, Issue, LlmAnnotation, PipelineRun, PullRequestComment


@pytest.fixture
def storage():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    value = Storage("", engine=engine)
    value.create_schema()
    yield value
    engine.dispose()


def common(body="Body"):
    return {
        "author_login": "octocat",
        "author_github_id": 7,
        "body": body,
        "github_url": "https://github.test/item",
        "created_at": datetime(2025, 1, 1),
        "updated_at": datetime(2025, 1, 2),
        "collected_at": datetime(2025, 1, 3),
    }


def seed_sources(storage):
    repository_id = storage.ensure_repository("rust-lang/rust")
    storage.upsert_issues(
        repository_id,
        [
            {
                **common("Issue body"),
                "github_id": 10,
                "number": 1,
                "title": "Issue title",
                "state": "open",
                "closed_at": None,
            }
        ],
    )
    storage.upsert_pull_requests(
        repository_id,
        [
            {
                **common("PR body"),
                "github_id": 20,
                "number": 2,
                "title": "PR title",
                "state": "open",
                "closed_at": None,
                "merged_at": None,
            }
        ],
    )
    storage.upsert_issue_comments(
        repository_id, [{**common("Issue comment"), "github_id": 30, "parent_number": 1}]
    )
    storage.upsert_pr_comments(
        repository_id,
        [
            {
                **common("PR comment"),
                "github_id": 40,
                "parent_number": 2,
                "comment_type": "issue_comment",
                "path": None,
                "in_reply_to_github_id": None,
            },
            {
                **common("Review comment"),
                "github_id": 50,
                "parent_number": 2,
                "comment_type": "review_comment",
                "path": "src/lib.rs",
                "in_reply_to_github_id": None,
            },
        ],
    )
    return repository_id


def test_raw_upserts_are_idempotent(storage):
    repository_id = seed_sources(storage)
    seed_sources(storage)
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Issue)) == 1
        assert session.scalar(select(func.count()).select_from(PullRequestComment)) == 2
    assert storage.classify_parent_numbers(repository_id, {1, 2}) == ({1}, {2})


def test_builds_all_four_raw_kinds_into_five_corpus_sources(storage):
    seed_sources(storage)
    first = CorpusBuilder(storage).build()
    second = CorpusBuilder(storage).build()
    assert first["written"] == 5
    assert second["written"] == 5
    with storage.sessions() as session:
        rows = session.scalars(select(Corpus).order_by(Corpus.source_type)).all()
    assert len(rows) == 5
    review = next(row for row in rows if row.source_type == "pr_review_comment")
    assert review.context_text == "Pull request title: PR title"
    assert "src/lib.rs" not in review.model_input
    issue = next(row for row in rows if row.source_type == "issue")
    assert issue.raw_text == "Issue title\n\nIssue body"


def test_annotation_upsert_keeps_single_trace_row(storage):
    seed_sources(storage)
    CorpusBuilder(storage).build()
    with storage.sessions() as session:
        corpus_id = session.scalar(select(Corpus.id).where(Corpus.source_type == "issue"))
    row = {
        "corpus_id": corpus_id,
        "taxonomy_version": "v1",
        "prompt_version": "p1",
        "model_name": "deepseek-chat",
        "raw_response": '{"annotations":[]}',
        "parsed_result": {"annotations": []},
        "status": "succeeded",
        "error_message": None,
    }
    storage.save_annotation(row)
    storage.save_annotation({**row, "raw_response": '{"annotations": []}'})
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(LlmAnnotation)) == 1


def test_all_tables_compile_for_mysql():
    for table in Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=mysql.dialect()))
        assert "CREATE TABLE" in ddl


def test_edited_source_creates_new_immutable_corpus_version(storage):
    repository_id = seed_sources(storage)
    CorpusBuilder(storage).build()
    storage.upsert_issues(
        repository_id,
        [
            {
                **common("Edited body"),
                "github_id": 10,
                "number": 1,
                "title": "Issue title",
                "state": "open",
                "closed_at": None,
            }
        ],
    )
    CorpusBuilder(storage).build()
    with storage.sessions() as session:
        versions = session.scalars(
            select(Corpus).where(Corpus.source_type == "issue").order_by(Corpus.id)
        ).all()
    assert [row.raw_text for row in versions] == [
        "Issue title\n\nIssue body",
        "Issue title\n\nEdited body",
    ]


def test_new_run_marks_abandoned_running_record_failed(storage):
    old_id = storage.start_pipeline_run("run")
    new_id = storage.start_pipeline_run("run")
    with storage.sessions() as session:
        old = session.get(PipelineRun, old_id)
        new = session.get(PipelineRun, new_id)
    assert old.status == "failed"
    assert "接管" in old.error_message
    assert new.status == "running"
