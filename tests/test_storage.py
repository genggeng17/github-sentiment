from datetime import datetime

import pytest
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.dialects import mysql
from sqlalchemy.schema import CreateTable

from corpus import CorpusBuilder
from storage import Storage
from storage.database import MissingParentError
from storage.models import (
    Base,
    Corpus,
    Issue,
    IssueComment,
    LlmAnnotation,
    PipelineRun,
    PullRequestComment,
    RepositoryCursor,
    SentimentFact,
)


@pytest.fixture
def storage():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    value = Storage("", engine=engine)
    value.create_schema()
    yield value
    engine.dispose()


def test_create_schema_adds_sample_metadata_to_existing_database():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE corpus_sample_sets (id INTEGER PRIMARY KEY)")
        )

    Storage("", engine=engine).create_schema()

    columns = {
        column["name"] for column in inspect(engine).get_columns("corpus_sample_sets")
    }
    assert "cleaning_version" in columns
    assert "max_model_input_chars" in columns
    engine.dispose()


def test_create_schema_backfills_corpus_input_length_and_sampling_index():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE corpus ("
                "id INTEGER PRIMARY KEY, "
                "cleaning_version VARCHAR(30) NOT NULL, "
                "source_type VARCHAR(30) NOT NULL, "
                "duplicate_of_id BIGINT NULL, "
                "model_input TEXT NOT NULL"
                ")"
            )
        )
        connection.execute(
            text(
                "INSERT INTO corpus "
                "(id, cleaning_version, source_type, duplicate_of_id, model_input) "
                "VALUES (1, 'clean-v1', 'issue', NULL, '中文abc')"
            )
        )

    Storage("", engine=engine).create_schema()

    columns = {column["name"] for column in inspect(engine).get_columns("corpus")}
    indexes = {index["name"] for index in inspect(engine).get_indexes("corpus")}
    with engine.connect() as connection:
        length = connection.scalar(
            text("SELECT model_input_chars FROM corpus WHERE id = 1")
        )
    assert "model_input_chars" in columns
    assert "ix_corpus_sampling" in indexes
    assert length == 5
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


def add_fact_annotation(storage, corpus_id, *, version="v1", updated_at=None,
                        labels=None, status="succeeded"):
    with storage.sessions.begin() as session:
        annotation = LlmAnnotation(
            corpus_id=corpus_id,
            taxonomy_version="rust-aspects-v2",
            prompt_version=version,
            model_name="test-model",
            status=status,
            parsed_result={"annotations": labels if labels is not None else [
                {"aspect": "compile_time", "class": "negative"}
            ]},
            updated_at=updated_at or datetime(2026, 1, 1),
        )
        session.add(annotation)
        session.flush()
        return annotation.id


def fact_corpus(storage):
    repository_id = seed_sources(storage)
    CorpusBuilder(storage).build()
    with storage.sessions() as session:
        rows = session.scalars(select(Corpus).order_by(Corpus.id)).all()
    return repository_id, rows


def test_sentiment_facts_use_github_time_for_all_five_source_types(storage):
    repository_id, rows = fact_corpus(storage)
    for row in rows:
        add_fact_annotation(storage, row.id, labels=[
            {"aspect": "compile_time", "class": "negative"},
            {"aspect": "tooling_documentation", "class": "positive"},
        ])
    stats = storage.refresh_sentiment_facts(batch_size=2)
    assert stats == {"scanned_corpus": 5, "included_corpus": 5, "empty_corpus": 0, "rows": 10}
    with storage.sessions() as session:
        facts = session.scalars(select(SentimentFact)).all()
    assert {fact.corpus_id for fact in facts} == {row.id for row in rows}
    assert {fact.repository_id for fact in facts} == {repository_id}
    assert {fact.created_at for fact in facts} == {datetime(2025, 1, 1)}
    assert {row.source_type for row in rows} == {
        "issue", "pull_request", "issue_comment", "pr_issue_comment", "pr_review_comment"
    }
    assert storage.refresh_sentiment_facts(batch_size=1) == stats
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(SentimentFact)) == 10


@pytest.mark.parametrize("versions", [1, 2, 3])
def test_sentiment_facts_select_latest_successful_annotation(storage, versions):
    _, rows = fact_corpus(storage)
    corpus_id = rows[0].id
    for version in reversed(range(versions)):
        add_fact_annotation(
            storage, corpus_id, version=f"v{version}",
            updated_at=datetime(2026, 1, version + 1),
            labels=[{"aspect": "compile_time", "class": "positive" if version == versions - 1
                     else "negative"}],
        )
    add_fact_annotation(storage, corpus_id, version="failed-newest",
                        updated_at=datetime(2026, 2, 1), status="failed")
    stats = storage.refresh_sentiment_facts(batch_size=1)
    assert stats["rows"] == 1
    with storage.sessions() as session:
        fact = session.scalars(select(SentimentFact)).one()
    assert fact.sentiment == "positive"


def test_sentiment_facts_break_timestamp_tie_by_annotation_id(storage):
    _, rows = fact_corpus(storage)
    add_fact_annotation(storage, rows[0].id)
    add_fact_annotation(storage, rows[0].id, version="v2", labels=[
        {"aspect": "ownership", "class": "neutral"}
    ])
    storage.refresh_sentiment_facts()
    with storage.sessions() as session:
        fact = session.scalars(select(SentimentFact)).one()
    assert (fact.aspect, fact.sentiment) == ("ownership", "neutral")


def test_sentiment_facts_latest_empty_annotation_removes_old_labels(storage):
    _, rows = fact_corpus(storage)
    add_fact_annotation(storage, rows[0].id)
    storage.refresh_sentiment_facts()
    add_fact_annotation(storage, rows[0].id, version="v2", labels=[],
                        updated_at=datetime(2026, 2, 1))
    add_fact_annotation(storage, rows[1].id, labels=[
        {"aspect": "", "class": "negative"}, {"aspect": None}, {"aspect": "   "}, {}
    ])
    add_fact_annotation(storage, rows[2].id, status="failed")
    stats = storage.refresh_sentiment_facts(batch_size=1)
    assert stats == {"scanned_corpus": 2, "included_corpus": 0, "empty_corpus": 2, "rows": 0}
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(SentimentFact)) == 0


def test_sentiment_facts_refresh_reads_updated_existing_annotation(storage):
    _, rows = fact_corpus(storage)
    older_id = add_fact_annotation(storage, rows[0].id)
    add_fact_annotation(storage, rows[0].id, version="v2", labels=[
        {"aspect": "ownership", "class": "neutral"}
    ])
    storage.refresh_sentiment_facts()
    with storage.sessions.begin() as session:
        annotation = session.get(LlmAnnotation, older_id)
        annotation.updated_at = datetime(2026, 2, 1)
    storage.refresh_sentiment_facts()
    with storage.sessions() as session:
        fact = session.scalars(select(SentimentFact)).one()
    assert fact.aspect == "compile_time"


def test_sentiment_facts_refresh_rolls_back_when_source_is_missing(storage):
    _, rows = fact_corpus(storage)
    add_fact_annotation(storage, rows[0].id)
    storage.refresh_sentiment_facts()
    add_fact_annotation(storage, rows[1].id)
    with storage.sessions.begin() as session:
        session.get(Corpus, rows[1].id).source_id = 999999
    with pytest.raises(RuntimeError, match="语料来源缺失"):
        storage.refresh_sentiment_facts(batch_size=1)
    with storage.sessions() as session:
        facts = session.scalars(select(SentimentFact)).all()
    assert [(fact.corpus_id, fact.aspect) for fact in facts] == [(rows[0].id, "compile_time")]


@pytest.mark.parametrize("labels", [
    [{"aspect": "compile_time", "class": "invalid"}],
    [{"aspect": "compile_time", "class": "negative"},
     {"aspect": "compile_time", "class": "positive"}],
])
def test_sentiment_facts_reject_invalid_labels(storage, labels):
    _, rows = fact_corpus(storage)
    add_fact_annotation(storage, rows[0].id, labels=labels)
    with pytest.raises(ValueError):
        storage.refresh_sentiment_facts()


def test_sentiment_facts_cli_validates_batch_size():
    from pipeline import build_parser

    args = build_parser().parse_args(["refresh-sentiment-facts", "--batch-size", "200"])
    assert args.batch_size == 200
    with pytest.raises(SystemExit):
        build_parser().parse_args(["refresh-sentiment-facts", "--batch-size", "0"])


def test_sentiment_facts_cli_refreshes_and_records_run(storage, monkeypatch, capsys):
    import json

    import pipeline

    _, rows = fact_corpus(storage)
    add_fact_annotation(storage, rows[0].id)
    monkeypatch.setattr(pipeline, "Storage", lambda _url: storage)
    assert pipeline.main(["refresh-sentiment-facts", "--batch-size", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "succeeded"
    assert result["stats"]["rows"] == 1
    with storage.sessions() as session:
        run = session.get(PipelineRun, result["run_id"])
        assert run.run_type == "refresh-sentiment-facts"
        assert run.stats == result["stats"]


def test_raw_upserts_are_idempotent(storage):
    repository_id = seed_sources(storage)
    seed_sources(storage)
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Issue)) == 1
        assert session.scalar(select(func.count()).select_from(PullRequestComment)) == 2
    assert storage.classify_parent_numbers(repository_id, {1, 2}) == ({1}, {2})


def test_schema_upgrade_keeps_existing_raw_rows(storage):
    seed_sources(storage)
    storage.create_schema()
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Issue)) == 1
        assert session.scalar(select(func.count()).select_from(PullRequestComment)) == 2


def test_page_data_and_cursor_commit_atomically(storage):
    repository_id = storage.ensure_repository("rust-lang/rust")
    cursor = datetime(2025, 1, 2)
    written = storage.commit_collection_page(
        repository_id,
        "issues_and_pull_requests",
        cursor,
        issues=[
            {
                **common(),
                "github_id": 10,
                "number": 1,
                "title": "Issue",
                "state": "open",
                "closed_at": None,
            }
        ],
    )
    assert written == 1
    assert storage.get_cursor(repository_id, "issues_and_pull_requests") == cursor
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Issue)) == 1
        assert session.scalar(select(func.count()).select_from(RepositoryCursor)) == 1


def test_missing_parent_rolls_back_page_and_cursor_together(storage):
    repository_id = storage.ensure_repository("rust-lang/rust")
    with pytest.raises(MissingParentError):
        storage.commit_collection_page(
            repository_id,
            "issue_comments",
            datetime(2025, 1, 2),
            issue_comments=[
                {**common(), "github_id": 30, "parent_number": 999}
            ],
        )
    with storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(IssueComment)) == 0
        assert session.scalar(select(func.count()).select_from(RepositoryCursor)) == 0


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
