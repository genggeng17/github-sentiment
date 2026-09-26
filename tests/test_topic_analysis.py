import csv
import logging
import sys
import types
from datetime import datetime

import numpy as np
from sqlalchemy import create_engine, select

from corpus.builder import make_corpus_row
from storage import Storage
from storage.models import Issue
from taxonomy import TAXONOMY_VERSION
from topic_analysis.data import TopicDocument, parse_sentiments, prepare_text, select_aspect_members
from topic_analysis.reporting import build_rollups
from topic_analysis.service import TopicAnalysis


def _source(issue_id: int, text: str) -> dict:
    return {
        "source_type": "issue",
        "source_id": issue_id,
        "title": "",
        "body": text,
        "source_updated_at": datetime(2025, 2, 1),
    }


def _annotation(corpus_id: int, model: str = "chosen") -> dict:
    return {
        "corpus_id": corpus_id,
        "taxonomy_version": TAXONOMY_VERSION,
        "prompt_version": "prompt-test",
        "model_name": model,
        "parsed_result": {"annotations": [{"aspect": "package_manager", "class": "negative"}]},
        "status": "succeeded",
    }


def test_topic_reader_uses_latest_corpus_and_source_creation_time():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    storage = Storage("", engine=engine)
    storage.create_schema()
    repo_id = storage.ensure_repository("rust-lang/cargo")
    created = datetime(2025, 1, 3)
    for github_id in (101, 102):
        storage.upsert_issues(
            repo_id,
            [
                {
                    "github_id": github_id,
                    "number": github_id,
                    "title": "topic",
                    "body": "body",
                    "state": "open",
                    "author_login": "author",
                    "author_github_id": 7,
                    "github_url": f"https://github.test/issues/{github_id}",
                    "created_at": created,
                    "updated_at": datetime(2025, 2, 1),
                    "collected_at": datetime(2025, 2, 1),
                }
            ],
        )
    with storage.sessions() as session:
        issue_ids = {
            github_id: issue_id
            for issue_id, github_id in session.execute(select(Issue.id, Issue.github_id))
        }

    def add(issue_id: int, text: str) -> int:
        row = make_corpus_row(_source(issue_id, text))
        storage.upsert_corpus([row])
        return storage.get_corpus_id("issue", issue_id, row["content_hash"])

    superseded_id = add(issue_ids[101], "Old dependency resolver behavior")
    add(issue_ids[101], "New dependency resolver behavior")
    current_id = add(issue_ids[102], "Workspace member resolution behavior")
    storage.save_annotations([_annotation(superseded_id), _annotation(current_id)])

    rows = [
        item
        for batch in storage.iter_confirmed_topic_documents(
            taxonomy_version=TAXONOMY_VERSION,
            prompt_version="prompt-test",
            model_name="chosen",
            cleaning_version="clean-v2",
            batch_size=1,
        )
        for item in batch
    ]
    assert [row["corpus_id"] for row in rows] == [current_id]
    assert rows[0]["repository"] == "rust-lang/cargo"
    assert rows[0]["created_at"] == created
    assert rows[0]["github_url"].endswith("/102")
    assert rows[0]["annotations"]["annotations"][0]["class"] == "negative"
    inspect_rows = [
        item
        for batch in storage.iter_confirmed_topic_documents(
            taxonomy_version=TAXONOMY_VERSION,
            prompt_version="prompt-test",
            model_name="chosen",
            cleaning_version="clean-v2",
            batch_size=1,
            include_source_metadata=False,
        )
        for item in batch
    ]
    assert inspect_rows == [
        {
            "corpus_id": current_id,
            "clean_text": rows[0]["clean_text"],
            "annotations": rows[0]["annotations"],
        }
    ]
    engine.dispose()


def test_inspect_streams_counts_without_source_metadata(tmp_path, caplog):
    class FakeStorage:
        def iter_confirmed_topic_documents(self, **kwargs):
            assert kwargs["include_source_metadata"] is False
            yield [
                {
                    "corpus_id": 1,
                    "clean_text": "Cargo workspace dependency resolution",
                    "annotations": {
                        "annotations": [{"aspect": "package_manager", "class": "negative"}]
                    },
                },
                {"corpus_id": 2, "clean_text": "Ignore", "annotations": {"annotations": []}},
            ]
            yield [
                {
                    "corpus_id": 3,
                    "clean_text": "Hi",
                    "annotations": {
                        "annotations": [{"aspect": "package_manager", "class": "positive"}]
                    },
                },
                {
                    "corpus_id": 4,
                    "clean_text": "Cargo feature resolver and build documentation",
                    "annotations": {
                        "annotations": [
                            {"aspect": "package_manager", "class": "neutral"},
                            {"aspect": "tooling_documentation", "class": "positive"},
                        ]
                    },
                },
            ]

    with caplog.at_level(logging.INFO, logger="topic_analysis.service"):
        result = TopicAnalysis(FakeStorage(), tmp_path).inspect(
            {
                "taxonomy_version": TAXONOMY_VERSION,
                "prompt_version": "prompt-test",
                "model_name": "chosen",
                "cleaning_version": "clean-v2",
                "language": "en",
                "max_chars": 4000,
                "min_chars": 12,
            }
        )
    assert result["selection"] == {
        "successful_rows": 4,
        "empty_aspect_rows": 1,
        "too_short_rows": 1,
        "truncated_rows": 0,
        "usable_rows": 2,
    }
    assert result["aspect_counts"]["package_manager"] == 2
    assert result["aspect_counts"]["tooling_documentation"] == 1
    assert "inspect 开始" in caplog.text
    assert "已读取 2 条" in caplog.text
    assert "已读取 4 条" in caplog.text
    assert "inspect 完成" in caplog.text


def test_preparation_and_aspect_sampling_keep_sentiment_out_of_training_groups():
    assert parse_sentiments(
        {
            "annotations": [
                {"aspect": "package_manager", "class": "negative"},
                {"aspect": "tooling_documentation", "class": "neutral"},
            ]
        },
        1,
    ) == {"package_manager": "negative", "tooling_documentation": "neutral"}
    text, truncated = prepare_text(
        "Docs https://example.com\n```rust\nlet x=1;\n```\nCargo help", 100
    )
    assert "let x" not in text
    assert "example.com" not in text
    assert "Cargo help" in text
    assert not truncated
    documents = {
        index: TopicDocument(
            index,
            "issue",
            index,
            index,
            str(index),
            "rust-lang/cargo",
            datetime(2025, 1, 1),
            "https://github.test/issues/1",
            "package resolver",
            {"package_manager": "negative"},
        )
        for index in range(1, 6)
    }
    first = select_aspect_members(documents, aspects=("package_manager",), max_per_aspect=3, seed=5)
    second = select_aspect_members(
        documents, aspects=("package_manager",), max_per_aspect=3, seed=5
    )
    assert first == second
    assert len(first["package_manager"]) == 3


def test_rollups_use_aspect_month_and_repository_denominators():
    assignments = [
        {
            "aspect": "package_manager",
            "topic_id": 0,
            "sentiment": "negative",
            "month": "2025-01",
            "repository": "repo/a",
        },
        {
            "aspect": "package_manager",
            "topic_id": 0,
            "sentiment": "neutral",
            "month": "2025-01",
            "repository": "repo/b",
        },
        {
            "aspect": "package_manager",
            "topic_id": -1,
            "sentiment": "positive",
            "month": "2025-01",
            "repository": "repo/a",
        },
    ]
    result = build_rollups(assignments)
    topic = next(row for row in result["topics"] if row["topic_id"] == 0)
    assert topic["document_count"] == 2
    assert topic["aspect_total"] == 3
    assert topic["share_within_aspect"] == round(2 / 3, 6)
    repo = next(
        row
        for row in result["repository"]
        if row["repository"] == "repo/a" and row["topic_id"] == 0
    )
    assert repo["aspect_repository_total"] == 2
    assert repo["share_within_aspect_repository"] == 0.5


def test_fit_writes_reviewable_snapshot_without_sentiment_split(tmp_path, monkeypatch):
    class FakeStorage:
        def iter_confirmed_topic_documents(self, **_kwargs):
            yield [
                {
                    "corpus_id": index,
                    "source_type": "issue",
                    "source_id": index,
                    "parent_id": index,
                    "content_hash": str(index),
                    "clean_text": f"Package resolver discussion number {index}",
                    "annotations": {
                        "annotations": [
                            {
                                "aspect": "package_manager",
                                "class": "negative" if index % 2 else "positive",
                            }
                        ]
                    },
                    "repository": "repo/a" if index < 3 else "repo/b",
                    "created_at": datetime(2025, 1 if index < 3 else 2, 1),
                    "github_url": f"https://github.test/issues/{index}",
                }
                for index in range(1, 5)
            ]

    class AcceptArguments:
        def __init__(self, **_kwargs):
            pass

    class FakeEncoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def save(self, path):
            from pathlib import Path

            Path(path).mkdir()

        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), 3), dtype=np.float32)

    class FakeBERTopic:
        def __init__(self, **_kwargs):
            self.docs = []

        def fit_transform(self, docs, embeddings):
            assert embeddings.shape == (4, 3)
            self.docs = docs
            return [0, 0, 1, -1], None

        def get_topic(self, topic_id):
            return [(f"word-{topic_id}", 1.0)]

        def get_representative_docs(self, topic_id):
            return [self.docs[0]] if topic_id == 0 else []

        def save(self, path, **_kwargs):
            from pathlib import Path

            Path(path).write_text("fake model", encoding="utf-8")

    def module(name, **items):
        fake = types.ModuleType(name)
        fake.__dict__.update(items)
        monkeypatch.setitem(sys.modules, name, fake)

    module("torch", set_num_threads=lambda _value: None)
    module("bertopic", BERTopic=FakeBERTopic)
    module("hdbscan", HDBSCAN=AcceptArguments)
    module("sentence_transformers", SentenceTransformer=FakeEncoder)
    module("sklearn", __path__=[])
    module("sklearn.feature_extraction", __path__=[])
    module(
        "sklearn.feature_extraction.text",
        CountVectorizer=AcceptArguments,
        ENGLISH_STOP_WORDS=frozenset(),
    )
    module("umap", UMAP=AcceptArguments)
    monkeypatch.setattr("topic_analysis.service.version", lambda _name: "test")

    config = {
        "taxonomy_version": TAXONOMY_VERSION,
        "prompt_version": "prompt-test",
        "model_name": "chosen",
        "cleaning_version": "clean-v2",
        "language": "en",
        "aspects": ["package_manager"],
        "embedding_model": "fake",
        "embedding_batch_size": 2,
        "cpu_threads": 1,
        "max_chars": 4000,
        "min_chars": 12,
        "max_docs_per_aspect": None,
        "min_documents": 4,
        "min_topic_size": 2,
        "min_samples": 1,
        "umap_neighbors": 2,
        "seed": 42,
    }
    result = TopicAnalysis(FakeStorage(), tmp_path).fit(
        config, output_root=tmp_path / "data/topics", run_name="pilot"
    )
    output = tmp_path / "data/topics/pilot"
    assert result["aspect_stats"]["package_manager"]["outlier_count"] == 1
    with (output / "assignments.csv").open(encoding="utf-8-sig", newline="") as handle:
        assignments = list(csv.DictReader(handle))
    assert len(assignments) == 4
    assert {row["sentiment"] for row in assignments} == {"positive", "negative"}
    assert (output / "review.csv").exists()
    assert (output / "models/package_manager.pkl").exists()
