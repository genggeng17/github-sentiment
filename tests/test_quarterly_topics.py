import json
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine, select

from corpus.builder import make_corpus_row
from storage import Storage
from storage.models import LlmAnnotation
from topic_analysis import quarterly
from topic_analysis.__main__ import main
from topic_analysis.quarterly import QuarterlyTopicAnalysis, quarter_bounds


def test_latest_nonempty_reader_across_versions_filters_before_ranking():
    storage = Storage("", engine=create_engine("sqlite+pysqlite:///:memory:"))
    storage.create_schema()
    repository_id = storage.ensure_repository("owner/repo")
    now = datetime(2025, 4, 1)
    storage.upsert_issues(
        repository_id,
        [
            {
                "github_id": 123,
                "number": 1,
                "title": "Quarter boundary",
                "body": "body",
                "state": "open",
                "author_login": "a",
                "author_github_id": 1,
                "github_url": "https://github.test/123",
                "created_at": now,
                "updated_at": now,
                "collected_at": now,
            }
        ],
    )
    corpus = make_corpus_row(
        {
            "source_type": "issue",
            "source_id": 1,
            "title": "",
            "body": "Workspace features and dependency version resolution",
            "source_updated_at": now,
        }
    )
    storage.upsert_corpus([corpus])
    corpus_id = storage.get_corpus_id("issue", 1, corpus["content_hash"])

    def add(model, labels, updated_at, status="succeeded"):
        with storage.sessions.begin() as session:
            item = LlmAnnotation(
                corpus_id=corpus_id,
                model_name=model,
                taxonomy_version=f"taxonomy-{model}",
                prompt_version=f"prompt-{model}",
                parsed_result={"annotations": labels},
                status=status,
                updated_at=updated_at,
            )
            session.add(item)
            session.flush()
            return item.id

    label = [
        {"aspect": "historical_aspect", "class": "neutral"},
        {"aspect": "package_manager", "class": "positive"},
    ]
    add("old", label, now)
    winning_id = add("chosen", label, now + timedelta(days=2))
    add("higher-id-older-time", label, now + timedelta(days=1))
    add("empty-newest", [], now + timedelta(days=4))
    add("blank-aspect", [{"aspect": "  ", "class": "neutral"}], now + timedelta(days=5))
    add("failed-newest", label, now + timedelta(days=6), "failed")
    rows = [
        row for batch in storage.iter_latest_nonempty_topic_documents(batch_size=1) for row in batch
    ]
    assert len(rows) == 1
    assert rows[0]["annotation_id"] == winning_id
    assert rows[0]["model_name"] == "chosen"
    assert rows[0]["annotations"] == label
    assert rows[0]["created_at"] == now
    assert rows[0]["created_at"] != rows[0]["annotation_updated_at"]
    tie_id = add("tied-time-larger-id", label, now + timedelta(days=2))
    assert next(storage.iter_latest_nonempty_topic_documents())[0]["annotation_id"] == tie_id
    with storage.sessions() as session:
        assert len(session.scalars(select(LlmAnnotation)).all()) == 7


def test_quarter_boundaries_are_natural_calendar_and_utc():
    assert quarter_bounds(datetime(2025, 3, 31, 23, 59, 59))[0] == "2025-Q1"
    assert quarter_bounds(datetime(2025, 4, 1))[0] == "2025-Q2"
    assert quarter_bounds(datetime(2025, 12, 31))[2] == datetime(2026, 1, 1)
    assert quarter_bounds(datetime(2025, 4, 1, tzinfo=timezone(timedelta(hours=8))))[0] == "2025-Q1"
    assert quarter_bounds(datetime(2025, 4, 1, tzinfo=UTC))[0] == "2025-Q2"


class FakeStorage:
    def iter_latest_nonempty_topic_documents(self):
        rows = []
        for index in range(1, 15):
            month = 1 if index <= 6 else 4 if index <= 12 else 10
            rows.append(
                {
                    "corpus_id": index,
                    "annotation_id": index + 100,
                    "source_type": "issue",
                    "source_id": index,
                    "parent_id": None,
                    "content_hash": str(index),
                    "repository": "owner/repo",
                    "created_at": datetime(2025, month, 1),
                    "annotation_updated_at": datetime(2026, 1, 1),
                    "github_url": f"https://github.test/{index}",
                    "clean_text": "Hi"
                    if index == 14
                    else f"Dependency resolution discussion {index}",
                    "model_name": "model-a" if index % 2 else "model-b",
                    "taxonomy_version": "old-taxonomy",
                    "prompt_version": "old-prompt",
                    "cleaning_version": "clean-v1",
                    "language": "en",
                    "annotations": [
                        {"aspect": "package_manager", "class": "negative"},
                        {"aspect": "type_system", "class": "positive"},
                    ],
                }
            )
        yield rows[:7]
        yield rows[7:]


def config():
    return dict(
        max_chars=4000,
        min_chars=12,
        embedding_model="fake-encoder",
        embedding_batch_size=8,
        cpu_threads=1,
        min_documents=6,
        min_topic_size=2,
        min_samples=1,
        umap_neighbors=3,
        seed=42,
    )


def test_quarterly_snapshot_fit_failure_resume_and_pooled_assignments(tmp_path, monkeypatch):
    calls = {"encode": [], "cluster": [], "fail": True}

    class Backend:
        def __init__(self, _config, destination):
            (destination / "encoder").mkdir(exist_ok=True)

        def encode(self, texts, _config):
            calls["encode"].append(texts[0])
            return np.ones((len(texts), 4), dtype=np.float32)

        def cluster(self, texts, embeddings, _config):
            assert embeddings.shape == (6, 4)
            calls["cluster"].append(texts[0])
            if texts[0].endswith("7") and calls["fail"]:
                calls["fail"] = False
                raise RuntimeError("interrupted second quarter")

            class Model:
                def save(self, filename, **_kwargs):
                    from pathlib import Path

                    Path(filename).write_bytes(b"model")

                def get_topic(self, _topic):
                    return [("dependency", 0.8), ("resolution", 0.2)]

                def get_representative_docs(self, _topic):
                    return [texts[0]]

            return Model(), [0, 0, 0, 1, 1, -1], np.full(6, 0.75)

    monkeypatch.setattr(quarterly, "_Backend", Backend)
    monkeypatch.setattr(quarterly, "version", lambda _name: "test")
    analysis = QuarterlyTopicAnalysis(FakeStorage(), tmp_path)
    root = tmp_path / "data/topics"
    inspected = analysis.inspect(config(), output_root=root, snapshot_name="source")
    assert inspected["selection"]["usable_rows"] == 13
    assert inspected["selection"]["too_short"] == 1
    snapshot = root / "snapshots/source"
    source_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert source_manifest["quarters"]["2025-Q3"]["document_count"] == 0
    assert source_manifest["quarters"]["2025-Q1"]["document_count"] == 6
    assert {v["model_name"] for v in source_manifest["versions"]} == {"model-a", "model-b"}
    with pytest.raises(RuntimeError, match="interrupted"):
        analysis.fit(config(), output_root=root, run_name="run", snapshot_name="source")
    run = root / "quarterly/run"
    assert not (root / "quarterly/latest.json").exists()
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    first_checkpoint = (run / "quarters/2025-Q1/manifest.json").read_bytes()
    result = analysis.fit(
        config(), output_root=root, run_name="run", snapshot_name="source", resume=True
    )
    assert result["trained_quarters"] == 2
    assert result["insufficient_quarters"] == 2
    assert (run / "quarters/2025-Q1/manifest.json").read_bytes() == first_checkpoint
    assert len(calls["encode"]) == 2  # Second-quarter embeddings survive its failed clustering.
    assert len(calls["cluster"]) == 3  # The completed first quarter is never retrained.
    assignments = [
        json.loads(line) for line in (run / "assignments.jsonl").read_text().splitlines()
    ]
    assert len(assignments) == len({a["corpus_id"] for a in assignments}) == 13
    assert sum(a["assignment_status"] == "outlier" for a in assignments) == 2
    assert assignments[-1]["assignment_status"] == "insufficient"
    assert assignments[-1]["topic_id"] is None
    assert all(len(a["aspects"]) == 2 for a in assignments)
    topics = json.loads((run / "topics.json").read_text(encoding="utf-8"))["topics"]
    assert len(topics) == 6
    assert topics[1]["share_within_quarter"] == 0.5
    assert json.loads((root / "quarterly/latest.json").read_text())["run_name"] == "run"
    with pytest.raises(ValueError, match="配置或输入"):
        analysis.fit(
            {**config(), "seed": 1},
            output_root=root,
            run_name="run",
            snapshot_name="source",
            resume=True,
        )
    damaged = run / "quarters/2025-Q1/topics.json"
    damaged.write_text("{}")
    with pytest.raises(ValueError, match="产物损坏"):
        analysis.fit(
            config(), output_root=root, run_name="run", snapshot_name="source", resume=True
        )


def test_quarterly_rejects_corrupt_snapshot_before_training(tmp_path):
    analysis = QuarterlyTopicAnalysis(FakeStorage(), tmp_path)
    root = tmp_path / "data/topics"
    analysis.inspect(config(), output_root=root, snapshot_name="source")
    file = root / "snapshots/source/quarters/2025-Q1.jsonl"
    with file.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="校验失败"):
        analysis.fit(config(), output_root=root, run_name="run", snapshot_name="source")


def test_cli_defaults_to_quarterly_and_rejects_ignored_version_filters():
    from topic_analysis.__main__ import build_parser

    assert build_parser().parse_args(["inspect"]).__dict__["mode"] == "quarterly"
    with pytest.raises(SystemExit):
        main(["inspect", "--model-name", "chosen"])
    with pytest.raises(SystemExit):
        main(["inspect", "--mode", "aspect"])
