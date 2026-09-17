from datetime import datetime

import pytest
from sqlalchemy import create_engine, func, select, update

from corpus import CLEANING_VERSION, CorpusBuilder, CorpusSampler
from pipeline import build_parser
from storage import Storage
from storage.models import Corpus, CorpusSampleItem, CorpusSampleSet, Issue


@pytest.fixture
def storage():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    value = Storage("", engine=engine)
    value.create_schema()
    yield value
    engine.dispose()


def issue_row(github_id: int, number: int, body: str) -> dict:
    return {
        "github_id": github_id,
        "number": number,
        "title": f"Issue {number}",
        "body": body,
        "state": "open",
        "author_login": "user",
        "author_github_id": github_id,
        "github_url": f"https://github.test/issues/{number}",
        "created_at": datetime(2025, 1, 1),
        "updated_at": datetime(2025, 1, 2),
        "collected_at": datetime(2025, 1, 3),
        "closed_at": None,
    }


def seed_repositories(storage: Storage) -> tuple[int, int]:
    first = storage.ensure_repository("example/first")
    second = storage.ensure_repository("example/second")
    storage.upsert_issues(
        first,
        [issue_row(100 + number, number, f"First body {number}") for number in range(1, 6)],
    )
    storage.upsert_issues(
        second,
        [issue_row(200 + number, number, f"Second body {number}") for number in range(1, 4)],
    )
    CorpusBuilder(storage).build()
    return first, second


def test_sample_is_capped_per_repository_and_is_reusable(storage):
    first, second = seed_repositories(storage)
    sampler = CorpusSampler(storage)
    stats = sampler.build("rust-v1", per_repository_limit=2, seed="fixed")
    assert stats["selected"] == 4
    assert stats["cleaning_version"] == CLEANING_VERSION
    assert stats["repositories"]["example/first"] == {"eligible": 5, "selected": 2}
    assert stats["repositories"]["example/second"] == {"eligible": 3, "selected": 2}

    with storage.sessions() as session:
        counts = dict(
            session.execute(
                select(
                    CorpusSampleItem.repository_id,
                    func.count(CorpusSampleItem.corpus_id),
                ).group_by(CorpusSampleItem.repository_id)
            ).all()
        )
        selected_ids = set(session.scalars(select(CorpusSampleItem.corpus_id)))
        sample_set = session.scalar(
            select(CorpusSampleSet).where(CorpusSampleSet.name == "rust-v1")
        )
    assert counts == {first: 2, second: 2}
    assert sample_set.status == "completed"
    assert sample_set.cleaning_version == CLEANING_VERSION
    assert sample_set.max_model_input_chars is None

    with storage.sessions.begin() as session:
        sample_set = session.scalar(
            select(CorpusSampleSet).where(CorpusSampleSet.name == "rust-v1")
        )
        sample_set.cleaning_version = None

    reused = sampler.build("rust-v1", per_repository_limit=2, seed="fixed")
    assert reused["reused"] is True
    with storage.sessions() as session:
        assert set(session.scalars(select(CorpusSampleItem.corpus_id))) == selected_ids
        sample_set = session.scalar(
            select(CorpusSampleSet).where(CorpusSampleSet.name == "rust-v1")
        )
        assert sample_set.cleaning_version == CLEANING_VERSION


def test_sample_scans_all_repositories_in_one_stream(storage, monkeypatch):
    first, second = seed_repositories(storage)
    original = storage.iter_sample_candidates
    calls = []

    def tracked(repository_ids, **kwargs):
        calls.append(tuple(repository_ids))
        yield from original(repository_ids, **kwargs)

    monkeypatch.setattr(storage, "iter_sample_candidates", tracked)
    CorpusSampler(storage).build(
        "rust-one-pass",
        per_repository_limit=2,
        seed="fixed",
    )

    assert calls == [(first, second)]


def test_sample_name_is_immutable(storage):
    seed_repositories(storage)
    sampler = CorpusSampler(storage)
    sampler.build("rust-v1", per_repository_limit=2, seed="fixed")
    with pytest.raises(ValueError, match="参数不同"):
        sampler.build("rust-v1", per_repository_limit=3, seed="fixed")


def test_sample_name_is_immutable_when_length_limit_changes(storage):
    seed_repositories(storage)
    sampler = CorpusSampler(storage)
    sampler.build(
        "rust-v1",
        per_repository_limit=2,
        seed="fixed",
        max_model_input_chars=4000,
    )
    with pytest.raises(ValueError, match="参数不同"):
        sampler.build(
            "rust-v1",
            per_repository_limit=2,
            seed="fixed",
            max_model_input_chars=8000,
        )


def test_sample_name_is_immutable_when_cleaning_version_changes(storage):
    seed_repositories(storage)
    with storage.sessions.begin() as session:
        oldest_id = session.scalar(select(func.min(Corpus.id)))
        session.execute(
            update(Corpus)
            .where(Corpus.id == oldest_id)
            .values(cleaning_version="clean-v1")
        )
    sampler = CorpusSampler(storage)
    sampler.build("rust-v1", per_repository_limit=2, seed="fixed")
    with pytest.raises(ValueError, match="参数不同"):
        sampler.build(
            "rust-v1",
            per_repository_limit=2,
            seed="fixed",
            cleaning_version="clean-v1",
        )


def test_sample_excludes_candidates_over_model_input_length_limit(storage):
    seed_repositories(storage)
    with storage.sessions.begin() as session:
        longest_id = session.scalar(select(func.max(Corpus.id)))
        session.execute(
            update(Corpus)
            .where(Corpus.id == longest_id)
            .values(model_input="x" * 401, model_input_chars=401)
        )

    stats = CorpusSampler(storage).build(
        "rust-max-400",
        per_repository_limit=100,
        seed="fixed",
        max_model_input_chars=400,
    )

    assert stats["max_model_input_chars"] == 400
    assert stats["eligible"] == 7
    assert stats["selected"] == 7
    with storage.sessions() as session:
        selected = set(session.scalars(select(CorpusSampleItem.corpus_id)))
    assert longest_id not in selected


def test_sample_cli_accepts_model_input_length_limit():
    args = build_parser().parse_args(
        [
            "sample",
            "--name",
            "rust-max-4000",
            "--max-model-input-chars",
            "4000",
        ]
    )
    assert args.max_model_input_chars == 4000


def test_sample_cli_accepts_cleaning_version():
    args = build_parser().parse_args(
        ["sample", "--name", "rust-clean-v1", "--cleaning-version", "clean-v1"]
    )
    assert args.cleaning_version == "clean-v1"


def test_run_cli_accepts_sample_cleaning_version():
    args = build_parser().parse_args(
        ["run", "--sample-cleaning-version", "clean-v1"]
    )
    assert args.sample_cleaning_version == "clean-v1"


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_sample_cli_rejects_invalid_model_input_length_limit(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "sample",
                "--name",
                "rust-invalid",
                "--max-model-input-chars",
                value,
            ]
        )


def test_annotation_iterator_only_returns_selected_corpus(storage):
    seed_repositories(storage)
    stats = CorpusSampler(storage).build(
        "rust-v1",
        per_repository_limit=1,
        seed="fixed",
    )
    batches = list(
        storage.iter_unannotated_corpus(
            "taxonomy",
            "prompt",
            "model",
            100,
            sample_set_id=stats["sample_set_id"],
        )
    )
    returned = {row["id"] for batch in batches for row in batch}
    with storage.sessions() as session:
        selected = set(session.scalars(select(CorpusSampleItem.corpus_id)))
        assert session.scalar(select(func.count()).select_from(Issue)) == 8
    assert returned == selected


def test_new_sample_only_uses_current_cleaning_version(storage):
    seed_repositories(storage)
    with storage.sessions.begin() as session:
        oldest_id = session.scalar(select(func.min(Corpus.id)))
        session.execute(
            update(Corpus)
            .where(Corpus.id == oldest_id)
            .values(cleaning_version="clean-v1")
        )

    stats = CorpusSampler(storage).build(
        "rust-clean-v2",
        per_repository_limit=100,
        seed="fixed",
    )
    assert stats["eligible"] == 7
    assert stats["selected"] == 7


def test_sample_can_select_an_existing_cleaning_version(storage):
    seed_repositories(storage)
    with storage.sessions.begin() as session:
        oldest_id = session.scalar(select(func.min(Corpus.id)))
        session.execute(
            update(Corpus)
            .where(Corpus.id == oldest_id)
            .values(cleaning_version="clean-v1")
        )

    stats = CorpusSampler(storage).build(
        "rust-clean-v1",
        per_repository_limit=100,
        seed="fixed",
        cleaning_version="clean-v1",
    )

    assert stats["cleaning_version"] == "clean-v1"
    assert stats["eligible"] == 1
    assert stats["selected"] == 1
    with storage.sessions() as session:
        selected_versions = set(
            session.scalars(
                select(Corpus.cleaning_version)
                .join(CorpusSampleItem, CorpusSampleItem.corpus_id == Corpus.id)
                .join(
                    CorpusSampleSet,
                    CorpusSampleSet.id == CorpusSampleItem.sample_set_id,
                )
                .where(CorpusSampleSet.name == "rust-clean-v1")
            )
        )
    assert selected_versions == {"clean-v1"}


def test_sample_rejects_unknown_cleaning_version(storage):
    seed_repositories(storage)
    with pytest.raises(ValueError, match="可用版本"):
        CorpusSampler(storage).build(
            "rust-unknown",
            cleaning_version="clean-v999",
        )
