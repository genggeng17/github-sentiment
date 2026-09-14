import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine, select

from corpus_builder import CLEANING_VERSION, CorpusBuilder
from lexicon_sampling import LexiconSampler, parse_quotas
from lexicon_sampling.matcher import LexiconMatcher
from pipeline import build_parser
from sampler import CorpusSampler
from storage import Storage
from storage.models import Corpus, CorpusSampleItem, LexiconSampleHit


@pytest.fixture
def dataset(tmp_path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    storage = Storage("", engine=engine)
    storage.create_schema()
    bodies = [
        "trait cargo",
        "trait only",
        "cargo only",
        "unrelated",
        "> trait cargo",
        "```\ntrait cargo\n```",
    ]
    for repo_number, texts in enumerate((bodies, ["another trait", "another cargo"])):
        repository_id = storage.ensure_repository(f"example/repo{repo_number}")
        rows = []
        for index, body in enumerate(texts, 1):
            rows.append(
                {
                    "github_id": repo_number * 100 + index,
                    "number": index,
                    "title": f"Item {index}",
                    "body": body,
                    "state": "open",
                    "author_login": "person",
                    "author_github_id": 1,
                    "github_url": f"https://github.test/{repo_number}/{index}",
                    "created_at": datetime(2026, 1, 1),
                    "updated_at": datetime(2026, 1, 1),
                    "collected_at": datetime(2026, 1, 2),
                    "closed_at": None,
                }
            )
        storage.upsert_issues(repository_id, rows)
    CorpusBuilder(storage).build()
    definition = {
        "schema_version": 1,
        "version": "test-v1",
        "aspects": {
            "type_system": {"rules": [{"id": "types", "all": [r"\btrait\b"]}]},
            "package_manager": {"rules": [{"id": "cargo", "all": [r"\bcargo\b"]}]},
        },
    }
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    yield storage, path, definition
    engine.dispose()


def build(dataset, name="targeted", **kwargs):
    storage, path, _ = dataset
    options = dict(
        lexicon_path=str(path),
        quotas={"type_system": 100, "package_manager": 100},
        cleaning_version=CLEANING_VERSION,
        seed="fixed",
    )
    options.update(kwargs)
    return LexiconSampler(storage).build(name, **options)


def test_union_keeps_aspect_memberships_and_labels_only_selected_set(dataset):
    storage, _, _ = dataset
    stats = build(dataset)
    assert stats["selected"] == 5
    assert stats["overlap_selections"] == 1
    assert stats["aspects"]["type_system"] == {
        "requested": 100,
        "eligible": 3,
        "selected": 3,
        "shortage": 97,
    }
    with storage.sessions() as session:
        selected = set(session.scalars(select(CorpusSampleItem.corpus_id)))
        hits = session.scalars(select(LexiconSampleHit)).all()
        assert len(hits) == 6
        for hit in hits:
            corpus = session.get(Corpus, hit.corpus_id)
            assert hit.selected_for_quota
            for evidence in hit.evidence:
                assert corpus.clean_text[evidence["start"] : evidence["end"]] == evidence["match"]
    pending = {
        row["id"]
        for batch in storage.iter_unannotated_corpus(
            "rust-aspects-v2",
            "aspect-sentiment-zh-v8",
            "test-model",
            100,
            sample_set_id=stats["sample_set_id"],
        )
        for row in batch
    }
    assert pending == selected
    assert build(dataset)["reused"] is True


def test_per_aspect_quota_repository_cap_and_determinism(dataset):
    storage, _, _ = dataset
    options = {"quotas": {"type_system": 2, "package_manager": 1}, "per_repository_limit": 1}
    first = build(dataset, "first", **options)
    second = build(dataset, "second", **options)
    assert first["aspects"]["type_system"]["selected"] == 2
    assert first["aspects"]["package_manager"]["selected"] == 1
    with storage.sessions() as session:

        def ids(sample_id):
            return set(
                session.scalars(
                    select(CorpusSampleItem.corpus_id).where(
                        CorpusSampleItem.sample_set_id == sample_id
                    )
                )
            )

        assert ids(first["sample_set_id"]) == ids(second["sample_set_id"])
        rows = (
            session.execute(
                select(CorpusSampleItem.repository_id)
                .join(
                    LexiconSampleHit,
                    (LexiconSampleHit.sample_set_id == CorpusSampleItem.sample_set_id)
                    & (LexiconSampleHit.corpus_id == CorpusSampleItem.corpus_id),
                )
                .where(
                    LexiconSampleHit.sample_set_id == first["sample_set_id"],
                    LexiconSampleHit.candidate_aspect == "type_system",
                    LexiconSampleHit.selected_for_quota.is_(True),
                )
            )
            .scalars()
            .all()
        )
        assert len(set(rows)) == 2


def test_aspect_specific_filters_and_shortage(dataset):
    _, path, definition = dataset
    definition["aspects"]["type_system"]["filters"] = {"repository_names": ["example/repo1"]}
    definition["aspects"]["package_manager"]["filters"] = {"min_target_chars": 1000}
    path.write_text(json.dumps(definition), encoding="utf-8")
    stats = build(dataset)
    assert stats["selected"] == 1
    assert stats["aspects"]["type_system"]["selected"] == 1
    assert stats["aspects"]["package_manager"]["shortage"] == 100


def test_changed_dictionary_and_random_name_collision_are_rejected(dataset):
    storage, path, definition = dataset
    build(dataset)
    with pytest.raises(ValueError, match="参数不同"):
        CorpusSampler(storage).build(
            "targeted",
            per_repository_limit=200,
            cleaning_version=CLEANING_VERSION,
            seed="fixed",
            max_model_input_chars=4000,
        )
    definition["aspects"]["type_system"]["rules"][0]["all"] = ["something else"]
    path.write_text(json.dumps(definition), encoding="utf-8")
    with pytest.raises(ValueError, match="参数不同"):
        build(dataset)


def test_partial_write_failure_is_not_labelable_and_retry_is_clean(dataset, monkeypatch):
    storage, _, _ = dataset
    original = storage.insert_lexicon_hits

    def fail(rows):
        original(rows[:1])
        raise RuntimeError("interrupted write")

    monkeypatch.setattr(storage, "insert_lexicon_hits", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        build(dataset)
    with pytest.raises(ValueError, match="尚未构建完成"):
        storage.completed_sample_set_id("targeted")
    monkeypatch.setattr(storage, "insert_lexicon_hits", original)
    stats = build(dataset)
    assert stats["selected"] == 5
    with storage.sessions() as session:
        assert len(session.scalars(select(LexiconSampleHit)).all()) == 6


def test_matcher_masks_quotes_and_code_preserving_unicode_offsets(dataset):
    matcher = LexiconMatcher(dataset[2], {"type_system": 1, "package_manager": 1})
    assert matcher.match("> trait\r\n```\r\ncargo\r\n```\r\n") == {}
    text = "中文🙂 trait and cargo"
    hits = matcher.match(text)
    assert set(hits) == {"type_system", "package_manager"}
    assert all(text[e["start"] : e["end"]] == e["match"] for group in hits.values() for e in group)


def test_cli_requires_quotas_and_preserves_all_four_aspects():
    args = build_parser().parse_args(
        [
            "sample-lexicon",
            "--name",
            "four",
            "--lexicon",
            "rules.json",
            "--cleaning-version",
            "clean-v1",
            "--quota",
            "libraries_frameworks=100",
            "--quota",
            "learning_curve=200",
            "--quota",
            "package_manager=300",
            "--quota",
            "type_system=400",
        ]
    )
    assert parse_quotas(args.quota) == {
        "libraries_frameworks": 100,
        "learning_curve": 200,
        "package_manager": 300,
        "type_system": 400,
    }
    with pytest.raises(ValueError):
        parse_quotas(["type_system=1", "type_system=2"])
    with pytest.raises(ValueError):
        parse_quotas(["type_system=0"])
