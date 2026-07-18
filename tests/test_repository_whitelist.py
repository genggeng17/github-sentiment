import pytest
from sqlalchemy import create_engine

import pipeline as pipeline_module
from config import Settings, normalize_repository_name
from pipeline import Pipeline, build_parser
from storage import Storage


@pytest.fixture
def storage():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    value = Storage("", engine=engine)
    value.create_schema()
    yield value
    engine.dispose()


def test_bootstrap_only_imports_when_repository_table_is_empty(storage):
    assert storage.bootstrap_repositories(["rust-lang/rust", "tokio-rs/tokio"]) == 2
    assert storage.bootstrap_repositories(["serde-rs/serde"]) == 0
    assert storage.enabled_repository_names() == ["rust-lang/rust", "tokio-rs/tokio"]


def test_disable_preserves_repository_and_removes_it_from_enabled_list(storage):
    repository_id = storage.ensure_repository("rust-lang/rust")
    assert storage.set_repository_enabled("rust-lang/rust", False) is True
    assert storage.enabled_repository_names() == []
    rows = storage.list_repositories()
    assert rows[0]["id"] == repository_id
    assert rows[0]["full_name"] == "rust-lang/rust"
    assert rows[0]["enabled"] is False
    assert storage.set_repository_enabled("rust-lang/rust", False) is False


def test_adding_disabled_repository_reenables_it(storage):
    storage.ensure_repository("rust-lang/rust")
    storage.set_repository_enabled("rust-lang/rust", False)
    storage.ensure_repository("rust-lang/rust")
    assert storage.enabled_repository_names() == ["rust-lang/rust"]


def test_unknown_repository_cannot_be_enabled(storage):
    with pytest.raises(ValueError, match="不在白名单"):
        storage.set_repository_enabled("rust-lang/rust", True)


@pytest.mark.parametrize("value", ["rust", "/rust", "rust-lang/", "a/b/c", ""])
def test_repository_name_requires_owner_and_repo(value):
    with pytest.raises(ValueError, match="owner/repo"):
        normalize_repository_name(value)


def test_repository_cli_parses_management_commands():
    args = build_parser().parse_args(["repo", "disable", "rust-lang/rust"])
    assert args.command == "repo"
    assert args.repo_command == "disable"
    assert args.full_name == "rust-lang/rust"


def test_collection_uses_enabled_database_repositories_not_environment(monkeypatch):
    collected = []

    class FakeStorage:
        def enabled_repository_names(self):
            return ["database/repository"]

        def heartbeat(self, run_id):
            pass

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class FakeCollector:
        def __init__(self, *args, **kwargs):
            pass

        def collect_repository(self, full_name, run_id, heartbeat):
            collected.append(full_name)
            return {"issues": {"status": "succeeded"}}

    monkeypatch.setattr(pipeline_module, "GitHubClient", FakeClient)
    monkeypatch.setattr(pipeline_module, "GitHubCollector", FakeCollector)
    settings = Settings(
        database_url="sqlite://",
        github_token="token",
        repositories=("environment/repository",),
    )
    result = Pipeline(settings, FakeStorage()).collect("run-id")
    assert collected == ["database/repository"]
    assert result["failed_streams"] == 0


def test_run_skips_downstream_steps_when_collection_is_partial(monkeypatch):
    settings = Settings(
        database_url="sqlite://",
        github_token="token",
        repositories=(),
    )
    value = Pipeline(settings, object())
    monkeypatch.setattr(value, "collect", lambda _run_id: {"failed_streams": 1})
    monkeypatch.setattr(
        value,
        "build_corpus",
        lambda: pytest.fail("partial collection must not build corpus"),
    )
    monkeypatch.setattr(
        value,
        "label",
        lambda: pytest.fail("partial collection must not start labeling"),
    )
    assert value.run_all("run-id") == {"collection": {"failed_streams": 1}}
