from datetime import datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import OperationalError

from crawler.github_client import GitHubPage, GitHubRequestError
from crawler.service import GitHubCollector, StreamFailure
from storage import Storage
from storage.models import (
    CollectionStreamRun,
    Issue,
    IssueComment,
    UnresolvedCollectionItem,
)


def issue(github_id, number, updated_at, *, pull_request=False):
    value = {
        "id": github_id,
        "number": number,
        "title": f"Title {number}",
        "body": "Body",
        "state": "open",
        "user": {"login": "u", "id": 1},
        "html_url": f"https://github.test/issues/{number}",
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": updated_at,
        "closed_at": None,
    }
    if pull_request:
        value["pull_request"] = {"url": "detail", "merged_at": None}
    return value


class FakeStorage:
    def __init__(self):
        self.issues = []
        self.prs = []
        self.advanced = []
        self.page_cursors = []

    def get_cursor(self, repository_id, stream):
        if self.page_cursors:
            return self.page_cursors[-1][2]
        return datetime(2025, 1, 2, 0, 10)

    def upsert_issues(self, repository_id, rows):
        self.issues.extend(rows)
        return len(rows)

    def upsert_pull_requests(self, repository_id, rows):
        self.prs.extend(rows)
        return len(rows)

    def advance_cursor(self, repository_id, stream, value):
        self.advanced.append((repository_id, stream, value))

    def commit_collection_page(
        self,
        repository_id,
        stream,
        cursor_updated_at,
        *,
        issues=None,
        pull_requests=None,
        **_kwargs,
    ):
        self.issues.extend(issues or [])
        self.prs.extend(pull_requests or [])
        self.page_cursors.append((repository_id, stream, cursor_updated_at))
        return len(issues or []) + len(pull_requests or [])


class SuccessfulClient:
    def __init__(self):
        self.params = None

    def paginate(self, path, params, *, max_pages=None):
        self.params = params
        yield GitHubPage(
            [
                issue(10, 1, "2025-01-03T00:00:00Z"),
                issue(999, 2, "2025-01-04T00:00:00Z", pull_request=True),
            ],
            retries=1,
        )

    def get_json(self, path):
        raise AssertionError(f"正常 PR 采集不应请求详情: {path}")


def test_issue_stream_splits_prs_and_advances_cursor_after_success():
    storage = FakeStorage()
    client = SuccessfulClient()
    collector = GitHubCollector(client, storage, cursor_overlap_seconds=300)
    stats = collector._collect_issues_and_prs(1, "rust-lang", "rust")
    assert stats.items_read == 2
    assert stats.items_written == 2
    assert stats.retries == 1
    assert len(storage.issues) == 1
    assert storage.prs[0]["github_id"] == 999
    assert storage.advanced[0][2] == datetime(2025, 1, 4)
    assert client.params["since"] == "2025-01-02T00:05:00Z"


class InterruptedClient:
    def paginate(self, path, params, *, max_pages=None):
        yield GitHubPage([issue(10, 1, "2025-01-03T00:00:00Z")], retries=0)
        raise RuntimeError("page two failed")


def test_interrupted_stream_keeps_committed_page_but_does_not_advance_cursor():
    storage = FakeStorage()
    collector = GitHubCollector(InterruptedClient(), storage)
    with pytest.raises(StreamFailure, match="page two failed") as error:
        collector._collect_issues_and_prs(1, "rust-lang", "rust")
    assert len(storage.issues) == 1
    assert storage.advanced == []
    assert storage.page_cursors[0][2] == datetime(2025, 1, 3)
    assert error.value.stream_stats.pages_read == 1


class SegmentedClient:
    def __init__(self):
        self.params = []

    def paginate(self, path, params, *, max_pages=None):
        self.params.append((dict(params), max_pages))
        if len(self.params) == 1:
            yield GitHubPage([issue(10, 1, "2025-01-03T00:00:00Z")], retries=0)
            yield GitHubPage([issue(11, 2, "2025-01-04T00:00:00Z")], retries=0)
        else:
            yield GitHubPage([issue(12, 3, "2025-01-05T00:00:00Z")], retries=0)


def test_full_segment_restarts_from_durable_cursor():
    storage = FakeStorage()
    client = SegmentedClient()
    collector = GitHubCollector(client, storage, segment_page_limit=2)
    stats = collector._collect_issues_and_prs(1, "rust-lang", "rust")
    assert stats.pages_read == 3
    assert len(client.params) == 2
    assert client.params[0][1] == 2
    assert client.params[1][0]["since"] == "2025-01-03T23:55:00Z"
    assert storage.advanced[0][2] == datetime(2025, 1, 5)


def comment(github_id, parent, updated_at="2025-01-03T00:00:00Z"):
    return {
        "id": github_id,
        "issue_url": f"https://api.github.test/repos/rust-lang/rust/issues/{parent}",
        "html_url": f"https://github.test/comment/{github_id}",
        "body": "Comment",
        "user": {"login": "u", "id": 1},
        "created_at": updated_at,
        "updated_at": updated_at,
    }


@pytest.fixture
def db_storage():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    value = Storage("", engine=engine)
    value.create_schema()
    yield value
    engine.dispose()


class MissingParentClient:
    def __init__(self, *, parent_two_available=False, empty_page=False):
        self.parent_two_available = parent_two_available
        self.empty_page = empty_page

    def paginate(self, path, params, *, max_pages=None):
        items = [] if self.empty_page else [comment(100, 1), comment(101, 2)]
        yield GitHubPage(items, retries=0)

    def get_json(self, path):
        number = int(path.rsplit("/", 1)[-1])
        if number == 2 and not self.parent_two_available:
            raise GitHubRequestError("not found", status_code=404)
        return issue(200 + number, number, "2025-01-02T00:00:00Z"), 0


def test_missing_parent_is_supplemented_or_quarantined_then_reconciled(db_storage):
    repository_id = db_storage.ensure_repository("rust-lang/rust")
    first = GitHubCollector(MissingParentClient(), db_storage)
    stats = first._collect_issue_comments(repository_id, "rust-lang", "rust")
    assert stats.items_written == 1
    with db_storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(IssueComment)) == 1
        pending = session.scalar(
            select(func.count()).select_from(UnresolvedCollectionItem).where(
                UnresolvedCollectionItem.resolved_at.is_(None)
            )
        )
        assert pending == 1

    second = GitHubCollector(
        MissingParentClient(parent_two_available=True, empty_page=True), db_storage
    )
    second._collect_issue_comments(repository_id, "rust-lang", "rust")
    with db_storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(IssueComment)) == 2
        pending = session.scalar(
            select(func.count()).select_from(UnresolvedCollectionItem).where(
                UnresolvedCollectionItem.resolved_at.is_(None)
            )
        )
        assert pending == 0


class InvalidRecordClient:
    def paginate(self, path, params, *, max_pages=None):
        invalid = issue(11, 2, "2025-01-04T00:00:00Z")
        invalid.pop("created_at")
        yield GitHubPage(
            [issue(10, 1, "2025-01-03T00:00:00Z"), invalid], retries=0
        )


def test_invalid_record_is_quarantined_without_stopping_page(db_storage):
    repository_id = db_storage.ensure_repository("rust-lang/rust")
    collector = GitHubCollector(InvalidRecordClient(), db_storage)
    stats = collector._collect_issues_and_prs(repository_id, "rust-lang", "rust")
    assert stats.items_written == 1
    assert db_storage.get_cursor(repository_id, "issues_and_pull_requests") == datetime(
        2025, 1, 4
    )
    with db_storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Issue)) == 1
        invalid_count = session.scalar(
            select(func.count()).select_from(UnresolvedCollectionItem).where(
                UnresolvedCollectionItem.category == "invalid_record",
                UnresolvedCollectionItem.resolved_at.is_(None),
            )
        )
        assert invalid_count == 1


def test_transient_database_failure_retries_same_page(db_storage):
    repository_id = db_storage.ensure_repository("rust-lang/rust")
    original = db_storage.commit_collection_page
    attempts = 0

    def flaky_commit(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OperationalError("insert", {}, RuntimeError("deadlock"))
        return original(*args, **kwargs)

    db_storage.commit_collection_page = flaky_commit
    sleeps = []
    collector = GitHubCollector(
        SuccessfulClient(),
        db_storage,
        page_write_max_retries=1,
        sleep=sleeps.append,
    )
    stats = collector._collect_issues_and_prs(repository_id, "rust-lang", "rust")
    assert stats.items_written == 2
    assert attempts == 2
    assert sleeps == [1.0]


class AlwaysFailClient:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def paginate(self, path, params, *, max_pages=None):
        self.calls += 1
        raise self.error
        yield  # pragma: no cover


def test_parent_stream_failure_skips_dependent_comment_streams(db_storage):
    run_id = db_storage.start_pipeline_run("collect")
    client = AlwaysFailClient(RuntimeError("temporary page failure"))
    collector = GitHubCollector(client, db_storage)
    result = collector.collect_repository("rust-lang/rust", run_id)
    assert result["issues_and_pull_requests"]["status"] == "failed"
    assert result["issue_comments"]["status"] == "skipped_dependency"
    assert result["review_comments"]["status"] == "skipped_dependency"
    assert client.calls == 1


def test_authentication_failure_aborts_after_recording_stream(db_storage):
    run_id = db_storage.start_pipeline_run("collect")
    client = AlwaysFailClient(GitHubRequestError("bad credentials", status_code=401))
    collector = GitHubCollector(client, db_storage)
    with pytest.raises(StreamFailure, match="bad credentials"):
        collector.collect_repository("rust-lang/rust", run_id)
    with db_storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(CollectionStreamRun)) == 1
