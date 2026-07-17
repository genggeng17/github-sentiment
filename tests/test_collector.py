from datetime import datetime

import pytest

from crawler.github_client import GitHubPage
from crawler.service import GitHubCollector, StreamFailure


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
        value["pull_request"] = {"url": "detail"}
    return value


class FakeStorage:
    def __init__(self):
        self.issues = []
        self.prs = []
        self.advanced = []

    def get_cursor(self, repository_id, stream):
        return datetime(2025, 1, 2, 0, 10)

    def upsert_issues(self, repository_id, rows):
        self.issues.extend(rows)
        return len(rows)

    def upsert_pull_requests(self, repository_id, rows):
        self.prs.extend(rows)
        return len(rows)

    def advance_cursor(self, repository_id, stream, value):
        self.advanced.append((repository_id, stream, value))


class SuccessfulClient:
    def __init__(self):
        self.params = None

    def paginate(self, path, params):
        self.params = params
        yield GitHubPage(
            [
                issue(10, 1, "2025-01-03T00:00:00Z"),
                issue(999, 2, "2025-01-04T00:00:00Z", pull_request=True),
            ],
            retries=1,
        )

    def get_json(self, path):
        return {
            **issue(20, 2, "2025-01-04T00:00:00Z"),
            "merged_at": None,
        }, 2


def test_issue_stream_splits_prs_and_advances_cursor_after_success():
    storage = FakeStorage()
    client = SuccessfulClient()
    collector = GitHubCollector(client, storage, cursor_overlap_seconds=300)
    stats = collector._collect_issues_and_prs(1, "rust-lang", "rust")
    assert stats.items_read == 2
    assert stats.items_written == 2
    assert stats.retries == 3
    assert len(storage.issues) == 1
    assert storage.prs[0]["github_id"] == 20
    assert storage.advanced[0][2] == datetime(2025, 1, 4)
    assert client.params["since"] == "2025-01-02T00:05:00Z"


class InterruptedClient:
    def paginate(self, path, params):
        yield GitHubPage([issue(10, 1, "2025-01-03T00:00:00Z")], retries=0)
        raise RuntimeError("page two failed")


def test_interrupted_stream_keeps_committed_page_but_does_not_advance_cursor():
    storage = FakeStorage()
    collector = GitHubCollector(InterruptedClient(), storage)
    with pytest.raises(StreamFailure, match="page two failed") as error:
        collector._collect_issues_and_prs(1, "rust-lang", "rust")
    assert len(storage.issues) == 1
    assert storage.advanced == []
    assert error.value.stream_stats.pages_read == 1
