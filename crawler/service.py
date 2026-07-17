from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from storage import Storage
from storage.models import utcnow

from .github_client import GitHubClient

logger = logging.getLogger(__name__)

STREAM_ISSUES = "issues_and_pull_requests"
STREAM_ISSUE_COMMENTS = "issue_comments"
STREAM_REVIEW_COMMENTS = "review_comments"


class StreamFailure(RuntimeError):
    def __init__(self, stats: StreamStats, cause: Exception):
        super().__init__(str(cause))
        self.stream_stats = stats


def parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC).replace(tzinfo=None)


def parent_number(url: str) -> int:
    match = re.search(r"/(?:issues|pulls)/(\d+)$", url)
    if not match:
        raise ValueError(f"无法从 URL 解析父编号: {url}")
    return int(match.group(1))


@dataclass(slots=True)
class StreamStats:
    status: str = "running"
    pages_read: int = 0
    items_read: int = 0
    items_written: int = 0
    retries: int = 0
    error_message: str | None = None


class GitHubCollector:
    def __init__(
        self,
        client: GitHubClient,
        storage: Storage,
        *,
        cursor_overlap_seconds: int = 300,
    ):
        self.client = client
        self.storage = storage
        self.cursor_overlap = timedelta(seconds=cursor_overlap_seconds)

    def collect_repository(
        self,
        full_name: str,
        run_id: str,
        heartbeat: Callable[[], None] | None = None,
    ) -> dict[str, dict[str, Any]]:
        owner, repo = full_name.split("/", 1)
        repository_id = self.storage.ensure_repository(full_name)
        results: dict[str, dict[str, Any]] = {}
        streams = (
            (STREAM_ISSUES, self._collect_issues_and_prs),
            (STREAM_ISSUE_COMMENTS, self._collect_issue_comments),
            (STREAM_REVIEW_COMMENTS, self._collect_review_comments),
        )
        for stream, handler in streams:
            started_at = utcnow()
            try:
                stats = handler(repository_id, owner, repo)
                stats.status = "succeeded"
            except Exception as exc:  # stream failure is recorded; other repositories can continue
                logger.exception("采集 %s/%s 失败", full_name, stream)
                stats = getattr(exc, "stream_stats", StreamStats())
                stats.status = "failed"
                stats.error_message = str(exc)
            if heartbeat:
                heartbeat()
            result = asdict(stats)
            results[stream] = result
            self.storage.save_stream_run(
                {
                    "pipeline_run_id": run_id,
                    "repository_id": repository_id,
                    "stream": stream,
                    **result,
                    "started_at": started_at,
                    "completed_at": utcnow(),
                }
            )
        return results

    def _since_params(self, repository_id: int, stream: str) -> dict[str, Any]:
        params: dict[str, Any] = {"per_page": 100, "sort": "updated", "direction": "asc"}
        cursor = self.storage.get_cursor(repository_id, stream)
        if cursor:
            since = (cursor - self.cursor_overlap).replace(tzinfo=UTC)
            params["since"] = since.isoformat().replace("+00:00", "Z")
        return params

    def _collect_issues_and_prs(self, repository_id: int, owner: str, repo: str) -> StreamStats:
        stats = StreamStats()
        latest: datetime | None = None
        params = {**self._since_params(repository_id, STREAM_ISSUES), "state": "all"}
        try:
            for page in self.client.paginate(f"/repos/{owner}/{repo}/issues", params):
                stats.pages_read += 1
                stats.items_read += len(page.items)
                stats.retries += page.retries
                issue_rows: list[dict[str, Any]] = []
                pr_rows: list[dict[str, Any]] = []
                for item in page.items:
                    updated_at = parse_github_datetime(item["updated_at"])
                    latest = max(filter(None, [latest, updated_at]), default=None)
                    if "pull_request" in item:
                        detail, retries = self.client.get_json(
                            f"/repos/{owner}/{repo}/pulls/{item['number']}"
                        )
                        stats.retries += retries
                        pr_rows.append(self._parse_pr(detail))
                    else:
                        issue_rows.append(self._parse_issue(item))
                stats.items_written += self.storage.upsert_issues(repository_id, issue_rows)
                stats.items_written += self.storage.upsert_pull_requests(repository_id, pr_rows)
            self.storage.advance_cursor(repository_id, STREAM_ISSUES, latest)
            return stats
        except Exception as exc:
            raise StreamFailure(stats, exc) from exc

    def _collect_issue_comments(self, repository_id: int, owner: str, repo: str) -> StreamStats:
        stats = StreamStats()
        latest: datetime | None = None
        try:
            for page in self.client.paginate(
                f"/repos/{owner}/{repo}/issues/comments",
                self._since_params(repository_id, STREAM_ISSUE_COMMENTS),
            ):
                stats.pages_read += 1
                stats.items_read += len(page.items)
                stats.retries += page.retries
                rows = [self._parse_comment(item) for item in page.items]
                numbers = {row["parent_number"] for row in rows}
                issue_numbers, pr_numbers = self.storage.classify_parent_numbers(
                    repository_id, numbers
                )
                unknown = numbers - issue_numbers - pr_numbers
                if unknown:
                    raise RuntimeError(f"普通评论父记录缺失: {sorted(unknown)[:10]}")
                issue_rows = [row for row in rows if row["parent_number"] in issue_numbers]
                pr_rows = [
                    {
                        **row,
                        "comment_type": "issue_comment",
                        "path": None,
                        "in_reply_to_github_id": None,
                    }
                    for row in rows
                    if row["parent_number"] in pr_numbers
                ]
                stats.items_written += self.storage.upsert_issue_comments(repository_id, issue_rows)
                stats.items_written += self.storage.upsert_pr_comments(repository_id, pr_rows)
                for row in rows:
                    latest = max(filter(None, [latest, row["updated_at"]]), default=None)
            self.storage.advance_cursor(repository_id, STREAM_ISSUE_COMMENTS, latest)
            return stats
        except Exception as exc:
            raise StreamFailure(stats, exc) from exc

    def _collect_review_comments(self, repository_id: int, owner: str, repo: str) -> StreamStats:
        stats = StreamStats()
        latest: datetime | None = None
        try:
            for page in self.client.paginate(
                f"/repos/{owner}/{repo}/pulls/comments",
                self._since_params(repository_id, STREAM_REVIEW_COMMENTS),
            ):
                stats.pages_read += 1
                stats.items_read += len(page.items)
                stats.retries += page.retries
                rows = [self._parse_review_comment(item) for item in page.items]
                stats.items_written += self.storage.upsert_pr_comments(repository_id, rows)
                for row in rows:
                    latest = max(filter(None, [latest, row["updated_at"]]), default=None)
            self.storage.advance_cursor(repository_id, STREAM_REVIEW_COMMENTS, latest)
            return stats
        except Exception as exc:
            raise StreamFailure(stats, exc) from exc

    @staticmethod
    def _common(item: dict[str, Any]) -> dict[str, Any]:
        user = item.get("user") or {}
        return {
            "author_login": user.get("login"),
            "author_github_id": user.get("id"),
            "body": item.get("body"),
            "github_url": item.get("html_url") or item.get("url") or "",
            "created_at": parse_github_datetime(item["created_at"]),
            "updated_at": parse_github_datetime(item["updated_at"]),
            "collected_at": utcnow(),
        }

    @classmethod
    def _parse_issue(cls, item: dict[str, Any]) -> dict[str, Any]:
        return {
            **cls._common(item),
            "github_id": item["id"],
            "number": item["number"],
            "title": item.get("title") or "",
            "state": item["state"],
            "closed_at": parse_github_datetime(item.get("closed_at")),
        }

    @classmethod
    def _parse_pr(cls, item: dict[str, Any]) -> dict[str, Any]:
        return {
            **cls._common(item),
            "github_id": item["id"],
            "number": item["number"],
            "title": item.get("title") or "",
            "state": item["state"],
            "closed_at": parse_github_datetime(item.get("closed_at")),
            "merged_at": parse_github_datetime(item.get("merged_at")),
        }

    @classmethod
    def _parse_comment(cls, item: dict[str, Any]) -> dict[str, Any]:
        return {
            **cls._common(item),
            "github_id": item["id"],
            "parent_number": parent_number(item["issue_url"]),
        }

    @classmethod
    def _parse_review_comment(cls, item: dict[str, Any]) -> dict[str, Any]:
        return {
            **cls._common(item),
            "github_id": item["id"],
            "parent_number": parent_number(item["pull_request_url"]),
            "comment_type": "review_comment",
            "path": item.get("path"),
            "in_reply_to_github_id": item.get("in_reply_to_id"),
        }
