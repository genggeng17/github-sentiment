from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.exc import OperationalError, ProgrammingError

from storage import Storage
from storage.models import utcnow

from .github_client import GitHubClient, GitHubPage, GitHubRequestError

logger = logging.getLogger(__name__)

STREAM_ISSUES = "issues_and_pull_requests"
STREAM_ISSUE_COMMENTS = "issue_comments"
STREAM_REVIEW_COMMENTS = "review_comments"


class StreamFailure(RuntimeError):
    def __init__(self, stats: StreamStats, cause: Exception):
        super().__init__(str(cause))
        self.stream_stats = stats
        self.cause = cause


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
        segment_page_limit: int = 250,
        page_write_max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.client = client
        self.storage = storage
        self.cursor_overlap = timedelta(seconds=cursor_overlap_seconds)
        self.segment_page_limit = segment_page_limit
        self.page_write_max_retries = page_write_max_retries
        self._sleep = sleep

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
        parents_available = True
        for stream, handler in streams:
            started_at = utcnow()
            fatal_error: Exception | None = None
            if stream != STREAM_ISSUES and not parents_available:
                stats = StreamStats(
                    status="skipped_dependency",
                    error_message="Issue/PR 父数据流失败，已跳过依赖的评论流",
                )
            else:
                try:
                    stats = handler(repository_id, owner, repo)
                    stats.status = "succeeded"
                except Exception as exc:  # failure is recorded before propagation/continuation
                    logger.exception("采集 %s/%s 失败", full_name, stream)
                    stats = getattr(exc, "stream_stats", StreamStats())
                    stats.status = "failed"
                    stats.error_message = str(exc)
                    if stream == STREAM_ISSUES:
                        parents_available = False
                    if self._is_fatal_error(exc):
                        fatal_error = exc
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
            if fatal_error is not None:
                raise fatal_error
        return results

    @staticmethod
    def _is_fatal_error(exc: Exception) -> bool:
        cause = exc.cause if isinstance(exc, StreamFailure) else exc
        return (
            isinstance(cause, GitHubRequestError) and cause.status_code in {401, 403}
        ) or isinstance(cause, ProgrammingError)

    def _since_params(self, repository_id: int, stream: str) -> dict[str, Any]:
        params: dict[str, Any] = {"per_page": 100, "sort": "updated", "direction": "asc"}
        cursor = self.storage.get_cursor(repository_id, stream)
        if cursor:
            since = (cursor - self.cursor_overlap).replace(tzinfo=UTC)
            params["since"] = since.isoformat().replace("+00:00", "Z")
        return params

    def _segmented_pages(
        self,
        repository_id: int,
        stream: str,
        path: str,
        extra_params: dict[str, Any] | None = None,
    ) -> Iterator[GitHubPage]:
        """Restart from the durable cursor before GitHub's page-300 boundary."""
        previous = self.storage.get_cursor(repository_id, stream)
        while True:
            params = {**self._since_params(repository_id, stream), **(extra_params or {})}
            pages_in_segment = 0
            for page in self.client.paginate(
                path, params, max_pages=self.segment_page_limit
            ):
                pages_in_segment += 1
                yield page
            if pages_in_segment < self.segment_page_limit:
                return
            current = self.storage.get_cursor(repository_id, stream)
            if current is None or (previous is not None and current <= previous):
                raise RuntimeError(
                    f"{stream} 分段处理没有推进游标；"
                    "安全重叠窗口内数据量过大，已停止以避免无限循环"
                )
            previous = current

    @staticmethod
    def _advance_raw_position(
        current_time: datetime | None, items: list[dict[str, Any]]
    ) -> datetime | None:
        values: list[datetime] = []
        for item in items:
            try:
                value = parse_github_datetime(item.get("updated_at"))
            except (AttributeError, TypeError, ValueError):
                continue
            if value is not None:
                values.append(value)
        if not values:
            return current_time
        return max(current_time or datetime.min, max(values))

    @staticmethod
    def _items_by_id(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        result: dict[int, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                result[int(item["id"])] = item
            except (KeyError, TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _unresolved_row(
        stream: str,
        item: Any,
        parent: int | None,
        reason: str,
        *,
        category: str = "missing_parent",
    ) -> dict[str, Any]:
        payload = item if isinstance(item, dict) else {"raw": item}
        raw_id = payload.get("id")
        try:
            github_id = int(raw_id) if raw_id is not None else None
        except (TypeError, ValueError):
            github_id = None
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        item_key = str(github_id) if github_id is not None else hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()
        return {
            "stream": stream,
            "item_key": item_key,
            "category": category,
            "github_id": github_id,
            "parent_number": parent,
            "payload": payload,
            "reason": reason,
        }

    @staticmethod
    def _validate_row(row: dict[str, Any]) -> None:
        limits = {
            "author_login": 255,
            "github_url": 1024,
            "title": 1024,
            "state": 20,
            "comment_type": 30,
            "path": 1024,
        }
        for field, limit in limits.items():
            value = row.get(field)
            if value is not None and len(str(value)) > limit:
                raise ValueError(f"字段 {field} 长度超过数据库上限 {limit}")

    def _parse_page_rows(
        self,
        stream: str,
        items: list[dict[str, Any]],
        parser: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        for item in items:
            try:
                if not isinstance(item, dict):
                    raise TypeError("GitHub 列表项不是对象")
                row = parser(item)
                self._validate_row(row)
                rows.append(row)
            except (KeyError, TypeError, ValueError) as exc:
                unresolved.append(
                    self._unresolved_row(
                        stream,
                        item,
                        None,
                        f"记录解析失败: {exc}",
                        category="invalid_record",
                    )
                )
        return rows, unresolved

    def _commit_page(self, *args: Any, **kwargs: Any) -> int:
        for attempt in range(self.page_write_max_retries + 1):
            try:
                return self.storage.commit_collection_page(*args, **kwargs)
            except OperationalError:
                if attempt >= self.page_write_max_retries:
                    raise
                delay = min(8.0, float(2**attempt))
                logger.warning("页面数据库事务暂时失败，%.1f 秒后重试", delay)
                self._sleep(delay)
        raise AssertionError("unreachable")

    def _collect_issues_and_prs(self, repository_id: int, owner: str, repo: str) -> StreamStats:
        stats = StreamStats()
        scan_started_at = utcnow()
        latest = self.storage.get_cursor(repository_id, STREAM_ISSUES)
        try:
            for page in self._segmented_pages(
                repository_id,
                STREAM_ISSUES,
                f"/repos/{owner}/{repo}/issues",
                {"state": "all"},
            ):
                stats.pages_read += 1
                stats.items_read += len(page.items)
                stats.retries += page.retries
                parsed_rows, unresolved = self._parse_page_rows(
                    STREAM_ISSUES,
                    page.items,
                    lambda item: {
                        "_collection_kind": "pr" if "pull_request" in item else "issue",
                        **(
                            self._parse_pr(item)
                            if "pull_request" in item
                            else self._parse_issue(item)
                        ),
                    },
                )
                issue_rows = [
                    {key: value for key, value in row.items() if key != "_collection_kind"}
                    for row in parsed_rows
                    if row["_collection_kind"] == "issue"
                ]
                pr_rows = [
                    {key: value for key, value in row.items() if key != "_collection_kind"}
                    for row in parsed_rows
                    if row["_collection_kind"] == "pr"
                ]
                latest = self._advance_raw_position(latest, page.items)
                if latest is not None:
                    stats.items_written += self._commit_page(
                        repository_id,
                        STREAM_ISSUES,
                        latest,
                        issues=issue_rows,
                        pull_requests=pr_rows,
                        unresolved=unresolved,
                    )
                elif unresolved:
                    self.storage.save_unresolved_items(repository_id, unresolved)
                    raise RuntimeError("当前页没有可用 updated_at，无法安全推进游标")
            self.storage.advance_cursor(
                repository_id, STREAM_ISSUES, latest or scan_started_at
            )
            return stats
        except Exception as exc:
            raise StreamFailure(stats, exc) from exc

    def _collect_issue_comments(self, repository_id: int, owner: str, repo: str) -> StreamStats:
        stats = StreamStats()
        scan_started_at = utcnow()
        latest = self.storage.get_cursor(repository_id, STREAM_ISSUE_COMMENTS)
        try:
            self._reconcile_unresolved(
                repository_id, owner, repo, STREAM_ISSUE_COMMENTS
            )
            for page in self._segmented_pages(
                repository_id,
                STREAM_ISSUE_COMMENTS,
                f"/repos/{owner}/{repo}/issues/comments",
            ):
                stats.pages_read += 1
                stats.items_read += len(page.items)
                stats.retries += page.retries
                rows, unresolved = self._parse_page_rows(
                    STREAM_ISSUE_COMMENTS, page.items, self._parse_comment
                )
                items_by_id = self._items_by_id(page.items)
                numbers = {row["parent_number"] for row in rows}
                issue_numbers, pr_numbers = self.storage.classify_parent_numbers(
                    repository_id, numbers
                )
                unknown = numbers - issue_numbers - pr_numbers
                if unknown:
                    failures = self._supplement_parents(
                        repository_id, owner, repo, unknown, require_pr=False
                    )
                    issue_numbers, pr_numbers = self.storage.classify_parent_numbers(
                        repository_id, numbers
                    )
                else:
                    failures = {}
                unknown = numbers - issue_numbers - pr_numbers
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
                unresolved.extend(
                    self._unresolved_row(
                        STREAM_ISSUE_COMMENTS,
                        items_by_id.get(row["github_id"], {"id": row["github_id"]}),
                        row["parent_number"],
                        failures.get(row["parent_number"], "补采后父记录仍不存在"),
                    )
                    for row in rows
                    if row["parent_number"] in unknown
                )
                latest = self._advance_raw_position(latest, page.items)
                if latest is not None:
                    stats.items_written += self._commit_page(
                        repository_id,
                        STREAM_ISSUE_COMMENTS,
                        latest,
                        issue_comments=issue_rows,
                        pr_comments=pr_rows,
                        unresolved=unresolved,
                    )
                elif unresolved:
                    self.storage.save_unresolved_items(repository_id, unresolved)
                    raise RuntimeError("当前页没有可用 updated_at，无法安全推进游标")
            self.storage.advance_cursor(
                repository_id,
                STREAM_ISSUE_COMMENTS,
                latest or scan_started_at,
            )
            return stats
        except Exception as exc:
            raise StreamFailure(stats, exc) from exc

    def _collect_review_comments(self, repository_id: int, owner: str, repo: str) -> StreamStats:
        stats = StreamStats()
        scan_started_at = utcnow()
        latest = self.storage.get_cursor(repository_id, STREAM_REVIEW_COMMENTS)
        try:
            self._reconcile_unresolved(repository_id, owner, repo, STREAM_REVIEW_COMMENTS)
            for page in self._segmented_pages(
                repository_id,
                STREAM_REVIEW_COMMENTS,
                f"/repos/{owner}/{repo}/pulls/comments",
            ):
                stats.pages_read += 1
                stats.items_read += len(page.items)
                stats.retries += page.retries
                rows, unresolved = self._parse_page_rows(
                    STREAM_REVIEW_COMMENTS, page.items, self._parse_review_comment
                )
                items_by_id = self._items_by_id(page.items)
                numbers = {row["parent_number"] for row in rows}
                _, pr_numbers = self.storage.classify_parent_numbers(repository_id, numbers)
                unknown = numbers - pr_numbers
                if unknown:
                    failures = self._supplement_parents(
                        repository_id, owner, repo, unknown, require_pr=True
                    )
                    _, pr_numbers = self.storage.classify_parent_numbers(repository_id, numbers)
                else:
                    failures = {}
                unknown = numbers - pr_numbers
                valid_rows = [row for row in rows if row["parent_number"] in pr_numbers]
                unresolved.extend(
                    self._unresolved_row(
                        STREAM_REVIEW_COMMENTS,
                        items_by_id.get(row["github_id"], {"id": row["github_id"]}),
                        row["parent_number"],
                        failures.get(row["parent_number"], "补采后 PR 父记录仍不存在"),
                    )
                    for row in rows
                    if row["parent_number"] in unknown
                )
                latest = self._advance_raw_position(latest, page.items)
                if latest is not None:
                    stats.items_written += self._commit_page(
                        repository_id,
                        STREAM_REVIEW_COMMENTS,
                        latest,
                        pr_comments=valid_rows,
                        unresolved=unresolved,
                    )
                elif unresolved:
                    self.storage.save_unresolved_items(repository_id, unresolved)
                    raise RuntimeError("当前页没有可用 updated_at，无法安全推进游标")
            self.storage.advance_cursor(
                repository_id,
                STREAM_REVIEW_COMMENTS,
                latest or scan_started_at,
            )
            return stats
        except Exception as exc:
            raise StreamFailure(stats, exc) from exc

    def _supplement_parents(
        self,
        repository_id: int,
        owner: str,
        repo: str,
        numbers: set[int],
        *,
        require_pr: bool,
    ) -> dict[int, str]:
        """Fetch exceptional missing parents once per distinct repository number."""
        failures: dict[int, str] = {}
        for number in sorted(numbers):
            try:
                item, _retries = self.client.get_json(
                    f"/repos/{owner}/{repo}/issues/{number}"
                )
            except GitHubRequestError as exc:
                if exc.status_code in {404, 410}:
                    failures[number] = str(exc)
                    continue
                raise
            if "pull_request" in item:
                self.storage.upsert_pull_requests(repository_id, [self._parse_pr(item)])
            elif require_pr:
                failures[number] = "补采返回的是 Issue，不是 PR"
            else:
                self.storage.upsert_issues(repository_id, [self._parse_issue(item)])
        return failures

    def _reconcile_unresolved(
        self,
        repository_id: int,
        owner: str,
        repo: str,
        stream: str,
    ) -> None:
        pending = self.storage.pending_unresolved_items(repository_id, stream)
        if not pending:
            return
        numbers = {int(row["parent_number"]) for row in pending}
        issue_numbers, pr_numbers = self.storage.classify_parent_numbers(repository_id, numbers)
        if stream == STREAM_REVIEW_COMMENTS:
            unknown = numbers - pr_numbers
            require_pr = True
        else:
            unknown = numbers - issue_numbers - pr_numbers
            require_pr = False
        failures = self._supplement_parents(
            repository_id, owner, repo, unknown, require_pr=require_pr
        )
        issue_numbers, pr_numbers = self.storage.classify_parent_numbers(repository_id, numbers)

        issue_rows: list[dict[str, Any]] = []
        pr_rows: list[dict[str, Any]] = []
        resolved_ids: list[int] = []
        failed_ids: list[int] = []
        for row in pending:
            number = int(row["parent_number"])
            payload = row["payload"]
            if stream == STREAM_REVIEW_COMMENTS:
                if number not in pr_numbers:
                    failed_ids.append(row["id"])
                    continue
                pr_rows.append(self._parse_review_comment(payload))
            else:
                parsed = self._parse_comment(payload)
                if number in issue_numbers:
                    issue_rows.append(parsed)
                elif number in pr_numbers:
                    pr_rows.append(
                        {
                            **parsed,
                            "comment_type": "issue_comment",
                            "path": None,
                            "in_reply_to_github_id": None,
                        }
                    )
                else:
                    failed_ids.append(row["id"])
                    continue
            resolved_ids.append(row["id"])

        self.storage.upsert_issue_comments(repository_id, issue_rows)
        self.storage.upsert_pr_comments(repository_id, pr_rows)
        self.storage.mark_unresolved_resolved(resolved_ids)
        if failed_ids:
            reason = "; ".join(sorted(set(failures.values()))) or "对账后父记录仍不存在"
            self.storage.mark_unresolved_attempt(failed_ids, reason)
        if resolved_ids:
            logger.info("仓库 %s/%s 对账恢复 %d 条评论", owner, repo, len(resolved_ids))

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
        pull_request = item.get("pull_request") or {}
        return {
            **cls._common(item),
            "github_id": item["id"],
            "number": item["number"],
            "title": item.get("title") or "",
            "state": item["state"],
            "closed_at": parse_github_datetime(item.get("closed_at")),
            "merged_at": parse_github_datetime(
                item.get("merged_at") or pull_request.get("merged_at")
            ),
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
