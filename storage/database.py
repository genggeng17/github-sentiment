from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import Engine, Select, create_engine, select, text, update
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from .models import (
    Base,
    CollectionStreamRun,
    Corpus,
    Issue,
    IssueComment,
    LlmAnnotation,
    PipelineRun,
    PullRequest,
    PullRequestComment,
    Repository,
    RepositoryCursor,
    RunStatus,
    utcnow,
)


class PipelineAlreadyRunning(RuntimeError):
    pass


class MissingParentError(RuntimeError):
    pass


class Storage:
    """唯一的数据库访问入口；业务模块不直接执行 SQL。"""

    def __init__(self, database_url: str, *, engine: Engine | None = None):
        self.engine = engine or create_engine(database_url, pool_pre_ping=True, pool_recycle=1800)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def pipeline_lock(self) -> Iterator[None]:
        """MySQL named lock prevents two cron processes from running together."""
        if self.engine.dialect.name != "mysql":
            yield
            return
        with self.engine.connect() as connection:
            acquired = connection.scalar(text("SELECT GET_LOCK('github_sentiment_pipeline', 0)"))
            if acquired != 1:
                raise PipelineAlreadyRunning("已有流水线任务正在运行")
            try:
                yield
            finally:
                connection.execute(text("SELECT RELEASE_LOCK('github_sentiment_pipeline')"))

    def ensure_repository(self, full_name: str) -> int:
        with self.sessions.begin() as session:
            existing = session.scalar(select(Repository).where(Repository.full_name == full_name))
            if existing:
                if not existing.enabled:
                    existing.enabled = True
                return existing.id
            repository = Repository(full_name=full_name)
            session.add(repository)
            session.flush()
            return repository.id

    def get_cursor(self, repository_id: int, stream: str) -> datetime | None:
        with self.sessions() as session:
            return session.scalar(
                select(RepositoryCursor.last_updated_at).where(
                    RepositoryCursor.repository_id == repository_id,
                    RepositoryCursor.stream == stream,
                )
            )

    def advance_cursor(self, repository_id: int, stream: str, value: datetime | None) -> None:
        if value is None:
            return
        row = {
            "repository_id": repository_id,
            "stream": stream,
            "last_updated_at": value,
            "updated_at": utcnow(),
        }
        self._upsert(
            RepositoryCursor, [row], ["repository_id", "stream"], ["last_updated_at", "updated_at"]
        )

    def upsert_issues(self, repository_id: int, rows: list[dict[str, Any]]) -> int:
        prepared = [{**row, "repository_id": repository_id} for row in rows]
        self._upsert(
            Issue,
            prepared,
            ["github_id"],
            [
                "number",
                "title",
                "body",
                "state",
                "author_login",
                "author_github_id",
                "github_url",
                "created_at",
                "updated_at",
                "closed_at",
                "collected_at",
            ],
        )
        return len(prepared)

    def upsert_pull_requests(self, repository_id: int, rows: list[dict[str, Any]]) -> int:
        prepared = [{**row, "repository_id": repository_id} for row in rows]
        self._upsert(
            PullRequest,
            prepared,
            ["github_id"],
            [
                "number",
                "title",
                "body",
                "state",
                "author_login",
                "author_github_id",
                "github_url",
                "created_at",
                "updated_at",
                "closed_at",
                "merged_at",
                "collected_at",
            ],
        )
        return len(prepared)

    def upsert_issue_comments(self, repository_id: int, rows: list[dict[str, Any]]) -> int:
        prepared = self._resolve_comment_parents(repository_id, rows, Issue, "issue_id")
        self._upsert(
            IssueComment,
            prepared,
            ["github_id"],
            [
                "issue_id",
                "body",
                "author_login",
                "author_github_id",
                "github_url",
                "created_at",
                "updated_at",
                "collected_at",
            ],
        )
        return len(prepared)

    def upsert_pr_comments(self, repository_id: int, rows: list[dict[str, Any]]) -> int:
        prepared = self._resolve_comment_parents(
            repository_id, rows, PullRequest, "pull_request_id"
        )
        self._upsert(
            PullRequestComment,
            prepared,
            ["github_id", "comment_type"],
            [
                "pull_request_id",
                "comment_type",
                "path",
                "in_reply_to_github_id",
                "body",
                "author_login",
                "author_github_id",
                "github_url",
                "created_at",
                "updated_at",
                "collected_at",
            ],
        )
        return len(prepared)

    def classify_parent_numbers(
        self, repository_id: int, numbers: Iterable[int]
    ) -> tuple[set[int], set[int]]:
        wanted = {int(number) for number in numbers}
        if not wanted:
            return set(), set()
        with self.sessions() as session:
            issue_numbers = set(
                session.scalars(
                    select(Issue.number).where(
                        Issue.repository_id == repository_id, Issue.number.in_(wanted)
                    )
                )
            )
            pr_numbers = set(
                session.scalars(
                    select(PullRequest.number).where(
                        PullRequest.repository_id == repository_id, PullRequest.number.in_(wanted)
                    )
                )
            )
        return issue_numbers, pr_numbers

    def _resolve_comment_parents(
        self,
        repository_id: int,
        rows: list[dict[str, Any]],
        model: type[Issue] | type[PullRequest],
        target_key: str,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        numbers = {int(row["parent_number"]) for row in rows}
        with self.sessions() as session:
            parents = session.execute(
                select(model.number, model.id).where(
                    model.repository_id == repository_id, model.number.in_(numbers)
                )
            ).all()
        mapping = dict(parents)
        missing = numbers - mapping.keys()
        if missing:
            raise MissingParentError(f"评论的父记录尚未采集: {sorted(missing)[:10]}")
        result = []
        for row in rows:
            item = dict(row)
            number = int(item.pop("parent_number"))
            item["repository_id"] = repository_id
            item[target_key] = mapping[number]
            result.append(item)
        return result

    def _upsert(
        self,
        model: type[Base],
        rows: list[dict[str, Any]],
        conflict_columns: list[str],
        update_columns: list[str],
    ) -> None:
        if not rows:
            return
        table = model.__table__
        with self.sessions.begin() as session:
            if self.engine.dialect.name == "mysql":
                statement = mysql_insert(table).values(rows)
                statement = statement.on_duplicate_key_update(
                    **{column: statement.inserted[column] for column in update_columns}
                )
            elif self.engine.dialect.name == "sqlite":
                statement = sqlite_insert(table).values(rows)
                statement = statement.on_conflict_do_update(
                    index_elements=conflict_columns,
                    set_={column: statement.excluded[column] for column in update_columns},
                )
            else:
                raise RuntimeError(f"暂不支持数据库方言: {self.engine.dialect.name}")
            session.execute(statement)

    def iter_corpus_candidates(self, batch_size: int = 500) -> Iterator[list[dict[str, Any]]]:
        queries: list[tuple[str, Select[Any]]] = [
            (
                "issue",
                select(
                    Issue.id.label("source_id"),
                    Issue.id.label("parent_id"),
                    Issue.title,
                    Issue.body,
                    Issue.updated_at.label("source_updated_at"),
                ),
            ),
            (
                "pull_request",
                select(
                    PullRequest.id.label("source_id"),
                    PullRequest.id.label("parent_id"),
                    PullRequest.title,
                    PullRequest.body,
                    PullRequest.updated_at.label("source_updated_at"),
                ),
            ),
            (
                "issue_comment",
                select(
                    IssueComment.id.label("source_id"),
                    IssueComment.issue_id.label("parent_id"),
                    Issue.title,
                    IssueComment.body,
                    IssueComment.updated_at.label("source_updated_at"),
                ).join(Issue, Issue.id == IssueComment.issue_id),
            ),
            (
                "pr_issue_comment",
                select(
                    PullRequestComment.id.label("source_id"),
                    PullRequestComment.pull_request_id.label("parent_id"),
                    PullRequest.title,
                    PullRequestComment.body,
                    PullRequestComment.path,
                    PullRequestComment.updated_at.label("source_updated_at"),
                )
                .join(PullRequest, PullRequest.id == PullRequestComment.pull_request_id)
                .where(PullRequestComment.comment_type == "issue_comment"),
            ),
            (
                "pr_review_comment",
                select(
                    PullRequestComment.id.label("source_id"),
                    PullRequestComment.pull_request_id.label("parent_id"),
                    PullRequest.title,
                    PullRequestComment.body,
                    PullRequestComment.path,
                    PullRequestComment.updated_at.label("source_updated_at"),
                )
                .join(PullRequest, PullRequest.id == PullRequestComment.pull_request_id)
                .where(PullRequestComment.comment_type == "review_comment"),
            ),
        ]
        with self.sessions() as session:
            for source_type, query in queries:
                last_id = 0
                while True:
                    rows = (
                        session.execute(
                            query.where(query.selected_columns.source_id > last_id)
                            .order_by(query.selected_columns.source_id)
                            .limit(batch_size)
                        )
                        .mappings()
                        .all()
                    )
                    if not rows:
                        break
                    batch = [{"source_type": source_type, **dict(row)} for row in rows]
                    yield batch
                    last_id = batch[-1]["source_id"]

    def find_corpus_by_hash(
        self, content_hash: str, *, exclude_id: int | None = None
    ) -> int | None:
        query = select(Corpus.id).where(Corpus.content_hash == content_hash)
        if exclude_id is not None:
            # 只允许指向更早的语料，避免重复组在重跑时形成环。
            query = query.where(Corpus.id < exclude_id)
        with self.sessions() as session:
            return session.scalar(query.order_by(Corpus.id).limit(1))

    def get_corpus_id(self, source_type: str, source_id: int, content_hash: str) -> int | None:
        with self.sessions() as session:
            return session.scalar(
                select(Corpus.id).where(
                    Corpus.source_type == source_type,
                    Corpus.source_id == source_id,
                    Corpus.content_hash == content_hash,
                )
            )

    def upsert_corpus(self, rows: list[dict[str, Any]]) -> int:
        self._upsert(
            Corpus,
            rows,
            ["source_type", "source_id", "content_hash"],
            [
                "duplicate_of_id",
                "source_updated_at",
                "updated_at",
            ],
        )
        return len(rows)

    def iter_unannotated_corpus(
        self,
        taxonomy_version: str,
        prompt_version: str,
        model_name: str,
        batch_size: int,
    ) -> Iterator[list[dict[str, Any]]]:
        last_id = 0
        with self.sessions() as session:
            while True:
                already = select(LlmAnnotation.corpus_id).where(
                    LlmAnnotation.taxonomy_version == taxonomy_version,
                    LlmAnnotation.prompt_version == prompt_version,
                    LlmAnnotation.model_name == model_name,
                    LlmAnnotation.status == "succeeded",
                )
                rows = (
                    session.execute(
                        select(Corpus.id, Corpus.model_input)
                        .where(
                            Corpus.id > last_id,
                            Corpus.duplicate_of_id.is_(None),
                            ~Corpus.id.in_(already),
                        )
                        .order_by(Corpus.id)
                        .limit(batch_size)
                    )
                    .mappings()
                    .all()
                )
                if not rows:
                    break
                batch = [dict(row) for row in rows]
                yield batch
                last_id = batch[-1]["id"]

    def save_annotation(self, row: dict[str, Any]) -> None:
        row = {**row, "updated_at": utcnow()}
        self._upsert(
            LlmAnnotation,
            [row],
            ["corpus_id", "taxonomy_version", "prompt_version", "model_name"],
            ["raw_response", "parsed_result", "status", "error_message", "updated_at"],
        )

    def start_pipeline_run(self, run_type: str) -> str:
        run = PipelineRun(run_type=run_type, status=RunStatus.RUNNING.value)
        with self.sessions.begin() as session:
            # 能取得 named lock 说明旧的 running 记录已不对应活跃进程（例如进程崩溃）。
            session.execute(
                update(PipelineRun)
                .where(PipelineRun.status == RunStatus.RUNNING.value)
                .values(
                    status=RunStatus.FAILED.value,
                    completed_at=utcnow(),
                    heartbeat_at=utcnow(),
                    error_message="任务进程中断，已由后续运行接管",
                )
            )
            session.add(run)
            session.flush()
            return run.id

    def finish_pipeline_run(
        self,
        run_id: str,
        status: str,
        stats: dict[str, Any],
        error_message: str | None = None,
    ) -> None:
        with self.sessions.begin() as session:
            session.execute(
                update(PipelineRun)
                .where(PipelineRun.id == run_id)
                .values(
                    status=status,
                    stats=stats,
                    error_message=error_message,
                    heartbeat_at=utcnow(),
                    completed_at=utcnow(),
                )
            )

    def heartbeat(self, run_id: str) -> None:
        with self.sessions.begin() as session:
            session.execute(
                update(PipelineRun).where(PipelineRun.id == run_id).values(heartbeat_at=utcnow())
            )

    def save_stream_run(self, row: dict[str, Any]) -> None:
        self._upsert(
            CollectionStreamRun,
            [row],
            ["pipeline_run_id", "repository_id", "stream"],
            [
                "status",
                "pages_read",
                "items_read",
                "items_written",
                "retries",
                "error_message",
                "completed_at",
            ],
        )

    def recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.execute(
                select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit)
            ).scalars()
            return [
                {
                    "id": row.id,
                    "run_type": row.run_type,
                    "status": row.status,
                    "started_at": row.started_at.isoformat(),
                    "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                    "stats": row.stats,
                    "error_message": row.error_message,
                }
                for row in rows
            ]
