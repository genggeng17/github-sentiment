from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Engine,
    Select,
    create_engine,
    delete,
    func,
    inspect,
    select,
    text,
    union_all,
    update,
)
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    Base,
    CollectionStreamRun,
    Corpus,
    CorpusSampleItem,
    CorpusSampleSet,
    Issue,
    IssueComment,
    LlmAnnotation,
    PipelineRun,
    PullRequest,
    PullRequestComment,
    Repository,
    RepositoryCursor,
    RunStatus,
    UnresolvedCollectionItem,
    utcnow,
)

logger = logging.getLogger(__name__)


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
        inspector = inspect(self.engine)
        if "corpus_sample_sets" not in inspector.get_table_names():
            return
        columns = {column["name"] for column in inspector.get_columns("corpus_sample_sets")}
        if "cleaning_version" not in columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE corpus_sample_sets "
                        "ADD COLUMN cleaning_version VARCHAR(30) NULL"
                    )
                )
        if "max_model_input_chars" not in columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE corpus_sample_sets "
                        "ADD COLUMN max_model_input_chars INTEGER NULL"
                    )
                )
        corpus_columns = {
            column["name"] for column in inspector.get_columns("corpus")
        }
        if "model_input_chars" not in corpus_columns:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "ALTER TABLE corpus "
                        "ADD COLUMN model_input_chars INTEGER NULL"
                    )
                )
        self._backfill_model_input_chars()
        corpus_indexes = {
            index["name"] for index in inspect(self.engine).get_indexes("corpus")
        }
        if "ix_corpus_sampling" not in corpus_indexes:
            with self.engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE INDEX ix_corpus_sampling ON corpus "
                        "(cleaning_version, source_type, duplicate_of_id, "
                        "model_input_chars)"
                    )
                )

    def _backfill_model_input_chars(self, batch_size: int = 10000) -> None:
        length_expression = (
            "CHAR_LENGTH(model_input)"
            if self.engine.dialect.name == "mysql"
            else "length(model_input)"
        )
        update_lengths = text(
            "UPDATE corpus "
            f"SET model_input_chars = {length_expression} "
            "WHERE id >= :first_id AND id <= :last_id "
            "AND model_input_chars IS NULL"
        )
        last_id = 0
        updated = 0
        while True:
            with self.engine.begin() as connection:
                ids = connection.scalars(
                    select(Corpus.id)
                    .where(
                        Corpus.id > last_id,
                        Corpus.model_input_chars.is_(None),
                    )
                    .order_by(Corpus.id)
                    .limit(batch_size)
                ).all()
                if not ids:
                    break
                connection.execute(
                    update_lengths,
                    {"first_id": ids[0], "last_id": ids[-1]},
                )
            last_id = ids[-1]
            updated += len(ids)
            if updated % 100000 == 0:
                logger.info("已回填 %d 条 corpus 输入长度", updated)
        if updated:
            logger.info("corpus 输入长度回填完成，共 %d 条", updated)

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

    def bootstrap_repositories(self, full_names: Iterable[str]) -> int:
        """仅当仓库表为空时导入环境变量中的初始白名单。"""
        names = tuple(dict.fromkeys(full_names))
        if not names:
            return 0
        with self.sessions.begin() as session:
            existing_count = session.scalar(select(func.count()).select_from(Repository)) or 0
            if existing_count:
                return 0
            session.add_all(Repository(full_name=name, enabled=True) for name in names)
        return len(names)

    def list_repositories(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        query = select(Repository).order_by(Repository.full_name)
        if enabled_only:
            query = query.where(Repository.enabled.is_(True))
        with self.sessions() as session:
            rows = session.scalars(query).all()
            return [
                {
                    "id": row.id,
                    "full_name": row.full_name,
                    "enabled": row.enabled,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in rows
            ]

    def enabled_repository_names(self) -> list[str]:
        return [row["full_name"] for row in self.list_repositories(enabled_only=True)]

    def set_repository_enabled(self, full_name: str, enabled: bool) -> bool:
        with self.sessions.begin() as session:
            repository = session.scalar(select(Repository).where(Repository.full_name == full_name))
            if repository is None:
                raise ValueError(f"仓库不在白名单中: {full_name}")
            changed = repository.enabled != enabled
            repository.enabled = enabled
            repository.updated_at = utcnow()
            return changed

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

    def commit_collection_page(
        self,
        repository_id: int,
        stream: str,
        cursor_updated_at: datetime,
        *,
        issues: list[dict[str, Any]] | None = None,
        pull_requests: list[dict[str, Any]] | None = None,
        issue_comments: list[dict[str, Any]] | None = None,
        pr_comments: list[dict[str, Any]] | None = None,
        unresolved: list[dict[str, Any]] | None = None,
    ) -> int:
        """Atomically persist one API page and advance its durable cursor."""
        issue_rows = [{**row, "repository_id": repository_id} for row in issues or []]
        pr_rows = [{**row, "repository_id": repository_id} for row in pull_requests or []]
        unresolved_rows = [
            {
                **row,
                "repository_id": repository_id,
                "attempt_count": 1,
                "first_seen_at": row.get("first_seen_at", utcnow()),
                "last_attempt_at": utcnow(),
                "resolved_at": None,
            }
            for row in unresolved or []
        ]
        with self.sessions.begin() as session:
            self._upsert_in_session(
                session,
                Issue,
                issue_rows,
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
            self._upsert_in_session(
                session,
                PullRequest,
                pr_rows,
                ["repository_id", "number"],
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
            prepared_issue_comments = self._resolve_comment_parents_in_session(
                session, repository_id, issue_comments or [], Issue, "issue_id"
            )
            prepared_pr_comments = self._resolve_comment_parents_in_session(
                session, repository_id, pr_comments or [], PullRequest, "pull_request_id"
            )
            self._upsert_in_session(
                session,
                IssueComment,
                prepared_issue_comments,
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
            self._upsert_in_session(
                session,
                PullRequestComment,
                prepared_pr_comments,
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
            self._upsert_in_session(
                session,
                UnresolvedCollectionItem,
                unresolved_rows,
                ["repository_id", "stream", "item_key"],
                [
                    "category",
                    "github_id",
                    "parent_number",
                    "payload",
                    "reason",
                    "last_attempt_at",
                    "resolved_at",
                ],
            )
            self._upsert_in_session(
                session,
                RepositoryCursor,
                [
                    {
                        "repository_id": repository_id,
                        "stream": stream,
                        "last_updated_at": cursor_updated_at,
                        "updated_at": utcnow(),
                    }
                ],
                ["repository_id", "stream"],
                ["last_updated_at", "updated_at"],
            )
        return len(issue_rows) + len(pr_rows) + len(prepared_issue_comments) + len(
            prepared_pr_comments
        )

    def save_unresolved_items(
        self, repository_id: int, rows: list[dict[str, Any]]
    ) -> None:
        prepared = [
            {
                **row,
                "repository_id": repository_id,
                "attempt_count": 1,
                "first_seen_at": row.get("first_seen_at", utcnow()),
                "last_attempt_at": utcnow(),
                "resolved_at": None,
            }
            for row in rows
        ]
        self._upsert(
            UnresolvedCollectionItem,
            prepared,
            ["repository_id", "stream", "item_key"],
            [
                "category",
                "github_id",
                "parent_number",
                "payload",
                "reason",
                "last_attempt_at",
                "resolved_at",
            ],
        )

    def pending_unresolved_items(
        self,
        repository_id: int,
        stream: str,
        *,
        category: str = "missing_parent",
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        with self.sessions() as session:
            rows = session.execute(
                select(UnresolvedCollectionItem)
                .where(
                    UnresolvedCollectionItem.repository_id == repository_id,
                    UnresolvedCollectionItem.stream == stream,
                    UnresolvedCollectionItem.category == category,
                    UnresolvedCollectionItem.resolved_at.is_(None),
                )
                .order_by(UnresolvedCollectionItem.id)
                .limit(limit)
            ).scalars()
            return [
                {
                    "id": row.id,
                    "github_id": row.github_id,
                    "parent_number": row.parent_number,
                    "payload": row.payload,
                }
                for row in rows
            ]

    def mark_unresolved_resolved(self, ids: Iterable[int]) -> None:
        values = tuple(ids)
        if not values:
            return
        with self.sessions.begin() as session:
            session.execute(
                update(UnresolvedCollectionItem)
                .where(UnresolvedCollectionItem.id.in_(values))
                .values(resolved_at=utcnow(), last_attempt_at=utcnow())
            )

    def mark_unresolved_attempt(self, ids: Iterable[int], reason: str) -> None:
        values = tuple(ids)
        if not values:
            return
        with self.sessions.begin() as session:
            session.execute(
                update(UnresolvedCollectionItem)
                .where(UnresolvedCollectionItem.id.in_(values))
                .values(
                    attempt_count=UnresolvedCollectionItem.attempt_count + 1,
                    reason=reason,
                    last_attempt_at=utcnow(),
                )
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
            ["repository_id", "number"],
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
        with self.sessions() as session:
            return self._resolve_comment_parents_in_session(
                session, repository_id, rows, model, target_key
            )

    @staticmethod
    def _resolve_comment_parents_in_session(
        session: Session,
        repository_id: int,
        rows: list[dict[str, Any]],
        model: type[Issue] | type[PullRequest],
        target_key: str,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        numbers = {int(row["parent_number"]) for row in rows}
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
        with self.sessions.begin() as session:
            self._upsert_in_session(session, model, rows, conflict_columns, update_columns)

    def _upsert_in_session(
        self,
        session: Session,
        model: type[Base],
        rows: list[dict[str, Any]],
        conflict_columns: list[str],
        update_columns: list[str],
    ) -> None:
        if not rows:
            return
        table = model.__table__
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

    def iter_sample_candidates(
        self,
        repository_ids: Iterable[int],
        *,
        cleaning_version: str | None = None,
        max_model_input_chars: int | None = None,
        batch_size: int = 5000,
    ) -> Iterator[list[dict[str, Any]]]:
        repository_ids = tuple(repository_ids)
        if not repository_ids:
            return
        queries: list[Select[Any]] = [
            select(
                Corpus.id,
                Corpus.source_type,
                Corpus.content_hash,
                Issue.repository_id.label("repository_id"),
            )
            .join(Issue, Issue.id == Corpus.source_id)
            .where(
                Corpus.source_type == "issue",
                Issue.repository_id.in_(repository_ids),
            ),
            select(
                Corpus.id,
                Corpus.source_type,
                Corpus.content_hash,
                PullRequest.repository_id.label("repository_id"),
            )
            .join(PullRequest, PullRequest.id == Corpus.source_id)
            .where(
                Corpus.source_type == "pull_request",
                PullRequest.repository_id.in_(repository_ids),
            ),
            select(
                Corpus.id,
                Corpus.source_type,
                Corpus.content_hash,
                IssueComment.repository_id.label("repository_id"),
            )
            .join(IssueComment, IssueComment.id == Corpus.source_id)
            .where(
                Corpus.source_type == "issue_comment",
                IssueComment.repository_id.in_(repository_ids),
            ),
            select(
                Corpus.id,
                Corpus.source_type,
                Corpus.content_hash,
                PullRequestComment.repository_id.label("repository_id"),
            )
            .join(PullRequestComment, PullRequestComment.id == Corpus.source_id)
            .where(
                Corpus.source_type == "pr_issue_comment",
                PullRequestComment.repository_id.in_(repository_ids),
                PullRequestComment.comment_type == "issue_comment",
            ),
            select(
                Corpus.id,
                Corpus.source_type,
                Corpus.content_hash,
                PullRequestComment.repository_id.label("repository_id"),
            )
            .join(PullRequestComment, PullRequestComment.id == Corpus.source_id)
            .where(
                Corpus.source_type == "pr_review_comment",
                PullRequestComment.repository_id.in_(repository_ids),
                PullRequestComment.comment_type == "review_comment",
            ),
        ]
        candidate_filters = [Corpus.duplicate_of_id.is_(None)]
        if cleaning_version is not None:
            candidate_filters.append(Corpus.cleaning_version == cleaning_version)
        if max_model_input_chars is not None:
            candidate_filters.append(
                Corpus.model_input_chars <= max_model_input_chars
            )
        statement = union_all(
            *(query.where(*candidate_filters) for query in queries)
        )
        with self.sessions() as session:
            result = session.execute(
                statement.execution_options(yield_per=batch_size)
            ).mappings()
            while rows := result.fetchmany(batch_size):
                yield [dict(row) for row in rows]

    def available_cleaning_versions(self) -> tuple[str, ...]:
        with self.sessions() as session:
            versions = session.scalars(
                select(Corpus.cleaning_version)
                .distinct()
                .order_by(Corpus.cleaning_version)
            ).all()
        return tuple(versions)

    def prepare_sample_set(
        self,
        name: str,
        per_repository_limit: int,
        seed: str,
        cleaning_version: str,
        max_model_input_chars: int | None,
    ) -> tuple[int, dict[str, Any] | None]:
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(CorpusSampleSet).where(CorpusSampleSet.name == name)
            )
            if existing is not None:
                existing_cleaning_version = existing.cleaning_version or (
                    existing.stats or {}
                ).get("cleaning_version")
                if (
                    existing.per_repository_limit != per_repository_limit
                    or existing.seed != seed
                    or (
                        existing_cleaning_version is not None
                        and existing_cleaning_version != cleaning_version
                    )
                    or existing.max_model_input_chars != max_model_input_chars
                ):
                    raise ValueError(
                        f"采样集 {name!r} 已存在但参数不同；请使用新的采样集名称"
                    )
                existing.cleaning_version = cleaning_version
                if existing.status == "completed":
                    return existing.id, dict(existing.stats)
                session.execute(
                    delete(CorpusSampleItem).where(
                        CorpusSampleItem.sample_set_id == existing.id
                    )
                )
                existing.status = "building"
                existing.stats = {}
                existing.updated_at = utcnow()
                return existing.id, None
            sample_set = CorpusSampleSet(
                name=name,
                per_repository_limit=per_repository_limit,
                seed=seed,
                cleaning_version=cleaning_version,
                max_model_input_chars=max_model_input_chars,
                status="building",
                stats={},
            )
            session.add(sample_set)
            session.flush()
            return sample_set.id, None

    def insert_sample_items(
        self, rows: list[dict[str, Any]], batch_size: int = 2000
    ) -> int:
        written = 0
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            with self.sessions.begin() as session:
                session.execute(CorpusSampleItem.__table__.insert(), batch)
            written += len(batch)
        return written

    def finish_sample_set(
        self,
        sample_set_id: int,
        status: str,
        stats: dict[str, Any],
    ) -> None:
        with self.sessions.begin() as session:
            sample_set = session.get(CorpusSampleSet, sample_set_id)
            if sample_set is None:
                raise ValueError(f"采样集不存在: id={sample_set_id}")
            sample_set.status = status
            sample_set.stats = stats
            sample_set.updated_at = utcnow()

    def completed_sample_set_id(self, name: str) -> int:
        with self.sessions() as session:
            sample_set = session.scalar(
                select(CorpusSampleSet).where(CorpusSampleSet.name == name)
            )
            if sample_set is None:
                raise ValueError(f"采样集不存在: {name}")
            if sample_set.status != "completed":
                raise ValueError(f"采样集尚未构建完成: {name} ({sample_set.status})")
            return sample_set.id

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
                "model_input_chars",
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
        *,
        sample_set_id: int | None = None,
        cleaning_version: str | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        last_id = 0
        while True:
            already = select(LlmAnnotation.corpus_id).where(
                LlmAnnotation.taxonomy_version == taxonomy_version,
                LlmAnnotation.prompt_version == prompt_version,
                LlmAnnotation.model_name == model_name,
                LlmAnnotation.status == "succeeded",
            )
            query = select(Corpus.id, Corpus.model_input)
            if sample_set_id is not None:
                query = query.join(
                    CorpusSampleItem,
                    CorpusSampleItem.corpus_id == Corpus.id,
                ).where(CorpusSampleItem.sample_set_id == sample_set_id)
            elif cleaning_version is not None:
                query = query.where(Corpus.cleaning_version == cleaning_version)
            with self.sessions() as session:
                rows = (
                    session.execute(
                        query.where(
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
        self.save_annotations([row])

    def save_annotations(self, rows: list[dict[str, Any]]) -> None:
        updated_at = utcnow()
        prepared = [{**row, "updated_at": updated_at} for row in rows]
        self._upsert(
            LlmAnnotation,
            prepared,
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
