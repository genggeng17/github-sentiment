from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


LONG_TEXT = Text().with_variant(MEDIUMTEXT(), "mysql")
ID_TYPE = BigInteger().with_variant(Integer(), "sqlite")


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class RunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


class Repository(Base):
    __tablename__ = "repositories"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    full_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class RepositoryCursor(Base):
    __tablename__ = "repository_cursors"
    __table_args__ = (UniqueConstraint("repository_id", "stream", name="uq_cursor_repo_stream"),)

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"), nullable=False)
    stream: Mapped[str] = mapped_column(String(40), nullable=False)
    last_updated_at: Mapped[datetime | None] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class RawMixin:
    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    github_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"), nullable=False)
    author_login: Mapped[str | None] = mapped_column(String(255))
    author_github_id: Mapped[int | None] = mapped_column(BigInteger)
    body: Mapped[str | None] = mapped_column(LONG_TEXT)
    github_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    collected_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class Issue(RawMixin, Base):
    __tablename__ = "issues"
    __table_args__ = (
        UniqueConstraint("github_id", name="uq_issue_github_id"),
        UniqueConstraint("repository_id", "number", name="uq_issue_repo_number"),
    )

    number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)


class PullRequest(RawMixin, Base):
    __tablename__ = "pull_requests"
    __table_args__ = (
        UniqueConstraint("github_id", name="uq_pr_github_id"),
        UniqueConstraint("repository_id", "number", name="uq_pr_repo_number"),
    )

    number: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(1024), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)
    merged_at: Mapped[datetime | None] = mapped_column(DateTime)


class IssueComment(RawMixin, Base):
    __tablename__ = "issue_comments"
    __table_args__ = (UniqueConstraint("github_id", name="uq_issue_comment_github_id"),)

    issue_id: Mapped[int] = mapped_column(ForeignKey("issues.id"), nullable=False)


class PullRequestComment(RawMixin, Base):
    __tablename__ = "pr_comments"
    __table_args__ = (
        UniqueConstraint("github_id", "comment_type", name="uq_pr_comment_github_type"),
    )

    pull_request_id: Mapped[int] = mapped_column(ForeignKey("pull_requests.id"), nullable=False)
    comment_type: Mapped[str] = mapped_column(String(30), nullable=False)
    path: Mapped[str | None] = mapped_column(String(1024))
    in_reply_to_github_id: Mapped[int | None] = mapped_column(BigInteger)


class UnresolvedCollectionItem(Base):
    __tablename__ = "unresolved_collection_items"
    __table_args__ = (
        UniqueConstraint(
            "repository_id", "stream", "item_key", name="uq_unresolved_repo_stream_item"
        ),
        Index(
            "ix_unresolved_pending", "repository_id", "stream", "category", "resolved_at"
        ),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"), nullable=False)
    stream: Mapped[str] = mapped_column(String(40), nullable=False)
    item_key: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(30), nullable=False)
    github_id: Mapped[int | None] = mapped_column(BigInteger)
    parent_number: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)


class Corpus(Base):
    __tablename__ = "corpus"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "source_id", "content_hash", name="uq_corpus_source_version"
        ),
        Index("ix_corpus_hash", "content_hash"),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False)
    source_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    parent_id: Mapped[int | None] = mapped_column(BigInteger)
    raw_text: Mapped[str] = mapped_column(LONG_TEXT, nullable=False)
    context_text: Mapped[str] = mapped_column(LONG_TEXT, nullable=False, default="")
    target_text: Mapped[str] = mapped_column(LONG_TEXT, nullable=False)
    model_input: Mapped[str] = mapped_column(LONG_TEXT, nullable=False)
    clean_text: Mapped[str] = mapped_column(LONG_TEXT, nullable=False)
    language: Mapped[str] = mapped_column(String(20), nullable=False, default="unknown")
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    cleaning_version: Mapped[str] = mapped_column(String(30), nullable=False)
    duplicate_of_id: Mapped[int | None] = mapped_column(ForeignKey("corpus.id"))
    source_updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class LlmAnnotation(Base):
    __tablename__ = "llm_annotations"
    __table_args__ = (
        UniqueConstraint(
            "corpus_id",
            "taxonomy_version",
            "prompt_version",
            "model_name",
            name="uq_llm_annotation_trace",
        ),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    corpus_id: Mapped[int] = mapped_column(ForeignKey("corpus.id"), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(30), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(30), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    raw_response: Mapped[str | None] = mapped_column(LONG_TEXT)
    parsed_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utcnow, onupdate=utcnow
    )


class BertPrediction(Base):
    __tablename__ = "bert_predictions"
    __table_args__ = (UniqueConstraint("corpus_id", "model_version", name="uq_bert_prediction"),)

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    corpus_id: Mapped[int] = mapped_column(ForeignKey("corpus.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    predictions: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_type: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default=RunStatus.RUNNING.value)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    stats: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(LONG_TEXT)


class CollectionStreamRun(Base):
    __tablename__ = "collection_stream_runs"
    __table_args__ = (
        UniqueConstraint("pipeline_run_id", "repository_id", "stream", name="uq_stream_run"),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    pipeline_run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id"), nullable=False)
    repository_id: Mapped[int] = mapped_column(ForeignKey("repositories.id"), nullable=False)
    stream: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    pages_read: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_read: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_written: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
