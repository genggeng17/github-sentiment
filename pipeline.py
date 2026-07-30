from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

from config import Settings, normalize_repository_name
from corpus_builder import CorpusBuilder
from crawler import CollectionLimits, GitHubClient, GitHubCollector
from llm_labeler.service import DeepSeekClient, DeepSeekLabeler
from sampler import CorpusSampler
from storage import Storage
from storage.models import RunStatus

logger = logging.getLogger(__name__)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return parsed


def add_collection_limit_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-issues",
        type=non_negative_int,
        default=None,
        help="每个仓库本次最多写入的 Issue 数；默认不限制",
    )
    parser.add_argument(
        "--max-pull-requests",
        type=non_negative_int,
        default=None,
        help="每个仓库本次最多写入的 PR 数；默认不限制",
    )
    parser.add_argument(
        "--max-issue-comments",
        type=non_negative_int,
        default=None,
        help="每个仓库本次最多写入的 Issue 评论数；默认不限制",
    )
    parser.add_argument(
        "--max-pr-comments",
        type=non_negative_int,
        default=None,
        help="每个仓库本次最多写入的 PR 普通讨论评论数；默认不限制",
    )
    parser.add_argument(
        "--max-review-comments",
        type=non_negative_int,
        default=None,
        help="每个仓库本次最多写入的 PR Review 评论数；默认不限制",
    )


def collection_limits_from_args(args: argparse.Namespace) -> CollectionLimits:
    return CollectionLimits(
        issues=getattr(args, "max_issues", None),
        pull_requests=getattr(args, "max_pull_requests", None),
        issue_comments=getattr(args, "max_issue_comments", None),
        pr_issue_comments=getattr(args, "max_pr_comments", None),
        pr_review_comments=getattr(args, "max_review_comments", None),
    )


class Pipeline:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage

    def collect(
        self,
        run_id: str,
        *,
        limits: CollectionLimits | None = None,
    ) -> dict[str, Any]:
        self.settings.require_collection()
        enabled_repositories = self.storage.enabled_repository_names()
        if not enabled_repositories:
            raise ValueError("没有启用的采集仓库，请先执行: python pipeline.py repo add owner/repo")
        repositories: dict[str, Any] = {}
        with GitHubClient(
            self.settings.github_token,
            base_url=self.settings.github_api_url,
            api_version=self.settings.github_api_version,
            timeout_seconds=self.settings.http_timeout_seconds,
            max_retries=self.settings.http_max_retries,
        ) as client:
            collector = GitHubCollector(
                client,
                self.storage,
                cursor_overlap_seconds=self.settings.cursor_overlap_seconds,
                limits=limits,
            )
            for full_name in enabled_repositories:
                logger.info("开始采集 %s", full_name)
                repositories[full_name] = collector.collect_repository(
                    full_name,
                    run_id,
                    heartbeat=lambda: self.storage.heartbeat(run_id),
                )
        failed = sum(
            stream["status"] == "failed"
            for repository in repositories.values()
            for stream in repository.values()
        )
        return {"repositories": repositories, "failed_streams": failed}

    def build_corpus(self) -> dict[str, int]:
        return CorpusBuilder(self.storage).build()

    def build_sample(
        self,
        name: str,
        *,
        per_repository_limit: int = 5000,
        seed: str = "0",
    ) -> dict[str, Any]:
        return CorpusSampler(self.storage).build(
            name,
            per_repository_limit=per_repository_limit,
            seed=seed,
        )

    def label(
        self,
        *,
        limit: int | None = None,
        sample_name: str | None = None,
    ) -> dict[str, int]:
        self.settings.require_labeling()
        sample_set_id = (
            self.storage.completed_sample_set_id(sample_name) if sample_name else None
        )
        client = DeepSeekClient(
            self.settings.deepseek_api_key,
            self.settings.deepseek_base_url,
            self.settings.deepseek_model,
            timeout_seconds=max(60, self.settings.http_timeout_seconds),
            max_retries=self.settings.http_max_retries,
        )
        try:
            return DeepSeekLabeler(
                client, self.storage, batch_size=self.settings.label_batch_size
            ).label_pending(limit=limit, sample_set_id=sample_set_id)
        finally:
            client.close()

    def run_all(
        self,
        run_id: str,
        *,
        skip_label: bool = False,
        limits: CollectionLimits | None = None,
        sample_name: str | None = None,
        sample_per_repository: int = 5000,
        sample_seed: str = "0",
    ) -> dict[str, Any]:
        collection = (
            self.collect(run_id)
            if limits is None
            else self.collect(run_id, limits=limits)
        )
        stats: dict[str, Any] = {"collection": collection}
        if collection["failed_streams"]:
            logger.error("采集存在失败流，已跳过 corpus 构建和 LLM 标注")
            return stats
        stats["corpus"] = self.build_corpus()
        if sample_name:
            stats["sample"] = self.build_sample(
                sample_name,
                per_repository_limit=sample_per_repository,
                seed=sample_seed,
            )
        if not skip_label:
            stats["llm_labeling"] = (
                self.label() if sample_name is None else self.label(sample_name=sample_name)
            )
        return stats


def tracked_run(
    storage: Storage,
    run_type: str,
    callback: Callable[[str], dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    with storage.pipeline_lock():
        run_id = storage.start_pipeline_run(run_type)
        try:
            stats = callback(run_id)
            has_collection_failures = bool(stats.get("failed_streams")) or bool(
                stats.get("collection", {}).get("failed_streams")
            )
            has_label_failures = bool(stats.get("failed")) or bool(
                stats.get("llm_labeling", {}).get("failed")
            )
            status = (
                RunStatus.PARTIAL.value
                if has_collection_failures or has_label_failures
                else RunStatus.SUCCEEDED.value
            )
            storage.finish_pipeline_run(run_id, status, stats)
            return run_id, status, stats
        except Exception as exc:
            storage.finish_pipeline_run(run_id, RunStatus.FAILED.value, {}, str(exc))
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GitHub Rust 社区情感分析流水线")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="创建 MySQL 表")
    collect = subparsers.add_parser("collect", help="执行历史回填或增量采集")
    add_collection_limit_arguments(collect)
    subparsers.add_parser("build-corpus", help="清洗原始数据并更新统一语料")
    sample = subparsers.add_parser("sample", help="按仓库构建版本化语料采样集")
    sample.add_argument("--name", required=True, help="不可变的采样集名称")
    sample.add_argument(
        "--per-repository",
        type=positive_int,
        default=5000,
        help="每个启用仓库最多抽取的语料数（默认 5000）",
    )
    sample.add_argument(
        "--seed",
        default="0",
        help="确定性抽样种子；相同候选集和种子会产生相同结果",
    )
    label = subparsers.add_parser("label", help="标注尚未成功标注的语料")
    label.add_argument(
        "--limit",
        type=positive_int,
        default=None,
        help="本次最多尝试标注的语料数（默认处理全部待标注语料）",
    )
    label.add_argument(
        "--sample",
        default=None,
        help="只标注指定的已完成采样集；默认沿用旧行为处理全部语料",
    )
    run = subparsers.add_parser("run", help="执行采集、语料构建和 DeepSeek 标注")
    run.add_argument("--skip-label", action="store_true", help="跳过 DeepSeek 标注")
    add_collection_limit_arguments(run)
    run.add_argument(
        "--sample-name",
        default=None,
        help="构建并只标注指定采样集；不传时沿用旧的全量标注行为",
    )
    run.add_argument(
        "--sample-per-repository",
        type=positive_int,
        default=5000,
        help="每仓库采样上限（默认 5000）",
    )
    run.add_argument("--sample-seed", default="0", help="确定性抽样种子")
    status = subparsers.add_parser("status", help="查询最近运行记录")
    status.add_argument("--limit", type=int, default=10)
    repo = subparsers.add_parser("repo", help="管理数据库中的仓库白名单")
    repo_commands = repo.add_subparsers(dest="repo_command", required=True)
    repo_commands.add_parser("list", help="列出全部仓库及启用状态")
    repo_add = repo_commands.add_parser("add", help="添加仓库并启用")
    repo_add.add_argument("full_name", help="owner/repo")
    repo_enable = repo_commands.add_parser("enable", help="重新启用已有仓库")
    repo_enable.add_argument("full_name", help="owner/repo")
    repo_disable = repo_commands.add_parser("disable", help="停用仓库但保留历史数据")
    repo_disable.add_argument("full_name", help="owner/repo")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    storage = Storage(settings.database_url)
    pipeline = Pipeline(settings, storage)

    if args.command == "init-db":
        storage.create_schema()
        imported = storage.bootstrap_repositories(settings.repositories)
        print(f"数据库表已创建；首次导入仓库数: {imported}")
        return 0
    if args.command == "status":
        print(json.dumps(storage.recent_runs(max(1, args.limit)), ensure_ascii=False, indent=2))
        return 0
    if args.command == "repo":
        if args.repo_command == "list":
            print(json.dumps(storage.list_repositories(), ensure_ascii=False, indent=2))
            return 0
        full_name = normalize_repository_name(args.full_name)
        if args.repo_command == "add":
            repository_id = storage.ensure_repository(full_name)
            print(f"仓库已添加并启用: {full_name} (id={repository_id})")
            return 0
        enabled = args.repo_command == "enable"
        changed = storage.set_repository_enabled(full_name, enabled)
        state = "启用" if enabled else "停用"
        suffix = "" if changed else "（状态未变化）"
        print(f"仓库已{state}: {full_name}{suffix}")
        return 0

    limits = collection_limits_from_args(args)
    callbacks: dict[str, Callable[[str], dict[str, Any]]] = {
        "collect": lambda run_id: pipeline.collect(run_id, limits=limits),
        "build-corpus": lambda _run_id: pipeline.build_corpus(),
        "sample": lambda _run_id: pipeline.build_sample(
            args.name,
            per_repository_limit=args.per_repository,
            seed=args.seed,
        ),
        "label": lambda _run_id: pipeline.label(
            limit=args.limit,
            sample_name=args.sample,
        ),
        "run": lambda run_id: pipeline.run_all(
            run_id,
            skip_label=args.skip_label,
            limits=limits,
            sample_name=args.sample_name,
            sample_per_repository=args.sample_per_repository,
            sample_seed=args.sample_seed,
        ),
    }
    run_id, status, stats = tracked_run(storage, args.command, callbacks[args.command])
    print(
        json.dumps(
            {"run_id": run_id, "status": status, "stats": stats}, ensure_ascii=False, indent=2
        )
    )
    return 0 if status == RunStatus.SUCCEEDED.value else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
