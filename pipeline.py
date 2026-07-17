from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

from config import Settings
from corpus_builder import CorpusBuilder
from crawler import GitHubClient, GitHubCollector
from llm_labeler.service import DeepSeekClient, DeepSeekLabeler
from storage import Storage
from storage.models import RunStatus

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, settings: Settings, storage: Storage):
        self.settings = settings
        self.storage = storage

    def collect(self, run_id: str) -> dict[str, Any]:
        self.settings.require_collection()
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
            )
            for full_name in self.settings.repositories:
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

    def label(self) -> dict[str, int]:
        self.settings.require_labeling()
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
            ).label_pending()
        finally:
            client.close()

    def run_all(self, run_id: str, *, skip_label: bool = False) -> dict[str, Any]:
        stats: dict[str, Any] = {"collection": self.collect(run_id)}
        stats["corpus"] = self.build_corpus()
        if not skip_label:
            stats["llm_labeling"] = self.label()
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
    subparsers.add_parser("collect", help="执行历史回填或增量采集")
    subparsers.add_parser("build-corpus", help="清洗原始数据并更新统一语料")
    subparsers.add_parser("label", help="标注尚未成功标注的语料")
    run = subparsers.add_parser("run", help="执行采集、语料构建和 DeepSeek 标注")
    run.add_argument("--skip-label", action="store_true", help="跳过 DeepSeek 标注")
    status = subparsers.add_parser("status", help="查询最近运行记录")
    status.add_argument("--limit", type=int, default=10)
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
        print("数据库表已创建")
        return 0
    if args.command == "status":
        print(json.dumps(storage.recent_runs(max(1, args.limit)), ensure_ascii=False, indent=2))
        return 0

    callbacks: dict[str, Callable[[str], dict[str, Any]]] = {
        "collect": pipeline.collect,
        "build-corpus": lambda _run_id: pipeline.build_corpus(),
        "label": lambda _run_id: pipeline.label(),
        "run": lambda run_id: pipeline.run_all(run_id, skip_label=args.skip_label),
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
