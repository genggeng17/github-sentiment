"""Server-side entry point for the BERTopic pilot."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from config import Settings
from llm_labeler.prompts import PROMPT_VERSION
from storage import Storage
from taxonomy import ASPECTS, TAXONOMY_VERSION

from .quarterly import QuarterlyTopicAnalysis
from .service import TopicAnalysis


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按自然季度汇总非空方面标注并运行 BERTopic")
    parser.add_argument("action", choices=["inspect", "fit"])
    parser.add_argument("--mode", choices=["quarterly", "aspect"], default="quarterly")
    parser.add_argument("--model-name", help="仅用于旧 aspect 模式")
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--taxonomy-version", default=TAXONOMY_VERSION)
    parser.add_argument("--cleaning-version", help="仅用于旧 aspect 模式")
    parser.add_argument("--language", choices=["en"], help="仅用于旧 aspect 模式，默认 en")
    parser.add_argument("--aspect", action="append", choices=sorted(ASPECTS))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=positive_int, default=32)
    parser.add_argument("--cpu-threads", type=positive_int, default=2)
    parser.add_argument("--max-chars", type=positive_int, default=4000)
    parser.add_argument("--min-chars", type=positive_int, default=12)
    parser.add_argument("--max-docs-per-aspect", type=positive_int)
    parser.add_argument("--min-documents", type=positive_int, default=60)
    parser.add_argument("--min-topic-size", type=positive_int, default=20)
    parser.add_argument("--min-samples", type=positive_int, default=5)
    parser.add_argument("--umap-neighbors", type=positive_int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("data/topics"))
    parser.add_argument("--name", help="inspect 快照名或 fit 运行名；默认使用 UTC 时间戳")
    parser.add_argument("--snapshot", help="fit 要读取的 inspect 快照名")
    parser.add_argument("--resume", action="store_true", help="恢复已有季度运行，复用已完成季度")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "aspect" and (not args.model_name or not args.cleaning_version):
        parser.error("aspect 模式必须指定 --model-name 和 --cleaning-version")
    if args.mode == "quarterly" and (
        args.model_name or args.cleaning_version or args.aspect or args.max_docs_per_aspect
        or args.language
        or args.prompt_version != PROMPT_VERSION or args.taxonomy_version != TAXONOMY_VERSION
    ):
        parser.error("quarterly 模式读取所有版本且不分方面，不能指定模型/版本/方面筛选或抽样")
    if args.resume and (args.mode != "quarterly" or args.action != "fit"):
        parser.error("--resume 只用于季度 fit")
    if args.action == "fit" and not args.snapshot:
        parser.error("fit 必须指定 --snapshot，即 inspect 输出的 snapshot_name")
    if args.action == "inspect" and args.snapshot:
        parser.error("--snapshot 只用于 fit")
    if args.umap_neighbors < 2:
        raise ValueError("--umap-neighbors 必须至少为 2")
    if args.min_topic_size < 2:
        parser.error("--min-topic-size 必须至少为 2")
    if args.min_chars > args.max_chars:
        parser.error("--min-chars 不能大于 --max-chars")
    topic_logger = logging.getLogger("topic_analysis")
    if not any(handler.name == "topic_analysis_cli" for handler in topic_logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name("topic_analysis_cli")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        topic_logger.addHandler(handler)
    topic_logger.setLevel(logging.INFO)
    topic_logger.propagate = False
    config = {
        "taxonomy_version": args.taxonomy_version,
        "prompt_version": args.prompt_version,
        "model_name": args.model_name,
        "cleaning_version": args.cleaning_version,
        "language": args.language or "en",
        "aspects": sorted(set(args.aspect or ASPECTS)),
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
        "cpu_threads": args.cpu_threads,
        "max_chars": args.max_chars,
        "min_chars": args.min_chars,
        "max_docs_per_aspect": args.max_docs_per_aspect,
        "min_documents": args.min_documents,
        "min_topic_size": args.min_topic_size,
        "min_samples": args.min_samples,
        "umap_neighbors": args.umap_neighbors,
        "seed": args.seed,
    }
    if args.mode == "quarterly":
        config = {key: config[key] for key in (
            "embedding_model", "embedding_batch_size", "cpu_threads", "max_chars", "min_chars",
            "min_documents", "min_topic_size", "min_samples", "umap_neighbors", "seed",
        )}
    repo_root = Path(__file__).resolve().parents[1]
    if args.action == "inspect":
        storage = Storage(Settings.from_env().database_url)
        analysis = (QuarterlyTopicAnalysis if args.mode == "quarterly" else TopicAnalysis)(
            storage, repo_root,
        )
        snapshot_name = args.name or datetime.now(UTC).strftime("corpus-%Y%m%dT%H%M%SZ")
        with storage.pipeline_lock():
            result = analysis.inspect(
                config, output_root=args.output_root, snapshot_name=snapshot_name
            )
    else:
        run_name = args.name or datetime.now(UTC).strftime("bertopic-%Y%m%dT%H%M%SZ")
        analysis = (QuarterlyTopicAnalysis if args.mode == "quarterly" else TopicAnalysis)(
            None, repo_root,
        )
        result = analysis.fit(
            config,
            output_root=args.output_root,
            run_name=run_name,
            snapshot_name=args.snapshot,
            **({"resume": args.resume} if args.mode == "quarterly" else {}),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
