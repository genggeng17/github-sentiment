"""Run one reproducible BERTopic snapshot per confirmed aspect."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from taxonomy import ASPECTS

from .data import (
    TOPIC_PREPROCESSING_VERSION,
    TopicDocument,
    load_documents,
    parse_sentiments,
    prepare_text,
    select_aspect_members,
)
from .reporting import build_rollups, write_csv

_RUN_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
_DOMAIN_STOP_WORDS = {
    "rust",
    "github",
    "issue",
    "issues",
    "pull",
    "request",
    "requests",
    "code",
    "example",
    "use",
    "using",
    "used",
    "like",
    "would",
    "could",
    "also",
    "one",
    "need",
    "needs",
}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_output_root(repo_root: Path, output_root: Path) -> Path:
    resolved = (output_root if output_root.is_absolute() else repo_root / output_root).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise ValueError("topic 输出目录必须位于本仓库内")
    return resolved


def _topic_words(model: Any, topic_id: int, count: int = 12) -> list[str]:
    if topic_id == -1:
        return []
    terms = model.get_topic(topic_id)
    return [str(word) for word, _score in (terms or [])[:count]]


def _review_examples(
    model: Any,
    topic_id: int,
    ids: list[int],
    documents: dict[int, TopicDocument],
    *,
    per_kind: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    by_text: dict[str, list[int]] = defaultdict(list)
    for corpus_id in ids:
        by_text[documents[corpus_id].text].append(corpus_id)
    representatives: list[int] = []
    if topic_id != -1:
        for text in model.get_representative_docs(topic_id) or []:
            for corpus_id in by_text.get(text, []):
                if corpus_id not in representatives:
                    representatives.append(corpus_id)
                    break
            if len(representatives) >= per_kind:
                break
    samples = sorted(
        (corpus_id for corpus_id in ids if corpus_id not in representatives),
        key=lambda corpus_id: (corpus_id * 2654435761) % (2**32),
    )[:per_kind]
    rows = []
    for kind, chosen in (("representative", representatives), ("sample", samples)):
        for corpus_id in chosen:
            document = documents[corpus_id]
            rows.append(
                {
                    "corpus_id": corpus_id,
                    "kind": kind,
                    "repository": document.repository,
                    "created_at": document.created_at.isoformat(),
                    "source_type": document.source_type,
                    "github_url": document.github_url,
                    "text": document.text[:1200],
                }
            )
    return representatives, rows


class TopicAnalysis:
    def __init__(self, storage: Any, repo_root: Path):
        self.storage = storage
        self.repo_root = repo_root.resolve()

    def _load(self, config: dict[str, Any]) -> tuple[dict[int, TopicDocument], dict[str, Any]]:
        return load_documents(
            self.storage,
            taxonomy_version=config["taxonomy_version"],
            prompt_version=config["prompt_version"],
            model_name=config["model_name"],
            cleaning_version=config["cleaning_version"],
            language=config["language"],
            max_chars=config["max_chars"],
            min_chars=config["min_chars"],
        )

    def inspect(self, config: dict[str, Any]) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        stats: Counter[str] = Counter()
        seen_ids: set[int] = set()
        for batch in self.storage.iter_confirmed_topic_documents(
            taxonomy_version=config["taxonomy_version"],
            prompt_version=config["prompt_version"],
            model_name=config["model_name"],
            cleaning_version=config["cleaning_version"],
            language=config["language"],
            include_source_metadata=False,
        ):
            for row in batch:
                stats["successful_rows"] += 1
                corpus_id = row["corpus_id"]
                sentiments = parse_sentiments(row["annotations"], corpus_id)
                if not sentiments:
                    stats["empty_aspect_rows"] += 1
                    continue
                text, truncated = prepare_text(row["clean_text"], config["max_chars"])
                if len(text) < config["min_chars"]:
                    stats["too_short_rows"] += 1
                    continue
                stats["truncated_rows"] += int(truncated)
                if corpus_id in seen_ids:
                    raise RuntimeError(f"重复读取 corpus_id={corpus_id}")
                seen_ids.add(corpus_id)
                counts.update(sentiments.keys())
        stats["usable_rows"] = len(seen_ids)
        return {
            "source": {
                key: config[key]
                for key in (
                    "taxonomy_version",
                    "prompt_version",
                    "model_name",
                    "cleaning_version",
                    "language",
                )
            },
            "selection": dict(stats),
            "aspect_counts": {aspect: counts[aspect] for aspect in sorted(ASPECTS)},
        }

    def fit(
        self,
        config: dict[str, Any],
        *,
        output_root: Path,
        run_name: str,
    ) -> dict[str, Any]:
        if not _RUN_NAME.fullmatch(run_name):
            raise ValueError("运行名称只能含字母、数字、下划线和连字符，长度不超过 80")
        root = _safe_output_root(self.repo_root, output_root)
        destination = root / run_name
        staging = root / f"{run_name}.building"
        if destination.exists() or staging.exists():
            raise ValueError(f"输出目录已存在: {destination} 或 {staging}")

        documents, stats = self._load(config)
        members = select_aspect_members(
            documents,
            aspects=tuple(config["aspects"]),
            max_per_aspect=config["max_docs_per_aspect"],
            seed=config["seed"],
        )
        skipped = {
            aspect: len(members.get(aspect, []))
            for aspect in config["aspects"]
            if len(members.get(aspect, []))
            < max(config["min_documents"], 2 * config["min_topic_size"])
        }
        eligible = {aspect: ids for aspect, ids in members.items() if aspect not in skipped}
        if not eligible:
            raise ValueError("没有达到最小语料量的方面；先用 inspect 核对标注版本和数量")

        try:
            import numpy as np
            import torch
            from bertopic import BERTopic
            from hdbscan import HDBSCAN
            from sentence_transformers import SentenceTransformer
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
            from umap import UMAP
        except ImportError as exc:
            raise RuntimeError(
                "缺少 BERTopic 依赖；请在服务器运行 pip install -e '.[topics]'"
            ) from exc

        root.mkdir(parents=True, exist_ok=True)
        staging.mkdir()
        (staging / "models").mkdir()
        torch.set_num_threads(config["cpu_threads"])
        unique_ids = sorted({corpus_id for ids in eligible.values() for corpus_id in ids})
        input_digest = hashlib.sha256()
        for aspect, ids in sorted(eligible.items()):
            for corpus_id in ids:
                document = documents[corpus_id]
                material = (
                    f"{aspect}\0{corpus_id}\0{document.content_hash}\0"
                    f"{document.sentiments[aspect]}\n"
                )
                input_digest.update(material.encode())
        index = {corpus_id: offset for offset, corpus_id in enumerate(unique_ids)}
        embedder = SentenceTransformer(config["embedding_model"], device="cpu")
        embedder.save(str(staging / "encoder"))
        embeddings = embedder.encode(
            [documents[corpus_id].text for corpus_id in unique_ids],
            batch_size=config["embedding_batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        np.save(staging / "embeddings.npy", embeddings)
        _write_json(staging / "embedding_corpus_ids.json", {"corpus_ids": unique_ids})

        assignments: list[dict[str, Any]] = []
        review_rows: list[dict[str, Any]] = []
        topic_details: dict[str, list[dict[str, Any]]] = {}
        aspect_stats: dict[str, dict[str, Any]] = {}
        for aspect in config["aspects"]:
            if aspect in skipped:
                aspect_stats[aspect] = {"status": "insufficient", "documents": skipped[aspect]}
                continue
            ids = eligible[aspect]
            texts = [documents[corpus_id].text for corpus_id in ids]
            aspect_embeddings = embeddings[[index[corpus_id] for corpus_id in ids]]
            vectorizer = CountVectorizer(
                stop_words=sorted(ENGLISH_STOP_WORDS | _DOMAIN_STOP_WORDS),
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            )
            reducer = UMAP(
                n_neighbors=config["umap_neighbors"],
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                low_memory=True,
                random_state=config["seed"],
            )
            clusterer = HDBSCAN(
                min_cluster_size=config["min_topic_size"],
                min_samples=config["min_samples"],
                metric="euclidean",
                prediction_data=True,
                core_dist_n_jobs=1,
            )
            model = BERTopic(
                embedding_model=embedder,
                umap_model=reducer,
                hdbscan_model=clusterer,
                vectorizer_model=vectorizer,
                calculate_probabilities=False,
                low_memory=True,
                verbose=True,
            )
            topic_ids, _probabilities = model.fit_transform(texts, embeddings=aspect_embeddings)
            by_topic: dict[int, list[int]] = defaultdict(list)
            for corpus_id, topic_id in zip(ids, topic_ids, strict=True):
                document = documents[corpus_id]
                topic_id = int(topic_id)
                by_topic[topic_id].append(corpus_id)
                assignments.append(
                    {
                        "corpus_id": corpus_id,
                        "aspect": aspect,
                        "topic_id": topic_id,
                        "sentiment": document.sentiments[aspect],
                        "month": document.created_at.strftime("%Y-%m"),
                        "created_at": document.created_at.isoformat(),
                        "repository": document.repository,
                        "source_type": document.source_type,
                        "source_id": document.source_id,
                        "parent_id": document.parent_id,
                        "github_url": document.github_url,
                    }
                )
            details = []
            for topic_id, topic_corpus_ids in sorted(by_topic.items()):
                representative_ids, examples = _review_examples(
                    model, topic_id, topic_corpus_ids, documents, per_kind=5
                )
                details.append(
                    {
                        "topic_id": topic_id,
                        "words": _topic_words(model, topic_id),
                        "representative_corpus_ids": representative_ids,
                    }
                )
                for example in examples:
                    review_rows.append({"aspect": aspect, "topic_id": topic_id, **example})
            topic_details[aspect] = details
            aspect_stats[aspect] = {
                "status": "trained",
                "documents": len(ids),
                "topics_excluding_outliers": sum(topic_id != -1 for topic_id in by_topic),
                "outlier_count": len(by_topic.get(-1, [])),
                "outlier_share": round(len(by_topic.get(-1, [])) / len(ids), 6),
            }
            # Keep UMAP/HDBSCAN; future inference supplies embeddings from the
            # separately versioned encoder rather than duplicating it 13 times.
            model.save(
                str(staging / "models" / f"{aspect}.pkl"),
                serialization="pickle",
                save_embedding_model=False,
            )
            del model, reducer, clusterer, vectorizer, aspect_embeddings
            gc.collect()

        rollups = build_rollups(assignments)
        details_by_key = {
            (aspect, item["topic_id"]): item
            for aspect, items in topic_details.items()
            for item in items
        }
        topic_rows = []
        for row in rollups["topics"]:
            detail = details_by_key[(row["aspect"], row["topic_id"])]
            topic_rows.append(
                {
                    **row,
                    "words": ", ".join(detail["words"]),
                    "representative_corpus_ids": ",".join(
                        str(item) for item in detail["representative_corpus_ids"]
                    ),
                }
            )
        write_csv(
            staging / "assignments.csv",
            [
                "corpus_id",
                "aspect",
                "topic_id",
                "sentiment",
                "month",
                "created_at",
                "repository",
                "source_type",
                "source_id",
                "parent_id",
                "github_url",
            ],
            assignments,
        )
        write_csv(
            staging / "topics.csv",
            [
                "aspect",
                "topic_id",
                "document_count",
                "aspect_total",
                "share_within_aspect",
                "words",
                "representative_corpus_ids",
            ],
            topic_rows,
        )
        for key, filename, fields in (
            (
                "sentiment",
                "topic_sentiment.csv",
                [
                    "aspect",
                    "topic_id",
                    "sentiment",
                    "document_count",
                    "topic_total",
                    "share_within_topic",
                ],
            ),
            (
                "time",
                "topic_time_sentiment.csv",
                [
                    "aspect",
                    "month",
                    "topic_id",
                    "sentiment",
                    "document_count",
                    "aspect_month_total",
                    "share_within_aspect_month",
                ],
            ),
            (
                "repository",
                "topic_repository_sentiment.csv",
                [
                    "aspect",
                    "repository",
                    "topic_id",
                    "sentiment",
                    "document_count",
                    "aspect_repository_total",
                    "share_within_aspect_repository",
                ],
            ),
            (
                "time_repository",
                "topic_time_repository_sentiment.csv",
                [
                    "aspect",
                    "month",
                    "repository",
                    "topic_id",
                    "sentiment",
                    "document_count",
                    "aspect_month_repository_total",
                    "share_within_aspect_month_repository",
                ],
            ),
        ):
            write_csv(staging / filename, fields, rollups[key])
        write_csv(
            staging / "review.csv",
            [
                "aspect",
                "topic_id",
                "corpus_id",
                "kind",
                "repository",
                "created_at",
                "source_type",
                "github_url",
                "text",
            ],
            review_rows,
        )
        manifest = {
            "run_name": run_name,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": config,
            "topic_preprocessing_version": TOPIC_PREPROCESSING_VERSION,
            "selected_input_sha256": input_digest.hexdigest(),
            "selection": stats,
            "selected_unique_documents": len(unique_ids),
            "aspect_stats": aspect_stats,
            "topic_details": topic_details,
            "software": {
                name: version(name)
                for name in (
                    "bertopic",
                    "sentence-transformers",
                    "umap-learn",
                    "hdbscan",
                    "scikit-learn",
                    "numpy",
                )
            },
            "encoder_snapshot": "encoder/",
            "note": "英文已标注样本；时间采用来源创建日期；带抽样上限时趋势不代表全量。",
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return {
            "output_dir": str(destination),
            "selected_unique_documents": len(unique_ids),
            "aspect_stats": aspect_stats,
        }
