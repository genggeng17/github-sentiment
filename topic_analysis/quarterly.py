"""Pooled calendar-quarter BERTopic with frozen inputs and resumable outputs."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import logging
import os
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .data import TOPIC_PREPROCESSING_VERSION, TopicDocument, prepare_text
from .service import (
    _DOMAIN_STOP_WORDS,
    _RUN_NAME,
    _review_examples,
    _safe_output_root,
    _write_json,
)

logger = logging.getLogger(__name__)
FORMAT = "quarterly-topics-v1"
POLICY = "all-versions-latest-nonempty-success-updated-at-id-per-corpus-v1"


def quarter_bounds(value: datetime) -> tuple[str, datetime, datetime]:
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    month = (value.month - 1) // 3 * 3 + 1
    start = datetime(value.year, month, 1)
    end = datetime(value.year + 1, 1, 1) if month == 10 else datetime(value.year, month + 3, 1)
    return f"{value.year}-Q{(month - 1) // 3 + 1}", start, end


def _name(value: str) -> None:
    if not _RUN_NAME.fullmatch(value):
        raise ValueError("名称只能含字母、数字、下划线和连字符，长度不超过 80")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


@contextmanager
def _run_lock(path: Path):
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("该季度任务正在运行，请勿同时恢复同一运行") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


class _Backend:
    def __init__(self, config: dict[str, Any], destination: Path):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(config["cpu_threads"])
        self.encoder = SentenceTransformer(config["embedding_model"], device="cpu")
        if not (destination / "encoder").exists():
            self.encoder.save(str(destination / "encoder"))

    def encode(self, texts: list[str], config: dict[str, Any]):
        return self.encoder.encode(
            texts,
            batch_size=config["embedding_batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def cluster(self, texts, embeddings, config):
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
        from umap import UMAP

        model = BERTopic(
            embedding_model=self.encoder,
            umap_model=UMAP(
                n_neighbors=min(config["umap_neighbors"], len(texts) - 1),
                n_components=min(5, len(texts) - 2),
                min_dist=0.0,
                metric="cosine",
                low_memory=True,
                random_state=config["seed"],
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=config["min_topic_size"],
                min_samples=config["min_samples"],
                metric="euclidean",
                prediction_data=True,
                core_dist_n_jobs=1,
            ),
            vectorizer_model=CountVectorizer(
                stop_words=sorted(ENGLISH_STOP_WORDS | _DOMAIN_STOP_WORDS),
                ngram_range=(1, 2),
                # BERTopic fits this on aggregated topic documents. A sparse
                # quarter may contain only one topic (all outliers included).
                min_df=1,
                max_df=1.0,
            ),
            calculate_probabilities=False,
            low_memory=True,
            verbose=True,
        )
        topics, probabilities = model.fit_transform(texts, embeddings=embeddings)
        return model, topics, probabilities


class QuarterlyTopicAnalysis:
    def __init__(self, storage: Any, repo_root: Path):
        self.storage = storage
        self.repo_root = repo_root.resolve()

    def inspect(self, config, *, output_root: Path, snapshot_name: str):
        _name(snapshot_name)
        root = _safe_output_root(self.repo_root, output_root) / "snapshots"
        destination = root / snapshot_name
        staging = root / f"{snapshot_name}.building"
        if destination.exists() or staging.exists():
            raise ValueError(f"快照目录已存在: {destination} 或 {staging}")
        (staging / "quarters").mkdir(parents=True)
        counts, languages, sources, versions = Counter(), Counter(), Counter(), Counter()
        stats = Counter(selected_nonempty=0, too_short=0, truncated=0, usable_rows=0)
        handles, digests = {}, {}
        earliest = latest = None
        seen = set()
        with ExitStack() as stack:
            for batch in self.storage.iter_latest_nonempty_topic_documents():
                for row in batch:
                    stats["selected_nonempty"] += 1
                    corpus_id = row["corpus_id"]
                    if corpus_id in seen:
                        raise ValueError(f"季度快照含重复 corpus_id={corpus_id}")
                    seen.add(corpus_id)
                    text, truncated = prepare_text(row["clean_text"], config["max_chars"])
                    if len(text) < config["min_chars"]:
                        stats["too_short"] += 1
                        continue
                    stats["truncated"] += int(truncated)
                    stamp = row["created_at"]
                    if stamp.tzinfo is not None:
                        stamp = stamp.astimezone(UTC).replace(tzinfo=None)
                    quarter, _, _ = quarter_bounds(stamp)
                    if quarter not in handles:
                        handles[quarter] = stack.enter_context(
                            (staging / "quarters" / f"{quarter}.jsonl").open("wb")
                        )
                        digests[quarter] = hashlib.sha256()
                    document = {k: v for k, v in row.items() if k != "clean_text"}
                    document.update(text=text, quarter=quarter, created_at=stamp.isoformat())
                    document["annotation_updated_at"] = row["annotation_updated_at"].isoformat()
                    document["aspects"] = sorted({a["aspect"].strip() for a in row["annotations"]})
                    line = (
                        json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n"
                    ).encode("utf-8")
                    handles[quarter].write(line)
                    digests[quarter].update(line)
                    counts[quarter] += 1
                    languages[row["language"]] += 1
                    sources[row["source_type"]] += 1
                    versions[
                        (
                            row["model_name"],
                            row["prompt_version"],
                            row["taxonomy_version"],
                            row["cleaning_version"],
                        )
                    ] += 1
                    earliest = min(earliest, stamp) if earliest else stamp
                    latest = max(latest, stamp) if latest else stamp
                    stats["usable_rows"] += 1
                logger.info(
                    "季度 inspect：已选 %d 条非空标注，合格 %d 条",
                    stats["selected_nonempty"],
                    stats["usable_rows"],
                )
        if earliest is None:
            raise ValueError("没有满足文本长度要求的非空方面标注")
        quarters = {}
        cursor = quarter_bounds(earliest)[1]
        while cursor <= latest:
            key, start, end = quarter_bounds(cursor)
            quarters[key] = {
                "start_utc": start.isoformat(),
                "end_utc_exclusive": end.isoformat(),
                "document_count": counts[key],
                "file": f"quarters/{key}.jsonl" if counts[key] else None,
                "sha256": digests[key].hexdigest() if counts[key] else None,
                "source_boundary": start < earliest or end > latest,
            }
            cursor = end
        manifest = {
            "format": FORMAT,
            "selection_policy": POLICY,
            "snapshot_name": snapshot_name,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "topic_preprocessing_version": TOPIC_PREPROCESSING_VERSION,
            "text_config": {k: config[k] for k in ("max_chars", "min_chars")},
            "selection": dict(stats),
            "quarters": quarters,
            "source_created_at_min": earliest.isoformat(),
            "source_created_at_max": latest.isoformat(),
            "languages": dict(languages),
            "source_types": dict(sources),
            "versions": [
                dict(
                    model_name=m,
                    prompt_version=p,
                    taxonomy_version=t,
                    cleaning_version=c,
                    documents=n,
                )
                for (m, p, t, c), n in sorted(versions.items())
            ],
            "note": "所有标注版本；每 corpus_id 最新非空成功标注；不按方面/情感拆分；"
            "不限制语言或清洗版本；来源创建时间为 UTC；文本过短会排除。",
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        return {
            "snapshot_dir": str(destination),
            "snapshot_name": snapshot_name,
            "selection": dict(stats),
            "quarter_counts": dict(sorted(counts.items())),
        }

    def fit(
        self, config, *, output_root: Path, run_name: str, snapshot_name: str, resume: bool = False
    ):
        _name(run_name)
        _name(snapshot_name)
        root = _safe_output_root(self.repo_root, output_root)
        snapshot = root / "snapshots" / snapshot_name
        manifest_bytes = (snapshot / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if (
            manifest.get("format") != FORMAT
            or manifest.get("selection_policy") != POLICY
            or manifest.get("topic_preprocessing_version") != TOPIC_PREPROCESSING_VERSION
            or manifest.get("text_config") != {k: config[k] for k in ("max_chars", "min_chars")}
        ):
            raise ValueError("季度快照的版本或文本参数不一致")
        destination = root / "quarterly" / run_name
        if destination.exists() and not resume:
            raise ValueError(f"输出目录已存在，请使用 --resume 恢复: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        identity = {
            "format": FORMAT,
            "run_name": run_name,
            "snapshot_name": snapshot_name,
            "snapshot_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "config": config,
        }
        with _run_lock(destination / ".lock"):
            run_file = destination / "run.json"
            if run_file.exists() and json.loads(run_file.read_text(encoding="utf-8")) != identity:
                raise ValueError("恢复任务的配置或输入快照与原运行不一致")
            _atomic_json(run_file, identity)
            _atomic_json(destination / "status.json", {"status": "running", **identity})
            try:
                result = self._fit_quarters(config, snapshot, manifest, destination, identity)
            except BaseException as exc:
                _atomic_json(
                    destination / "status.json",
                    {
                        "status": "failed",
                        "error": str(exc),
                        **identity,
                    },
                )
                raise
            _atomic_json(
                root / "quarterly" / "latest.json",
                {
                    "run_name": run_name,
                    "path": run_name,
                    "manifest": f"{run_name}/manifest.json",
                },
            )
            return result

    def _fit_quarters(self, config, snapshot, manifest, destination, identity):
        import numpy as np

        backend = None
        quarter_stats = {}
        all_ids = set()
        for quarter, info in manifest["quarters"].items():
            documents = []
            if info["file"]:
                content = (snapshot / info["file"]).read_bytes()
                if hashlib.sha256(content).hexdigest() != info["sha256"]:
                    raise ValueError(f"季度快照校验失败: {quarter}")
                documents = [json.loads(line) for line in content.splitlines()]
            if len(documents) != info["document_count"]:
                raise ValueError(f"季度快照数量不一致: {quarter}")
            for row in documents:
                if row["corpus_id"] in all_ids:
                    raise ValueError("季度快照含重复 corpus_id")
                all_ids.add(row["corpus_id"])
                if quarter_bounds(datetime.fromisoformat(row["created_at"]))[0] != quarter:
                    raise ValueError("语料来源创建时间与季度不一致")
            folder = destination / "quarters" / quarter
            folder.mkdir(parents=True, exist_ok=True)
            checkpoint = folder / "manifest.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved["source_sha256"] != info["sha256"]:
                    raise ValueError(f"已完成季度的输入与快照不一致: {quarter}")
                for filename, digest in saved["artifact_sha256"].items():
                    if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != digest:
                        raise ValueError(f"已完成季度的产物损坏: {quarter}/{filename}")
                quarter_stats[quarter] = saved
                logger.info("季度 fit 复用已完成结果：%s，%d 条", quarter, len(documents))
                continue
            logger.info("季度 fit 开始：%s，%d 条", quarter, len(documents))
            required = max(6, config["min_documents"], 2 * config["min_topic_size"])
            model = None
            if len(documents) < required:
                topics, probabilities = [None] * len(documents), None
                status = "insufficient"
            else:
                if backend is None:
                    backend = _Backend(config, destination)
                ids = [row["corpus_id"] for row in documents]
                cache_ids = folder / "embedding_corpus_ids.json"
                cache_array = folder / "embeddings.npy"
                if cache_ids.exists() and cache_array.exists():
                    cached = json.loads(cache_ids.read_text(encoding="utf-8"))
                    if cached["corpus_ids"] != ids:
                        raise ValueError(f"向量缓存语料不一致: {quarter}")
                    if hashlib.sha256(cache_array.read_bytes()).hexdigest() != cached["sha256"]:
                        raise ValueError(f"向量缓存损坏: {quarter}")
                    embeddings = np.load(cache_array, allow_pickle=False)
                else:
                    embeddings = backend.encode([row["text"] for row in documents], config)
                    with (folder / "embeddings.tmp").open("wb") as handle:
                        np.save(handle, embeddings)
                    os.replace(folder / "embeddings.tmp", cache_array)
                    _atomic_json(
                        cache_ids,
                        {
                            "corpus_ids": ids,
                            "sha256": hashlib.sha256(cache_array.read_bytes()).hexdigest(),
                        },
                    )
                if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
                    raise ValueError(f"向量缓存维度错误: {quarter}")
                model, topics, probabilities = backend.cluster(
                    [row["text"] for row in documents],
                    embeddings,
                    config,
                )
                if len(topics) != len(documents):
                    raise ValueError("聚类分配数与语料数不一致")
                topics = [int(topic) for topic in topics]
                model.save(
                    str(folder / "model.pkl"), serialization="pickle", save_embedding_model=False
                )
                del embeddings
                status = "trained"
            topic_rows, assignments, review = self._quarter_results(
                quarter,
                documents,
                topics,
                probabilities,
                model,
                identity["run_name"],
            )
            _write_json(folder / "topics.json", {"quarter": quarter, "topics": topic_rows})
            self._write_assignments(folder / "assignments.jsonl", assignments)
            from .reporting import write_csv

            write_csv(
                folder / "review.csv",
                [
                    "quarter",
                    "topic_id",
                    "corpus_id",
                    "kind",
                    "repository",
                    "created_at",
                    "source_type",
                    "github_url",
                    "text",
                ],
                review,
            )
            files = ["topics.json", "assignments.jsonl", "review.csv"]
            if model is not None:
                files.extend(["model.pkl", "embeddings.npy", "embedding_corpus_ids.json"])
            result = {
                "quarter": quarter,
                **info,
                "status": status,
                "source_sha256": info["sha256"],
                "required_documents": required,
                "topics_excluding_outliers": sum(t["topic_id"] != -1 for t in topic_rows),
                "outlier_count": sum(t == -1 for t in topics),
                "outlier_share": (sum(t == -1 for t in topics) / len(topics))
                if status == "trained"
                else None,
                "artifact_sha256": {
                    f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in files
                },
            }
            _atomic_json(checkpoint, result)
            quarter_stats[quarter] = result
            _atomic_json(
                destination / "status.json",
                {
                    "status": "running",
                    "completed_quarters": len(quarter_stats),
                    "total_quarters": len(manifest["quarters"]),
                    "last_quarter": quarter,
                },
            )
            logger.info(
                "季度 fit 完成：%s，status=%s，topics=%d，outliers=%d",
                quarter,
                status,
                result["topics_excluding_outliers"],
                result["outlier_count"],
            )
            del model, assignments, review, documents
            gc.collect()
        if len(all_ids) != manifest["selection"]["usable_rows"]:
            raise ValueError("季度快照总语料数不一致")
        self._combine(destination, quarter_stats)
        final = {
            **identity,
            "status": "complete",
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "selection": manifest["selection"],
            "source": manifest,
            "quarters": quarter_stats,
            "encoder": "encoder/",
            "software": {
                name: version(name)
                for name in (
                    "bertopic",
                    "sentence-transformers",
                    "umap-learn",
                    "hdbscan",
                    "numpy",
                )
            },
            "note": "各季度独立聚类，topic_id 仅在本运行的本季度有效；"
            "insufficient 的 topic_id 为 null，训练离群项为 -1；保留各方面原始情感标注。",
        }
        _atomic_json(destination / "manifest.json", final)
        _atomic_json(
            destination / "status.json",
            {
                "status": "complete",
                "completed_quarters": len(quarter_stats),
                "documents": len(all_ids),
            },
        )
        return {
            "output_dir": str(destination),
            "documents": len(all_ids),
            "trained_quarters": sum(q["status"] == "trained" for q in quarter_stats.values()),
            "insufficient_quarters": sum(
                q["status"] == "insufficient" for q in quarter_stats.values()
            ),
        }

    @staticmethod
    def _quarter_results(quarter, documents, topics, probabilities, model, run_name):
        groups = defaultdict(list)
        assignments = []
        for offset, (row, topic) in enumerate(zip(documents, topics, strict=True)):
            topic = int(topic) if topic is not None else None
            probability = float(probabilities[offset]) if probabilities is not None else None
            assignments.append(
                {
                    **{k: v for k, v in row.items() if k != "text"},
                    "run_name": run_name,
                    "topic_id": topic,
                    "probability": probability,
                    "assignment_status": "insufficient"
                    if topic is None
                    else "outlier"
                    if topic == -1
                    else "assigned",
                }
            )
            if topic is not None:
                groups[topic].append(row["corpus_id"])
        docs = {
            r["corpus_id"]: TopicDocument(
                r["corpus_id"],
                r["source_type"],
                r["source_id"],
                r["parent_id"],
                r["content_hash"],
                r["repository"],
                datetime.fromisoformat(r["created_at"]),
                r["github_url"],
                r["text"],
                {},
            )
            for r in documents
        }
        topic_rows, review = [], []
        for topic, ids in sorted(groups.items()):
            terms = model.get_topic(topic) if topic != -1 else []
            keywords = [
                {"term": str(term), "weight": float(weight)} for term, weight in (terms or [])[:12]
            ]
            representatives, examples = _review_examples(
                model,
                topic,
                ids,
                docs,
                per_kind=5,
            )
            topic_rows.append(
                {
                    "run_name": run_name,
                    "quarter": quarter,
                    "topic_id": topic,
                    "name": " / ".join(k["term"] for k in keywords[:3])
                    if topic != -1
                    else "未归类",
                    "document_count": len(ids),
                    "quarter_total": len(documents),
                    "share_within_quarter": len(ids) / len(documents),
                    "keywords": keywords,
                    "representative_corpus_ids": representatives,
                }
            )
            review.extend(
                {"quarter": quarter, "topic_id": topic, **example} for example in examples
            )
        return topic_rows, assignments, review

    @staticmethod
    def _write_assignments(path, assignments):
        with path.open("w", encoding="utf-8") as handle:
            for row in assignments:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @staticmethod
    def _combine(destination, quarter_stats):
        topics = []
        columns = [
            "run_name",
            "quarter",
            "corpus_id",
            "annotation_id",
            "topic_id",
            "assignment_status",
            "probability",
            "created_at",
            "repository",
            "source_type",
            "source_id",
            "parent_id",
            "github_url",
            "model_name",
            "prompt_version",
            "taxonomy_version",
            "cleaning_version",
            "aspects",
            "annotations",
        ]
        with (
            (destination / "assignments.jsonl").open("w", encoding="utf-8") as combined,
            (destination / "assignments.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as csv_file,
        ):
            writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for quarter in quarter_stats:
                folder = destination / "quarters" / quarter
                topics.extend(
                    json.loads((folder / "topics.json").read_text(encoding="utf-8"))["topics"]
                )
                with (folder / "assignments.jsonl").open(encoding="utf-8") as source:
                    for line in source:
                        combined.write(line)
                        row = json.loads(line)
                        for key in ("aspects", "annotations"):
                            row[key] = json.dumps(row[key], ensure_ascii=False)
                        writer.writerow(row)
        _write_json(destination / "topics.json", {"topics": topics})
        from .reporting import write_csv

        fields = [
            "run_name",
            "quarter",
            "topic_id",
            "name",
            "document_count",
            "quarter_total",
            "share_within_quarter",
            "keywords",
            "representative_corpus_ids",
        ]
        write_csv(
            destination / "topics.csv",
            fields,
            [
                {
                    **row,
                    "keywords": json.dumps(row["keywords"], ensure_ascii=False),
                    "representative_corpus_ids": json.dumps(row["representative_corpus_ids"]),
                }
                for row in topics
            ],
        )
