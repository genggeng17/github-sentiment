"""Build inspectable topic and sentiment summaries from document assignments."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rollups(assignments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Count documents; denominators keep monthly and repository changes comparable."""
    aspect_total: Counter[str] = Counter()
    topic_total: Counter[tuple[str, int]] = Counter()
    month_total: Counter[tuple[str, str]] = Counter()
    repository_total: Counter[tuple[str, str]] = Counter()
    month_repository_total: Counter[tuple[str, str, str]] = Counter()
    sentiment: Counter[tuple[str, int, str]] = Counter()
    monthly: Counter[tuple[str, str, int, str]] = Counter()
    repository: Counter[tuple[str, str, int, str]] = Counter()
    monthly_repository: Counter[tuple[str, str, str, int, str]] = Counter()

    for row in assignments:
        aspect = row["aspect"]
        topic = int(row["topic_id"])
        feeling = row["sentiment"]
        month = row["month"]
        repo = row["repository"]
        aspect_total[aspect] += 1
        topic_total[(aspect, topic)] += 1
        month_total[(aspect, month)] += 1
        repository_total[(aspect, repo)] += 1
        month_repository_total[(aspect, month, repo)] += 1
        sentiment[(aspect, topic, feeling)] += 1
        monthly[(aspect, month, topic, feeling)] += 1
        repository[(aspect, repo, topic, feeling)] += 1
        monthly_repository[(aspect, month, repo, topic, feeling)] += 1

    topics = [
        {
            "aspect": aspect,
            "topic_id": topic,
            "document_count": count,
            "aspect_total": aspect_total[aspect],
            "share_within_aspect": round(count / aspect_total[aspect], 6),
        }
        for (aspect, topic), count in sorted(topic_total.items())
    ]
    sentiments = [
        {
            "aspect": aspect,
            "topic_id": topic,
            "sentiment": feeling,
            "document_count": count,
            "topic_total": topic_total[(aspect, topic)],
            "share_within_topic": round(count / topic_total[(aspect, topic)], 6),
        }
        for (aspect, topic, feeling), count in sorted(sentiment.items())
    ]
    time_rows = [
        {
            "aspect": aspect,
            "month": month,
            "topic_id": topic,
            "sentiment": feeling,
            "document_count": count,
            "aspect_month_total": month_total[(aspect, month)],
            "share_within_aspect_month": round(count / month_total[(aspect, month)], 6),
        }
        for (aspect, month, topic, feeling), count in sorted(monthly.items())
    ]
    repo_rows = [
        {
            "aspect": aspect,
            "repository": repo,
            "topic_id": topic,
            "sentiment": feeling,
            "document_count": count,
            "aspect_repository_total": repository_total[(aspect, repo)],
            "share_within_aspect_repository": round(count / repository_total[(aspect, repo)], 6),
        }
        for (aspect, repo, topic, feeling), count in sorted(repository.items())
    ]
    month_repo_rows = [
        {
            "aspect": aspect,
            "month": month,
            "repository": repo,
            "topic_id": topic,
            "sentiment": feeling,
            "document_count": count,
            "aspect_month_repository_total": month_repository_total[(aspect, month, repo)],
            "share_within_aspect_month_repository": round(
                count / month_repository_total[(aspect, month, repo)], 6
            ),
        }
        for (aspect, month, repo, topic, feeling), count in sorted(monthly_repository.items())
    ]
    return {
        "topics": topics,
        "sentiment": sentiments,
        "time": time_rows,
        "repository": repo_rows,
        "time_repository": month_repo_rows,
    }
