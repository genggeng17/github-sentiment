from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from taxonomy import ASPECTS

ASPECT_NAMES = {
    "package_manager": "包管理",
    "api_extensibility": "API 与扩展性",
    "tooling_documentation": "工具与文档",
    "diagnostics_debugging": "诊断与调试",
    "runtime_performance": "运行性能",
    "compile_time": "编译时间",
    "safety": "安全性",
    "readability_maintainability": "可读性与可维护性",
    "ownership": "所有权",
    "libraries_frameworks": "库与框架",
    "type_system": "类型系统",
    "learning_curve": "学习曲线",
    "community": "社区",
}


@dataclass(frozen=True)
class Filters:
    start_date: str
    end_date: str
    dimension: str = "overall"
    sentiment: str = "all"
    repository_id: int | None = None
    granularity: str = "month"

    @property
    def bounds(self) -> tuple[datetime, datetime]:
        start = datetime.strptime(self.start_date, "%Y-%m")
        end = datetime.strptime(self.end_date, "%Y-%m")
        return start, next_month(end)


def next_month(value: datetime) -> datetime:
    return datetime(value.year + (value.month == 12), value.month % 12 + 1, 1)


def validate_filters(start_date, end_date, dimension, sentiment, repository_id, granularity):
    try:
        start = datetime.strptime(start_date, "%Y-%m")
        end = datetime.strptime(end_date, "%Y-%m")
    except ValueError as exc:
        raise ValueError("日期必须是有效的 YYYY-MM") from exc
    if start.strftime("%Y-%m") != start_date or end.strftime("%Y-%m") != end_date:
        raise ValueError("日期必须使用 YYYY-MM 格式")
    start, end = sorted((start, end))
    if end.year >= 9999 or (end.year - start.year) * 12 + end.month - start.month >= 600:
        raise ValueError("时间范围最多 600 个月，年份须小于 9999")
    if dimension != "overall" and dimension not in ASPECTS:
        raise ValueError("未知 aspect/dimension")
    return Filters(
        start.strftime("%Y-%m"),
        end.strftime("%Y-%m"),
        dimension,
        sentiment,
        repository_id,
        granularity,
    )


def empty_counts():
    return {"total": 0, "positive": 0, "neutral": 0, "negative": 0}


def dashboard_payload(filters: Filters, result: dict) -> dict:
    start, stop = filters.bounds
    months = {}
    while start < stop:
        month = start.strftime("%Y-%m")
        months[month] = {
            "month": month,
            **empty_counts(),
            "categories": {aspect: empty_counts() for aspect in ASPECT_NAMES},
        }
        start = next_month(start)
    categories = {
        aspect: {"id": aspect, "name": name, **empty_counts()}
        for aspect, name in ASPECT_NAMES.items()
    }
    for row in result["counts"]:
        month = f"{int(row['year']):04d}-{int(row['month']):02d}"
        # Preserve historical aspect names if they exist in the query table.
        if row["aspect"] not in categories:
            categories[row["aspect"]] = {
                "id": row["aspect"],
                "name": row["aspect"],
                **empty_counts(),
            }
        period = months[month]
        bucket = period["categories"].setdefault(row["aspect"], empty_counts())
        for counts in (period, bucket, categories[row["aspect"]]):
            counts[row["sentiment"]] += row["count"]
            counts["total"] += row["count"]
    return {
        "source": "sentiment_facts",
        "timezone": "UTC",
        "filters": {
            "start_date": filters.start_date,
            "end_date": filters.end_date,
            "dimension": filters.dimension,
            "sentiment": filters.sentiment,
            "repository_id": filters.repository_id,
            "granularity": "month",
        },
        "summary": {**result["summary"], "valid_count": result["summary"]["total_count"]},
        "updated_at": result["updated_at"],
        "analytics": {"trend": list(months.values()), "categories": list(categories.values())},
    }
