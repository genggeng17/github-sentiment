from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from threading import Lock
from time import monotonic
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from storage import Storage

from .analytics import Filters, dashboard_payload, validate_filters

logger = logging.getLogger(__name__)


def encode_utc(value):
    return jsonable_encoder(
        value,
        custom_encoder={
            datetime: lambda item: item.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")
        },
    )


class ResponseCache:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.entries = OrderedDict()
        self.lock = Lock()

    def get(self, key, build):
        if not self.seconds:
            return build()
        with self.lock:
            now = monotonic()
            cached = self.entries.get(key)
            if cached and cached[0] > now:
                self.entries.move_to_end(key)
                return cached[1]
            value = build()
            self.entries[key] = (monotonic() + self.seconds, value)
            self.entries.move_to_end(key)
            while len(self.entries) > 128:
                self.entries.popitem(last=False)
            return value


def query_filters(
    start_date: str,
    end_date: str,
    dimension: str = "overall",
    sentiment: Literal["all", "positive", "neutral", "negative"] = "all",
    repository_id: Annotated[int | None, Query(gt=0)] = None,
    granularity: Literal["month", "quarter", "year"] = "month",
) -> Filters:
    try:
        return validate_filters(
            start_date, end_date, dimension, sentiment, repository_id, granularity
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_app(storage: Storage | None = None, *, cache_seconds: int | None = None) -> FastAPI:
    load_dotenv()
    owns_storage = storage is None
    if storage is None:
        database_url = os.getenv("API_DATABASE_URL") or os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("HTTP API 需要配置 DATABASE_URL 或 API_DATABASE_URL")
        storage = Storage(database_url)
    if cache_seconds is None:
        cache_seconds = int(os.getenv("API_CACHE_SECONDS", "30"))
    if cache_seconds < 0:
        raise ValueError("API_CACHE_SECONDS 不能小于 0")
    cache = ResponseCache(cache_seconds)

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if owns_storage:
            storage.engine.dispose()

    app = FastAPI(title="Developer Voice Query API", lifespan=lifespan)
    origins = [
        item.strip() for item in os.getenv("API_CORS_ORIGINS", "").split(",") if item.strip()
    ]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET"],
            allow_headers=["Accept", "Content-Type"],
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_request: Request, exc: SQLAlchemyError):
        logger.error("Query API database request failed: %s", type(exc).__name__)
        return JSONResponse(
            status_code=503,
            content={
                "detail": "查询数据库暂不可用，请检查连接并执行 init-db 和 refresh-sentiment-facts"
            },
        )

    def dashboard(filters):
        key = ("dashboard", *asdict(filters).values())

        def build():
            result = storage.sentiment_statistics(
                *filters.bounds,
                aspect=None if filters.dimension == "overall" else filters.dimension,
                repository_id=filters.repository_id,
            )
            return encode_utc(dashboard_payload(filters, result))

        return cache.get(key, build)

    @app.get("/api/v1/health")
    def health():
        return cache.get(
            ("health",),
            lambda: encode_utc(
                {
                    "status": "ok",
                    "source": "sentiment_facts",
                    "timezone": "UTC",
                    "data_range": storage.sentiment_fact_metadata(),
                    "topics_source": "mock",
                }
            ),
        )

    @app.get("/api/v1/repositories")
    def repositories():
        return {"source": "sentiment_facts", "repositories": storage.sentiment_repositories()}

    @app.get("/api/v1/analytics/dashboard")
    def get_dashboard(filters: Annotated[Filters, Depends(query_filters)]):
        return dashboard(filters)

    @app.get("/api/v1/analytics/trend")
    def trend(filters: Annotated[Filters, Depends(query_filters)]):
        data = dashboard(filters)
        series = []
        for row in data["analytics"]["trend"]:
            counts = row if filters.dimension == "overall" else row["categories"][filters.dimension]
            series.append(
                {
                    "period": row["month"],
                    "month": row["month"],
                    **{key: counts[key] for key in ("total", "positive", "neutral", "negative")},
                    "sentiment_score": (counts["positive"] - counts["negative"]) / counts["total"]
                    if counts["total"]
                    else 0,
                }
            )
        return {
            "source": "sentiment_facts",
            "dimension": filters.dimension,
            "granularity": "month",
            "series": series,
        }

    @app.get("/api/v1/analytics/dimensions")
    def dimensions(filters: Annotated[Filters, Depends(query_filters)]):
        return {
            "source": "sentiment_facts",
            "categories": dashboard(filters)["analytics"]["categories"],
        }

    @app.get("/api/v1/sentiment-facts")
    def facts(
        filters: Annotated[Filters, Depends(query_filters)],
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        after_corpus_id: Annotated[int, Query(ge=0)] = 0,
        after_aspect: Annotated[str, Query(max_length=40)] = "",
    ):
        rows = storage.query_sentiment_facts(
            *filters.bounds,
            aspect=None if filters.dimension == "overall" else filters.dimension,
            sentiment=None if filters.sentiment == "all" else filters.sentiment,
            repository_id=filters.repository_id,
            after_corpus_id=after_corpus_id,
            after_aspect=after_aspect,
            limit=limit + 1,
        )
        has_more = len(rows) > limit
        items = rows[:limit]
        next_cursor = (
            {"after_corpus_id": items[-1]["corpus_id"], "after_aspect": items[-1]["aspect"]}
            if has_more
            else None
        )
        return encode_utc(
            {"source": "sentiment_facts", "items": items, "next_cursor": next_cursor},
        )

    return app
