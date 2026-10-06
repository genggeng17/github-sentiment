from datetime import datetime

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.pool import StaticPool

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.app import create_app
from corpus import CorpusBuilder
from storage import Storage
from storage.models import Corpus, Issue, PipelineRun, SentimentFact


def populate_fact_table(storage):
    first_repo = storage.ensure_repository("rust-lang/rust")
    second_repo = storage.ensure_repository("tokio-rs/tokio")
    for github_id, repository_id, created_at in (
        (1, first_repo, datetime(2025, 1, 1)),
        (2, first_repo, datetime(2025, 1, 31, 23, 59, 59)),
        (3, first_repo, datetime(2025, 3, 1)),
        (4, second_repo, datetime(2025, 2, 1)),
    ):
        storage.upsert_issues(
            repository_id,
            [
                {
                    "github_id": github_id,
                    "number": github_id,
                    "title": f"Compiler discussion {github_id}",
                    "body": f"Unique content {github_id}",
                    "state": "open",
                    "github_url": f"https://github.test/{github_id}",
                    "created_at": created_at,
                    "updated_at": created_at,
                }
            ],
        )
    CorpusBuilder(storage).build()
    with storage.sessions.begin() as session:
        sources = {
            github_id: (corpus_id, repo, created_at)
            for github_id, corpus_id, repo, created_at in session.execute(
                select(Issue.github_id, Corpus.id, Issue.repository_id, Issue.created_at).join(
                    Corpus, Corpus.source_id == Issue.id
                )
            )
        }
        for github_id, aspect, sentiment in (
            (1, "ownership", "positive"),
            (1, "compile_time", "negative"),
            (2, "ownership", "neutral"),
            (3, "compile_time", "positive"),
            (4, "ownership", "negative"),
        ):
            corpus_id, repo, created_at = sources[github_id]
            session.add(
                SentimentFact(
                    corpus_id=corpus_id,
                    repository_id=repo,
                    created_at=created_at,
                    aspect=aspect,
                    sentiment=sentiment,
                )
            )
        session.add(
            PipelineRun(
                id="fixture-refresh",
                run_type="refresh-sentiment-facts",
                status="succeeded",
                completed_at=datetime(2025, 4, 1),
            )
        )
    return first_repo, second_repo


@pytest.fixture
def api_storage():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    storage = Storage("", engine=engine)
    storage.create_schema()
    populate_fact_table(storage)
    yield storage
    engine.dispose()


@pytest.fixture
def client(api_storage):
    with TestClient(create_app(api_storage, cache_seconds=0)) as value:
        yield value


PARAMS = {"start_date": "2025-01", "end_date": "2025-02"}


def test_dashboard_deduplicates_corpus_and_aggregates_aspect_mentions(client):
    response = client.get("/api/v1/analytics/dashboard", params=PARAMS)
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "sentiment_facts"
    assert data["summary"] == {
        "total_count": 3,
        "valid_count": 3,
        "aspect_count": 4,
        "positive_count": 1,
        "negative_count": 2,
    }
    assert data["updated_at"] == "2025-04-01T00:00:00Z"
    assert len(data["analytics"]["categories"]) == 13
    january, february = data["analytics"]["trend"]
    assert january["month"] == "2025-01"
    assert (january["total"], january["positive"], january["neutral"], january["negative"]) == (
        3,
        1,
        1,
        1,
    )
    assert february["total"] == 1
    for row in data["analytics"]["categories"]:
        assert row["total"] == row["positive"] + row["neutral"] + row["negative"]
        assert "keywords" not in row


def test_dimension_filters_cards_but_preserves_full_aspect_profile(client):
    data = client.get(
        "/api/v1/analytics/dashboard", params={**PARAMS, "dimension": "compile_time"}
    ).json()
    assert data["summary"]["total_count"] == 1
    assert data["summary"]["positive_count"] == 0
    assert data["summary"]["negative_count"] == 1
    assert (
        next(row for row in data["analytics"]["categories"] if row["id"] == "ownership")["total"]
        == 3
    )
    # The legacy sentiment parameter does not remove classes from a score denominator.
    assert (
        client.get(
            "/api/v1/analytics/dashboard", params={**PARAMS, "sentiment": "negative"}
        ).json()["summary"]["positive_count"]
        == 1
    )


def test_month_bounds_include_end_month_and_zero_fill_empty_months(client):
    data = client.get(
        "/api/v1/analytics/dashboard", params={"start_date": "2025-02", "end_date": "2025-01"}
    ).json()
    assert data["filters"]["start_date"] == "2025-01"
    assert data["summary"]["total_count"] == 3
    empty = client.get(
        "/api/v1/analytics/dashboard", params={"start_date": "2025-04", "end_date": "2025-05"}
    ).json()
    assert empty["summary"]["total_count"] == 0
    assert [row["month"] for row in empty["analytics"]["trend"]] == ["2025-04", "2025-05"]
    assert all(row["total"] == 0 for row in empty["analytics"]["trend"])


def test_repository_and_trend_endpoints(client):
    repos = client.get("/api/v1/repositories").json()["repositories"]
    repository_id = next(
        row["repository_id"] for row in repos if row["full_name"] == "tokio-rs/tokio"
    )
    data = client.get(
        "/api/v1/analytics/dashboard", params={**PARAMS, "repository_id": repository_id}
    ).json()
    assert data["summary"]["total_count"] == 1
    assert data["summary"]["negative_count"] == 1
    trend = client.get(
        "/api/v1/analytics/trend",
        params={**PARAMS, "dimension": "compile_time", "granularity": "quarter"},
    ).json()
    assert trend["granularity"] == "month"
    assert [row["sentiment_score"] for row in trend["series"]] == [-1, 0]
    dimensions = client.get("/api/v1/analytics/dimensions", params=PARAMS).json()
    assert len(dimensions["categories"]) == 13


def test_fact_cursor_keeps_multiple_aspects_and_utc_time(client):
    cursor = {}
    items = []
    while True:
        data = client.get("/api/v1/sentiment-facts", params={**PARAMS, "limit": 1, **cursor}).json()
        items.extend(data["items"])
        cursor = data["next_cursor"]
        if not cursor:
            break
    assert len(items) == 4
    assert len({(row["corpus_id"], row["aspect"]) for row in items}) == 4
    assert all(
        set(row) == {"corpus_id", "repository_id", "created_at", "aspect", "sentiment"}
        for row in items
    )
    assert items[0]["created_at"] == "2025-01-01T00:00:00Z"
    negative = client.get(
        "/api/v1/sentiment-facts", params={**PARAMS, "sentiment": "negative"}
    ).json()
    assert len(negative["items"]) == 2


@pytest.mark.parametrize(
    "params",
    [
        {"start_date": "2025-13"},
        {"start_date": "2025-1"},
        {"end_date": "9999-12"},
        {"end_date": "2099-01"},
        {"dimension": "invalid"},
        {"sentiment": "0"},
        {"repository_id": "0"},
        {"limit": "1001"},
        {"after_corpus_id": "-1"},
    ],
)
def test_invalid_queries_are_rejected(client, params):
    assert client.get("/api/v1/sentiment-facts", params={**PARAMS, **params}).status_code == 422


def test_health_and_requests_do_not_modify_facts(client, api_storage):
    metadata = client.get("/api/v1/health").json()["data_range"]
    assert metadata["aspect_count"] == 5
    assert metadata["first_created_at"] == "2025-01-01T00:00:00Z"
    assert client.post("/api/v1/analytics/dashboard", json={}).status_code == 405
    with api_storage.sessions() as session:
        assert session.scalar(select(func.count()).select_from(SentimentFact)) == 5


def test_missing_schema_is_reported_without_creating_tables():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    with TestClient(create_app(Storage("", engine=engine), cache_seconds=0)) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 503
        assert "init-db" in response.json()["detail"]
    engine.dispose()


def test_explicit_cors_origin(api_storage, monkeypatch):
    monkeypatch.setenv("API_CORS_ORIGINS", "https://frontend.example")
    with TestClient(create_app(api_storage, cache_seconds=0)) as client:
        response = client.options(
            "/api/v1/health",
            headers={"Origin": "https://frontend.example", "Access-Control-Request-Method": "GET"},
        )
        assert response.headers["access-control-allow-origin"] == "https://frontend.example"
