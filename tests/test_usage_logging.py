import asyncio
import json
import logging

import httpx
import pytest

from config import Settings
from llm_labeler.service import LLMClient, LLMFatalError, LLMLabeler
from llm_labeler.usage import TokenUsage


class MemoryStorage:
    def __init__(self, count=1):
        self.count = count
        self.saved = []

    def iter_unannotated_corpus(self, *args, **kwargs):
        yield [{"id": i, "model_input": "[TARGET]\nhello"} for i in range(self.count)]

    def save_annotations(self, rows):
        self.saved.extend(rows)


def usage_logs(caplog):
    return [json.loads(r.message.split("[LLM_USAGE] ", 1)[1])
            for r in caplog.records if "[LLM_USAGE] " in r.message]


def response(finish="stop", content='{"annotations":[]}'):
    return httpx.Response(200, json={
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {
            "prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 150},
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    })


def test_usage_legacy_cache_and_unknown_fields():
    usage = TokenUsage()
    usage.record({"prompt_cache_hit_tokens": 10, "prompt_cache_miss_tokens": 5})
    usage.record(None)
    usage.record({"completion_tokens": -1})
    totals = usage.snapshot()
    assert totals["input_tokens"] == totals["total_tokens"] == 15
    assert totals["cache_hit_tokens"] == 10
    assert totals["usage_responses"] == 1
    assert totals["missing_usage_responses"] == 2
    assert totals["output_usage_responses"] == totals["reasoning_usage_responses"] == 0


@pytest.mark.parametrize(("finish", "content", "succeeded"), [
    ("stop", '{"annotations":[]}', 1), ("length", "truncated", 0),
    ("stop", "not JSON", 0),
])
def test_final_usage_includes_failed_annotations(caplog, finish, content, succeeded):
    caplog.set_level(logging.INFO)

    async def run():
        client = LLMClient("key", "https://example.test/v4", "glm-5.3-flash", provider="glm",
                           transport=httpx.MockTransport(lambda r: response(finish, content)))
        try:
            return await LLMLabeler(client, MemoryStorage()).label_pending_async(
                run_id="run-123", sample_set_id=42,
            )
        finally:
            await client.close()

    stats = asyncio.run(run())
    assert stats["succeeded"] == succeeded
    assert stats["input_tokens"] == 200
    assert stats["output_tokens"] == 100
    assert stats["reasoning_tokens"] == 60
    assert stats["total_tokens"] == 300  # reasoning is already part of output
    assert stats["cache_hit_tokens"] == 150
    logs = usage_logs(caplog)
    assert [r["event"] for r in logs] == ["start", "final"]
    assert logs[-1]["run_id"] == "run-123"
    assert logs[-1]["sample_set_id"] == 42
    assert logs[-1]["completed"] == 1
    assert logs[-1]["http_attempts"] == 1
    assert logs[-1]["total_tokens"] == 300


def test_periodic_log_runs_while_request_pending_and_stops_on_completion(caplog):
    caplog.set_level(logging.INFO)

    async def run():
        observed = asyncio.Event()
        calls = 0

        class Observer(logging.Handler):
            def emit(self, record):
                if '"event":"progress"' in record.getMessage():
                    observed.set()

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 2:
                await asyncio.wait_for(observed.wait(), timeout=2)
            return response()

        logger = logging.getLogger("llm_labeler.service")
        observer = Observer()
        logger.addHandler(observer)
        client = LLMClient("key", "https://example.test", "test-model",
                           transport=httpx.MockTransport(handler))
        try:
            await LLMLabeler(client, MemoryStorage(2), usage_log_interval_seconds=0.01
                             ).label_pending_async()
            count = len(usage_logs(caplog))
            await asyncio.sleep(0.04)
            assert len(usage_logs(caplog)) == count
        finally:
            logger.removeHandler(observer)
            await client.close()

    asyncio.run(run())
    progress = next(r for r in usage_logs(caplog) if r["event"] == "progress")
    assert progress["in_flight"] == 1
    assert progress["completed"] == 1
    assert progress["output_tokens"] == 100
    assert progress["total_tokens"] == 300
    assert usage_logs(caplog)[-1]["total_tokens"] == 600
    assert usage_logs(caplog)[-1]["cache_hit_tokens"] == 300


def test_missing_usage_is_marked_unknown_not_complete_zero(caplog):
    caplog.set_level(logging.INFO)

    async def run():
        client = LLMClient("key", "https://example.test", "test-model",
                           transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
                               "choices": [{"message": {"content": '{"annotations":[]}'}}],
                           })))
        try:
            await LLMLabeler(client, MemoryStorage()).label_pending_async()
        finally:
            await client.close()

    asyncio.run(run())
    final = usage_logs(caplog)[-1]
    assert final["missing_usage_responses"] == 1
    assert final["output_tokens"] is None
    assert final["reasoning_tokens"] is None


def test_final_log_on_fatal_error_keeps_previous_usage(caplog):
    caplog.set_level(logging.INFO)
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return response() if calls == 1 else httpx.Response(401)

    async def run():
        client = LLMClient("key", "https://example.test", "test-model",
                           transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(LLMFatalError):
                await LLMLabeler(client, MemoryStorage(2)).label_pending_async()
        finally:
            await client.close()

    asyncio.run(run())
    last = usage_logs(caplog)[-1]
    assert last["event"] == "final" and last["status"] == "failed"
    assert last["total_tokens"] == 300
    assert last["http_attempts"] == 2
    assert last["usage_responses"] == 1


def test_default_interval_is_thirty_minutes():
    assert Settings("sqlite://", "", ()).llm_usage_log_interval_seconds == 1800
    assert LLMLabeler(object(), MemoryStorage()).usage_log_interval_seconds == 1800
