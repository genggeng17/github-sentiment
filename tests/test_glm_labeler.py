import asyncio
import json
import os

import httpx
import pytest

import config
import pipeline
from config import Settings
from llm_labeler.service import LLMClient


@pytest.mark.parametrize("as_json", [True, False])
def test_api_error_diagnostics(caplog, as_json):
    async def run():
        def handler(request):
            kwargs = ({"json": {"error": {"code": "1302", "message": "limit secret-key"}}}
                      if as_json else {"text": "private response body"})
            return httpx.Response(429, headers={"Retry-After": "20"}, **kwargs)

        client = LLMClient("secret-key", "https://example.com", "test", max_retries=0,
                           transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(RuntimeError, match="重试耗尽"):
                await client.complete({"id": 1, "model_input": "private corpus"})
        finally:
            await client.close()

    asyncio.run(run())
    records = [r.message for r in caplog.records if "[LLM_API_ERROR]" in r.message]
    assert len(records) == 1
    data = json.loads(records[0].split("[LLM_API_ERROR] ", 1)[1])
    assert data["http_status"] == 429
    assert data["retry_after"] == "20"
    assert data["error_code"] == ("1302" if as_json else None)
    assert "secret-key" not in records[0]
    assert "private" not in records[0]


@pytest.fixture
def clean_environment(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for name in os.environ:
        if name.startswith(("GLM_", "DEEPSEEK_", "LLM_")):
            monkeypatch.delenv(name)
    return monkeypatch


def test_default_glm_never_uses_legacy_deepseek_credentials(clean_environment):
    clean_environment.setenv("DEEPSEEK_API_KEY", "legacy-key")
    clean_environment.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    settings = Settings.from_env()
    assert settings.llm_provider == "glm"
    assert settings.labeling_model == "glm-5.3-flash"
    assert settings.labeling_base_url == "https://open.bigmodel.cn/api/paas/v4"
    with pytest.raises(ValueError, match="GLM_API_KEY"):
        settings.require_labeling()


def test_explicit_deepseek_switch_still_works(clean_environment):
    clean_environment.setenv("LLM_PROVIDER", "deepseek")
    clean_environment.setenv("DEEPSEEK_API_KEY", "legacy-key")
    settings = Settings.from_env()
    settings.require_labeling()
    assert settings.labeling_api_key == "legacy-key"
    assert settings.labeling_model == "deepseek-v4-flash"
    assert settings.labeling_base_url == "https://api.deepseek.com"


@pytest.mark.parametrize(
    ("name", "value"),
    [("LLM_PROVIDER", "unknown"), ("GLM_REASONING_EFFORT", "none"),
     ("GLM_MAX_TOKENS", "0"), ("GLM_MAX_TOKENS", "131073"),
     ("GLM_TIMEOUT_SECONDS", "0")],
)
def test_rejects_invalid_glm_configuration(clean_environment, name, value):
    clean_environment.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        Settings.from_env()


@pytest.mark.parametrize("finish_reason", ["stop", "length", "sensitive", "network_error"])
def test_glm_pipeline_routes_request_and_saves_only_complete_content(monkeypatch, finish_reason):
    captured = {}

    async def handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"finish_reason": finish_reason, "message": {
                "content": '{"annotations":[]}',
                "reasoning_content": "Reasoning must not be parsed as annotation JSON.",
            }}],
            "usage": {"prompt_tokens": 200, "prompt_tokens_details": {"cached_tokens": 150}},
        })

    def client_factory(*args, **kwargs):
        captured["timeout"] = kwargs["timeout_seconds"]
        return LLMClient(*args, **kwargs, transport=httpx.MockTransport(handler))

    class Storage:
        saved = []

        def completed_sample_set_id(self, name):
            assert name == "all-aspects"
            return 9

        def iter_unannotated_corpus(self, taxonomy, prompt, model, batch_size, **kwargs):
            assert model == "glm-5.3-flash"
            assert kwargs["sample_set_id"] == 9
            yield [{"id": 7, "model_input": "[TARGET]\nhello"}]

        def save_annotations(self, rows):
            self.saved.extend(rows)

    monkeypatch.setattr(pipeline, "LLMClient", client_factory)
    settings = Settings(
        database_url="sqlite://", github_token="", repositories=(),
        glm_api_key="glm-key", deepseek_api_key="unused-key",
        glm_reasoning_effort="high", glm_max_tokens=4096, glm_timeout_seconds=120,
    )
    storage = Storage()
    result = pipeline.Pipeline(settings, storage).label(sample_name="all-aspects", limit=1)
    assert captured["url"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert captured["authorization"] == "Bearer glm-key"
    assert captured["timeout"] == 120
    payload = captured["payload"]
    assert payload["model"] == "glm-5.3-flash"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["max_tokens"] == 4096
    assert payload["temperature"] == 1
    assert payload["response_format"] == {"type": "json_object"}
    assert "user_id" not in payload
    assert storage.saved[0]["model_name"] == "glm-5.3-flash"
    if finish_reason == "stop":
        assert result["succeeded"] == 1
        assert result["cache_hit_tokens"] == 150
        assert result["cache_miss_tokens"] == 50
        assert storage.saved[0]["parsed_result"] == {"annotations": []}
    else:
        assert result["failed"] == 1
        assert storage.saved[0]["parsed_result"] is None


def test_glm_defaults_reserve_reasoning_tokens():
    async def run():
        client = LLMClient("key", "https://example.test/v4", "glm-5.3-flash", provider="glm")
        try:
            assert client.max_tokens == 8192
            assert client.reasoning_effort == "low"
            assert client.user_id == ""
        finally:
            await client.close()

    asyncio.run(run())
