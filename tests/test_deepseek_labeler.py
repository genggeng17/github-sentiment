import json

import httpx

from llm_labeler.service import DeepSeekClient, DeepSeekLabeler


def test_deepseek_client_requests_json_mode():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"annotations":[]}'}}]},
        )

    client = DeepSeekClient(
        "key",
        "https://api.deepseek.test",
        "deepseek-chat",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.complete("[TARGET]\nhello") == '{"annotations":[]}'
    finally:
        client.close()
    assert captured["response_format"] == {"type": "json_object"}
    prompt = captured["messages"][0]["content"]
    assert "CONTEXT 仅用于消歧" in prompt
    assert "只输出 TARGET 明确提及的标签" in prompt
    assert "未提及的标签不要输出" in prompt
    assert "runtime_performance：程序运行速度" in prompt


class FakeAnnotationStorage:
    def __init__(self):
        self.saved = []

    def iter_unannotated_corpus(self, *args):
        yield [{"id": 7, "model_input": "target"}]

    def save_annotation(self, row):
        self.saved.append(row)


class InvalidClient:
    model = "deepseek-chat"

    def complete(self, model_input):
        return '{"annotations":[{"aspect":"not_allowed","class":"positive"}]}'


def test_invalid_model_output_is_recorded_as_failed():
    storage = FakeAnnotationStorage()
    stats = DeepSeekLabeler(InvalidClient(), storage).label_pending()
    assert stats == {"read": 1, "succeeded": 0, "failed": 1}
    assert storage.saved[0]["status"] == "failed"
    assert storage.saved[0]["raw_response"].startswith("{")
    assert "未知 aspect" in storage.saved[0]["error_message"]
