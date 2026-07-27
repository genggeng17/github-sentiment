import json

import httpx
import pytest

from llm_labeler.service import DeepSeekClient, DeepSeekLabeler
from pipeline import build_parser


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


class ValidClient:
    model = "deepseek-chat"

    def complete(self, model_input):
        return '{"annotations":[]}'


class MultipleCorpusStorage(FakeAnnotationStorage):
    def __init__(self, count):
        super().__init__()
        self.count = count
        self.requested_batch_size = None

    def iter_unannotated_corpus(
        self, taxonomy_version, prompt_version, model_name, batch_size
    ):
        self.requested_batch_size = batch_size
        yield [{"id": index, "model_input": f"target-{index}"} for index in range(self.count)]


def test_label_pending_stops_at_limit():
    storage = MultipleCorpusStorage(5)
    stats = DeepSeekLabeler(ValidClient(), storage, batch_size=20).label_pending(limit=2)
    assert stats == {"read": 2, "succeeded": 2, "failed": 0}
    assert storage.requested_batch_size == 2
    assert [row["corpus_id"] for row in storage.saved] == [0, 1]


def test_label_pending_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="必须大于 0"):
        DeepSeekLabeler(ValidClient(), FakeAnnotationStorage()).label_pending(limit=0)


def test_label_command_accepts_positive_limit():
    args = build_parser().parse_args(["label", "--limit", "30000"])
    assert args.command == "label"
    assert args.limit == 30000


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_label_command_rejects_invalid_limit(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["label", "--limit", value])
