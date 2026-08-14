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
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": (
                                '{"results":[{"corpus_id":7,"annotations":[]}]}'
                            )
                        },
                    }
                ]
            },
        )

    client = DeepSeekClient(
        "key",
        "https://api.deepseek.test",
        "deepseek-chat",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert (
            client.complete_batch([{"id": 7, "model_input": "[TARGET]\nhello"}])
            == '{"results":[{"corpus_id":7,"annotations":[]}]}'
        )
    finally:
        client.close()
    assert captured["response_format"] == {"type": "json_object"}
    prompt = captured["messages"][0]["content"]
    assert "CONTEXT 可用于确定 TARGET 所指的对象" in prompt
    assert "输出所有且仅输出 TARGET 明确讨论的方面" in prompt
    assert "不得仅根据工具名" in prompt
    assert "runtime_performance：讨论程序运行阶段" in prompt
    assert "results 必须与输入 items 一一对应" in prompt
    user_payload = json.loads(captured["messages"][1]["content"])
    assert user_payload == {
        "items": [{"corpus_id": 7, "model_input": "[TARGET]\nhello"}]
    }


class FakeAnnotationStorage:
    def __init__(self):
        self.saved = []

    def iter_unannotated_corpus(self, *args, **kwargs):
        yield [{"id": 7, "model_input": "target"}]

    def save_annotation(self, row):
        self.saved.append(row)


class InvalidClient:
    model = "deepseek-chat"

    def complete_batch(self, corpus):
        return (
            '{"results":[{"corpus_id":7,"annotations":'
            '[{"aspect":"not_allowed","class":"positive"}]}]}'
        )


def test_invalid_model_output_is_recorded_as_failed():
    storage = FakeAnnotationStorage()
    stats = DeepSeekLabeler(InvalidClient(), storage).label_pending()
    assert stats == {"read": 1, "succeeded": 0, "failed": 1}
    assert storage.saved[0]["status"] == "failed"
    assert storage.saved[0]["raw_response"].startswith("{")
    assert "未知 aspect" in storage.saved[0]["error_message"]


class ValidClient:
    model = "deepseek-chat"

    def __init__(self):
        self.calls = []

    def complete_batch(self, corpus):
        self.calls.append(corpus)
        return json.dumps(
            {
                "results": [
                    {"corpus_id": item["id"], "annotations": []}
                    for item in corpus
                ]
            }
        )


class MultipleCorpusStorage(FakeAnnotationStorage):
    def __init__(self, count):
        super().__init__()
        self.count = count
        self.requested_batch_size = None

    def iter_unannotated_corpus(
        self, taxonomy_version, prompt_version, model_name, batch_size, **kwargs
    ):
        self.requested_batch_size = batch_size
        yield [{"id": index, "model_input": f"target-{index}"} for index in range(self.count)]


def test_label_pending_stops_at_limit():
    storage = MultipleCorpusStorage(5)
    client = ValidClient()
    stats = DeepSeekLabeler(client, storage, batch_size=20).label_pending(limit=2)
    assert stats == {"read": 2, "succeeded": 2, "failed": 0}
    assert storage.requested_batch_size == 2
    assert [row["corpus_id"] for row in storage.saved] == [0, 1]
    assert [[item["id"] for item in call] for call in client.calls] == [[0, 1]]


def test_label_pending_batches_multiple_corpus_into_one_request():
    storage = MultipleCorpusStorage(3)
    client = ValidClient()
    stats = DeepSeekLabeler(client, storage, batch_size=20).label_pending()
    assert stats == {"read": 3, "succeeded": 3, "failed": 0}
    assert len(client.calls) == 1
    assert [item["id"] for item in client.calls[0]] == [0, 1, 2]


def test_invalid_item_does_not_fail_other_items_in_batch():
    class PartlyInvalidClient:
        model = "deepseek-chat"

        def complete_batch(self, corpus):
            return json.dumps(
                {
                    "results": [
                        {"corpus_id": 0, "annotations": []},
                        {
                            "corpus_id": 1,
                            "annotations": [
                                {"aspect": "not_allowed", "class": "positive"}
                            ],
                        },
                    ]
                }
            )

    storage = MultipleCorpusStorage(2)
    stats = DeepSeekLabeler(PartlyInvalidClient(), storage).label_pending()
    assert stats == {"read": 2, "succeeded": 1, "failed": 1}
    assert [row["status"] for row in storage.saved] == ["succeeded", "failed"]
    assert "未知 aspect" in storage.saved[1]["error_message"]


def test_missing_item_is_recorded_as_failed_without_failing_present_item():
    class MissingItemClient:
        model = "deepseek-chat"

        def complete_batch(self, corpus):
            return '{"results":[{"corpus_id":0,"annotations":[]}]}'

    storage = MultipleCorpusStorage(2)
    stats = DeepSeekLabeler(MissingItemClient(), storage).label_pending()
    assert stats == {"read": 2, "succeeded": 1, "failed": 1}
    assert storage.saved[1]["error_message"] == "批量响应缺少 corpus_id=1"
    assert storage.saved[1]["raw_response"].startswith('{"results"')


def test_invalid_batch_envelope_fails_every_item_with_raw_response():
    class InvalidEnvelopeClient:
        model = "deepseek-chat"

        def complete_batch(self, corpus):
            return '{"annotations":[]}'

    storage = MultipleCorpusStorage(2)
    stats = DeepSeekLabeler(InvalidEnvelopeClient(), storage).label_pending()
    assert stats == {"read": 2, "succeeded": 0, "failed": 2}
    assert all(row["raw_response"] == '{"annotations":[]}' for row in storage.saved)
    assert all("批量响应根对象" in row["error_message"] for row in storage.saved)


def test_label_pending_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="必须大于 0"):
        DeepSeekLabeler(ValidClient(), FakeAnnotationStorage()).label_pending(limit=0)


def test_label_command_accepts_positive_limit():
    args = build_parser().parse_args(["label", "--limit", "30000"])
    assert args.command == "label"
    assert args.limit == 30000


def test_collection_command_accepts_per_type_limits():
    args = build_parser().parse_args(
        [
            "collect",
            "--max-issues",
            "2000",
            "--max-pull-requests",
            "1000",
            "--max-issue-comments",
            "4000",
            "--max-pr-comments",
            "1500",
            "--max-review-comments",
            "1500",
        ]
    )
    assert args.max_issues == 2000
    assert args.max_pull_requests == 1000
    assert args.max_issue_comments == 4000
    assert args.max_pr_comments == 1500
    assert args.max_review_comments == 1500


def test_labeler_passes_sample_set_to_storage():
    class SampleStorage(FakeAnnotationStorage):
        def __init__(self):
            super().__init__()
            self.sample_set_id = None

        def iter_unannotated_corpus(self, *args, sample_set_id=None):
            self.sample_set_id = sample_set_id
            yield [{"id": 7, "model_input": "target"}]

    storage = SampleStorage()
    DeepSeekLabeler(ValidClient(), storage).label_pending(sample_set_id=42)
    assert storage.sample_set_id == 42


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_label_command_rejects_invalid_limit(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["label", "--limit", value])
