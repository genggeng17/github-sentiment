"""累计接口实际返回的用量；不从正文长度猜测 token，也不重复累加推理子项。"""

from typing import Any


class TokenUsage:
    def __init__(self):
        self.totals = dict.fromkeys((
            "input_tokens", "output_tokens", "total_tokens", "reasoning_tokens",
            "cache_hit_tokens", "cache_miss_tokens", "usage_responses",
            "missing_usage_responses", "output_usage_responses", "reasoning_usage_responses",
            "http_attempts",
        ), 0)

    @staticmethod
    def _count(value: Any) -> int | None:
        return value if type(value) is int and value >= 0 else None

    def record(self, usage: Any) -> None:
        if not isinstance(usage, dict) or not usage:
            self.totals["missing_usage_responses"] += 1
            return
        prompt_details = usage.get("prompt_tokens_details")
        prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
        output_details = usage.get("completion_tokens_details")
        output_details = output_details if isinstance(output_details, dict) else {}
        hit = self._count(usage.get("prompt_cache_hit_tokens"))
        if hit is None:
            hit = self._count(prompt_details.get("cached_tokens"))
        miss = self._count(usage.get("prompt_cache_miss_tokens"))
        inputs = self._count(usage.get("prompt_tokens"))
        outputs = self._count(usage.get("completion_tokens"))
        total = self._count(usage.get("total_tokens"))
        reasoning = self._count(output_details.get("reasoning_tokens"))
        if all(value is None for value in (hit, miss, inputs, outputs, total, reasoning)):
            self.totals["missing_usage_responses"] += 1
            return
        self.totals["usage_responses"] += 1
        self.totals["output_usage_responses"] += outputs is not None
        self.totals["reasoning_usage_responses"] += reasoning is not None
        if inputs is None:
            inputs = (hit or 0) + (miss or 0)
        if miss is None:
            miss = max(0, inputs - (hit or 0))
        self.totals["input_tokens"] += inputs
        self.totals["output_tokens"] += outputs or 0
        self.totals["total_tokens"] += total if total is not None else inputs + (outputs or 0)
        self.totals["reasoning_tokens"] += reasoning or 0
        self.totals["cache_hit_tokens"] += hit or 0
        self.totals["cache_miss_tokens"] += miss

    def snapshot(self) -> dict[str, int]:
        return dict(self.totals)
