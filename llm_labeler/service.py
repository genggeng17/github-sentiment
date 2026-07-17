from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import httpx

from storage import Storage

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, TAXONOMY_VERSION
from .validation import AnnotationValidationError, validate_annotation

logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def complete(self, model_input: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": model_input},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 800,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        "DeepSeek 暂时不可用", request=response.request, response=response
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("DeepSeek 返回空内容")
                return content.strip()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                status = (
                    exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                )
                if status is not None and status not in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"DeepSeek 不可恢复错误: HTTP {status}") from exc
                if attempt >= self.max_retries:
                    break
                retry_after = (
                    exc.response.headers.get("Retry-After")
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else 2**attempt + random.random()
                )
                logger.warning("DeepSeek 调用失败，%.1f 秒后重试", delay)
                self._sleep(delay)
        raise RuntimeError(f"DeepSeek 请求重试耗尽: {last_error}") from last_error


class DeepSeekLabeler:
    def __init__(self, client: DeepSeekClient, storage: Storage, batch_size: int = 20):
        self.client = client
        self.storage = storage
        self.batch_size = batch_size

    def label_pending(self) -> dict[str, int]:
        stats = {"read": 0, "succeeded": 0, "failed": 0}
        for batch in self.storage.iter_unannotated_corpus(
            TAXONOMY_VERSION, PROMPT_VERSION, self.client.model, self.batch_size
        ):
            for corpus in batch:
                stats["read"] += 1
                raw_response: str | None = None
                try:
                    raw_response = self.client.complete(corpus["model_input"])
                    parsed = validate_annotation(raw_response)
                    status = "succeeded"
                    error = None
                    stats["succeeded"] += 1
                except (AnnotationValidationError, RuntimeError, ValueError, KeyError) as exc:
                    parsed = None
                    status = "failed"
                    error = str(exc)
                    stats["failed"] += 1
                    logger.error("语料 %s 标注失败: %s", corpus["id"], exc)
                self.storage.save_annotation(
                    {
                        "corpus_id": corpus["id"],
                        "taxonomy_version": TAXONOMY_VERSION,
                        "prompt_version": PROMPT_VERSION,
                        "model_name": self.client.model,
                        "raw_response": raw_response,
                        "parsed_result": parsed,
                        "status": status,
                        "error_message": error,
                    }
                )
        return stats
