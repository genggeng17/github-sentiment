from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from config import MAX_LLM_CONCURRENCY
from corpus_builder import CLEANING_VERSION
from storage import Storage

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, TAXONOMY_VERSION
from .validation import AnnotationValidationError, validate_annotation

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
FATAL_STATUS_CODES = frozenset({400, 401, 402, 403, 404, 405, 407, 415, 422})


@dataclass(frozen=True, slots=True)
class CompletionResult:
    content: str
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    retries: int = 0
    rate_limited: int = 0
    server_errors: int = 0


class DeepSeekRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retries: int = 0,
        rate_limited: int = 0,
        server_errors: int = 0,
    ):
        super().__init__(message)
        self.retries = retries
        self.rate_limited = rate_limited
        self.server_errors = server_errors


class DeepSeekFatalError(DeepSeekRequestError):
    def __init__(self, message: str, *, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class LabelingCircuitBreakerError(RuntimeError):
    pass


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        *,
        concurrency: int = 20,
        user_id: str = "rust-sentiment-labeler",
        timeout_seconds: int = 60,
        max_retries: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not 1 <= concurrency <= MAX_LLM_CONCURRENCY:
            raise ValueError(
                f"DeepSeek 并发量必须在 1 到 {MAX_LLM_CONCURRENCY} 之间"
            )
        normalized_user_id = user_id.strip()
        if normalized_user_id and (
            len(normalized_user_id) > 512
            or re.fullmatch(r"[a-zA-Z0-9\-_]+", normalized_user_id) is None
        ):
            raise ValueError("DeepSeek user_id 只能包含字母、数字、连字符和下划线")
        self.model = model
        self.max_retries = max_retries
        self.user_id = normalized_user_id
        self._sleep = sleep
        self._cooldown_lock = asyncio.Lock()
        self._cooldown_until = 0.0
        self._fatal_status: int | None = None
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout_seconds,
            limits=httpx.Limits(
                max_connections=concurrency,
                max_keepalive_connections=concurrency,
            ),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _wait_for_cooldown(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            async with self._cooldown_lock:
                delay = self._cooldown_until - loop.time()
            if delay <= 0:
                return
            await self._sleep(delay)

    async def _extend_cooldown(self, delay: float) -> None:
        loop = asyncio.get_running_loop()
        async with self._cooldown_lock:
            self._cooldown_until = max(self._cooldown_until, loop.time() + delay)

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After") if response is not None else None
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return 2**attempt + random.random()

    async def complete(self, corpus: dict[str, Any]) -> CompletionResult:
        if self._fatal_status is not None:
            raise DeepSeekFatalError(
                f"DeepSeek 全局熔断已触发: HTTP {self._fatal_status}",
                status_code=self._fatal_status,
            )
        model_input = json.dumps(
            {"model_input": corpus["model_input"]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": model_input},
            ],
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": 800,
            "stream": False,
        }
        if self.user_id:
            payload["user_id"] = self.user_id

        last_error: Exception | None = None
        rate_limited = 0
        server_errors = 0
        for attempt in range(self.max_retries + 1):
            await self._wait_for_cooldown()
            response: httpx.Response | None = None
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code in FATAL_STATUS_CODES:
                    self._fatal_status = response.status_code
                    raise DeepSeekFatalError(
                        "DeepSeek 全局配置或鉴权错误，已立即停止全部标注请求: "
                        f"HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        "DeepSeek 暂时不可用",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("DeepSeek 单条响应因达到长度上限而被截断")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("DeepSeek 返回空内容")
                usage = body.get("usage") or {}
                prompt_details = usage.get("prompt_tokens_details") or {}
                cache_hit_tokens = usage.get(
                    "prompt_cache_hit_tokens",
                    prompt_details.get("cached_tokens", 0),
                )
                cache_miss_tokens = usage.get(
                    "prompt_cache_miss_tokens",
                    max(0, usage.get("prompt_tokens", 0) - cache_hit_tokens),
                )
                return CompletionResult(
                    content=content.strip(),
                    cache_hit_tokens=int(cache_hit_tokens or 0),
                    cache_miss_tokens=int(cache_miss_tokens or 0),
                    retries=attempt,
                    rate_limited=rate_limited,
                    server_errors=server_errors,
                )
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                status = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                if status is not None and status not in RETRYABLE_STATUS_CODES:
                    raise DeepSeekRequestError(
                        f"DeepSeek 不可恢复错误: HTTP {status}",
                        retries=attempt,
                        rate_limited=rate_limited,
                        server_errors=server_errors,
                    ) from exc
                if status == 429:
                    rate_limited += 1
                elif status is not None and status >= 500:
                    server_errors += 1
                if attempt >= self.max_retries:
                    break
                delay = self._retry_delay(response, attempt)
                if status in {429, 503}:
                    await self._extend_cooldown(delay)
                logger.warning(
                    "DeepSeek 语料 %s 调用失败，%.1f 秒后重试",
                    corpus["id"],
                    delay,
                )
                await self._sleep(delay)
        raise DeepSeekRequestError(
            f"DeepSeek 请求重试耗尽: {last_error}",
            retries=self.max_retries,
            rate_limited=rate_limited,
            server_errors=server_errors,
        ) from last_error


class DeepSeekLabeler:
    def __init__(
        self,
        client: DeepSeekClient,
        storage: Storage,
        *,
        concurrency: int = 20,
        cache_warmup_requests: int = 2,
        fetch_size: int = 200,
        write_batch_size: int = 50,
        max_consecutive_failures: int = 10,
    ):
        if not 1 <= concurrency <= MAX_LLM_CONCURRENCY:
            raise ValueError(
                f"LLM 并发量必须在 1 到 {MAX_LLM_CONCURRENCY} 之间"
            )
        if fetch_size <= 0:
            raise ValueError("语料读取批量必须大于 0")
        if write_batch_size <= 0:
            raise ValueError("标注写入批量必须大于 0")
        if cache_warmup_requests < 0:
            raise ValueError("缓存预热请求数不能小于 0")
        if max_consecutive_failures <= 0:
            raise ValueError("连续失败熔断阈值必须大于 0")
        self.client = client
        self.storage = storage
        self.concurrency = concurrency
        self.cache_warmup_requests = cache_warmup_requests
        self.fetch_size = max(fetch_size, concurrency)
        self.write_batch_size = write_batch_size
        self.max_consecutive_failures = max_consecutive_failures

    async def _label_one(
        self,
        corpus: dict[str, Any],
        semaphore: asyncio.Semaphore,
        in_flight: dict[str, int],
    ) -> tuple[dict[str, Any], dict[str, int]]:
        raw_response: str | None = None
        metrics = {
            "retries": 0,
            "rate_limited": 0,
            "server_errors": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
        }
        try:
            async with semaphore:
                in_flight["current"] += 1
                in_flight["maximum"] = max(
                    in_flight["maximum"], in_flight["current"]
                )
                try:
                    completion = await self.client.complete(corpus)
                finally:
                    in_flight["current"] -= 1
            raw_response = completion.content
            metrics.update(
                retries=completion.retries,
                rate_limited=completion.rate_limited,
                server_errors=completion.server_errors,
                cache_hit_tokens=completion.cache_hit_tokens,
                cache_miss_tokens=completion.cache_miss_tokens,
            )
            parsed = validate_annotation(raw_response)
            status = "succeeded"
            error = None
        except DeepSeekFatalError:
            raise
        except DeepSeekRequestError as exc:
            metrics.update(
                retries=exc.retries,
                rate_limited=exc.rate_limited,
                server_errors=exc.server_errors,
            )
            parsed = None
            status = "failed"
            error = str(exc)
        except (
            AnnotationValidationError,
            RuntimeError,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            parsed = None
            status = "failed"
            error = str(exc)

        if error is not None:
            logger.error("语料 %s 标注失败: %s", corpus["id"], error)
        return (
            {
                "corpus_id": corpus["id"],
                "taxonomy_version": TAXONOMY_VERSION,
                "prompt_version": PROMPT_VERSION,
                "model_name": self.client.model,
                "raw_response": raw_response,
                "parsed_result": parsed,
                "status": status,
                "error_message": error,
            },
            metrics,
        )

    async def label_pending_async(
        self,
        *,
        limit: int | None = None,
        sample_set_id: int | None = None,
    ) -> dict[str, int]:
        if limit is not None and limit <= 0:
            raise ValueError("标注数量上限必须大于 0")
        stats = {
            "read": 0,
            "requests": 0,
            "succeeded": 0,
            "failed": 0,
            "retries": 0,
            "rate_limited": 0,
            "server_errors": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "max_in_flight": 0,
        }
        fetch_size = min(self.fetch_size, limit) if limit is not None else self.fetch_size
        iterator_args = (
            TAXONOMY_VERSION,
            PROMPT_VERSION,
            self.client.model,
            fetch_size,
        )
        if sample_set_id is None:
            batches = self.storage.iter_unannotated_corpus(
                *iterator_args,
                cleaning_version=CLEANING_VERSION,
            )
        else:
            batches = self.storage.iter_unannotated_corpus(
                *iterator_args,
                sample_set_id=sample_set_id,
            )

        semaphore = asyncio.Semaphore(self.concurrency)
        in_flight = {"current": 0, "maximum": 0}
        # 即使禁用缓存预热，也必须先串行验证一次鉴权和请求配置，避免错误配置
        # 在并发启动后放大成大量无效 HTTP 请求。
        warmup_remaining = max(1, self.cache_warmup_requests)
        write_buffer: list[dict[str, Any]] = []
        consecutive_failures = 0

        async def flush_write_buffer() -> None:
            if not write_buffer:
                return
            rows_to_write = list(write_buffer)
            write_buffer.clear()
            await asyncio.to_thread(self.storage.save_annotations, rows_to_write)

        async def record_result(
            result: tuple[dict[str, Any], dict[str, int]],
        ) -> None:
            nonlocal consecutive_failures
            row, metrics = result
            stats["requests"] += 1
            stats[row["status"]] += 1
            for key, value in metrics.items():
                stats[key] += value
            stats["max_in_flight"] = max(
                stats["max_in_flight"], in_flight["maximum"]
            )
            write_buffer.append(row)
            if row["status"] == "succeeded":
                consecutive_failures = 0
            else:
                consecutive_failures += 1
            if len(write_buffer) >= self.write_batch_size:
                await flush_write_buffer()
            if consecutive_failures >= self.max_consecutive_failures:
                raise LabelingCircuitBreakerError(
                    "DeepSeek 标注连续失败达到 "
                    f"{self.max_consecutive_failures} 条，已熔断并取消剩余请求"
                )

        active_tasks: list[asyncio.Task[tuple[dict[str, Any], dict[str, int]]]] = []
        try:
            for batch in batches:
                remaining = limit - stats["read"] if limit is not None else len(batch)
                current_batch = batch[:remaining]
                if not current_batch:
                    return stats
                stats["read"] += len(current_batch)
                warmup_count = min(warmup_remaining, len(current_batch))
                for corpus in current_batch[:warmup_count]:
                    await record_result(
                        await self._label_one(corpus, semaphore, in_flight)
                    )
                warmup_remaining -= warmup_count

                active_tasks = [
                    asyncio.create_task(
                        self._label_one(corpus, semaphore, in_flight)
                    )
                    for corpus in current_batch[warmup_count:]
                ]
                for future in asyncio.as_completed(active_tasks):
                    await record_result(await future)
                active_tasks = []
                await flush_write_buffer()
                if limit is not None and stats["read"] >= limit:
                    return stats
            return stats
        except (DeepSeekFatalError, LabelingCircuitBreakerError):
            for task in active_tasks:
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
            await flush_write_buffer()
            logger.critical("LLM 标注安全熔断，剩余请求已取消")
            raise

    def label_pending(
        self,
        *,
        limit: int | None = None,
        sample_set_id: int | None = None,
    ) -> dict[str, int]:
        return asyncio.run(
            self.label_pending_async(limit=limit, sample_set_id=sample_set_id)
        )
