from __future__ import annotations

import email.utils
import logging
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GitHubRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class GitHubPage:
    items: list[dict[str, Any]]
    retries: int


class GitHubClient:
    """带连接复用、Link 分页、限流等待和有界重试的 GitHub REST 客户端。"""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.github.com",
        api_version: str = "2022-11-28",
        timeout_seconds: int = 30,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._sleep = sleep
        self._random_uniform = random_uniform
        self._blocked_until = 0.0
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": api_version,
            "User-Agent": "github-sentiment-pipeline/0.1",
        }
        self._client = httpx.Client(
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def paginate(
        self, path: str, params: dict[str, Any], *, max_pages: int | None = None
    ) -> Iterator[GitHubPage]:
        url: str | None = f"{self.base_url}{path}"
        request_params: dict[str, Any] | None = params
        pages_read = 0
        while url:
            response, retries = self._request("GET", url, params=request_params)
            payload = response.json()
            if not isinstance(payload, list):
                raise GitHubRequestError(f"GitHub 分页接口返回了非数组数据: {url}")
            yield GitHubPage(items=payload, retries=retries)
            pages_read += 1
            if max_pages is not None and pages_read >= max_pages:
                return
            url = response.links.get("next", {}).get("url")
            request_params = None

    def get_json(self, path_or_url: str) -> tuple[dict[str, Any], int]:
        url = path_or_url if path_or_url.startswith("http") else f"{self.base_url}{path_or_url}"
        response, retries = self._request("GET", url)
        payload = response.json()
        if not isinstance(payload, dict):
            raise GitHubRequestError(f"GitHub 详情接口返回了非对象数据: {url}")
        return payload, retries

    def _request(
        self, method: str, url: str, *, params: dict[str, Any] | None = None
    ) -> tuple[httpx.Response, int]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_if_blocked()
            try:
                response = self._client.request(method, url, params=params)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._backoff(attempt)
                continue

            self._update_rate_limit(response)
            if 200 <= response.status_code < 300:
                return response, attempt

            if self._is_recoverable(response):
                last_error = GitHubRequestError(
                    self._error_message(response), status_code=response.status_code
                )
                if attempt >= self.max_retries:
                    break
                delay = self._retry_delay(response, attempt)
                logger.warning(
                    "GitHub 请求受限或暂时失败，%.1f 秒后重试 (%s, 第 %d 次)",
                    delay,
                    response.status_code,
                    attempt + 1,
                )
                self._sleep(delay)
                continue

            raise GitHubRequestError(
                self._error_message(response), status_code=response.status_code
            )

        raise GitHubRequestError(f"GitHub 请求重试耗尽: {url}; {last_error}") from last_error

    def _wait_if_blocked(self) -> None:
        remaining = self._blocked_until - time.time()
        if remaining > 0:
            logger.warning("GitHub 主限流已耗尽，等待 %.1f 秒", remaining)
            self._sleep(remaining)
        self._blocked_until = 0.0

    def _update_rate_limit(self, response: httpx.Response) -> None:
        remaining = response.headers.get("X-RateLimit-Remaining")
        reset = response.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset:
            try:
                self._blocked_until = max(self._blocked_until, float(reset) + 1)
            except ValueError:
                logger.warning("无法解析 X-RateLimit-Reset: %s", reset)

    @staticmethod
    def _is_recoverable(response: httpx.Response) -> bool:
        if response.status_code in {429, 500, 502, 503, 504}:
            return True
        if response.status_code == 403:
            message = response.text.lower()
            return (
                response.headers.get("X-RateLimit-Remaining") == "0"
                or "secondary rate limit" in message
                or "abuse detection" in message
            )
        return False

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    value = email.utils.parsedate_to_datetime(retry_after)
                    return max(0.0, value.timestamp() - datetime.now(UTC).timestamp())
                except (TypeError, ValueError):
                    pass
        reset = response.headers.get("X-RateLimit-Reset")
        if response.headers.get("X-RateLimit-Remaining") == "0" and reset:
            try:
                return max(1.0, float(reset) - time.time() + 1)
            except ValueError:
                pass
        return min(60.0, 2**attempt + self._random_uniform(0, 1))

    def _backoff(self, attempt: int) -> None:
        self._sleep(min(60.0, 2**attempt + self._random_uniform(0, 1)))

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        try:
            message = response.json().get("message", response.text)
        except ValueError:
            message = response.text
        return f"GitHub API {response.status_code}: {str(message)[:500]}"
