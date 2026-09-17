from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

MAX_LLM_CONCURRENCY = 500


def normalize_repository_name(value: str) -> str:
    full_name = value.strip()
    parts = full_name.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"仓库名必须为 owner/repo 格式: {value!r}")
    return full_name


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
    return value


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value < 0:
        raise ValueError(f"{name} 不能小于 0")
    return value


def _bounded_positive_int(name: str, default: int, maximum: int) -> int:
    value = _positive_int(name, default)
    if value > maximum:
        raise ValueError(f"{name} 不能大于 {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    database_url: str
    github_token: str
    repositories: tuple[str, ...]
    github_api_url: str = "https://api.github.com"
    github_api_version: str = "2022-11-28"
    cursor_overlap_seconds: int = 300
    http_timeout_seconds: int = 30
    http_max_retries: int = 5
    llm_provider: str = "glm"
    glm_api_key: str = ""
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-5.3-flash"
    glm_reasoning_effort: str = "low"
    glm_max_tokens: int = 8192
    glm_timeout_seconds: int = 180
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_user_id: str = "rust-sentiment-labeler"
    llm_concurrency: int = 20
    llm_usage_log_interval_seconds: int = 1800
    llm_cache_warmup_requests: int = 2
    llm_max_consecutive_failures: int = 10
    label_fetch_size: int = 200
    annotation_write_batch_size: int = 50
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "glm").strip().lower()
        if provider not in {"glm", "deepseek"}:
            raise ValueError("LLM_PROVIDER 必须为 glm 或 deepseek")
        reasoning_effort = os.getenv("GLM_REASONING_EFFORT", "low").strip().lower()
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("GLM_REASONING_EFFORT 必须为 low、high 或 max")
        repos = tuple(
            dict.fromkeys(
                normalize_repository_name(item)
                for item in os.getenv("GITHUB_REPOSITORIES", "").split(",")
                if item.strip()
            )
        )
        return cls(
            database_url=os.getenv(
                "DATABASE_URL",
                "mysql+pymysql://github_sentiment:change-me@127.0.0.1:3306/"
                "github_sentiment?charset=utf8mb4",
            ),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            repositories=repos,
            github_api_url=os.getenv("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
            github_api_version=os.getenv("GITHUB_API_VERSION", "2022-11-28"),
            cursor_overlap_seconds=_positive_int("GITHUB_CURSOR_OVERLAP_SECONDS", 300),
            http_timeout_seconds=_positive_int("HTTP_TIMEOUT_SECONDS", 30),
            http_max_retries=_positive_int("HTTP_MAX_RETRIES", 5),
            llm_provider=provider,
            glm_api_key=os.getenv("GLM_API_KEY", "").strip(),
            glm_base_url=os.getenv(
                "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
            ).strip().rstrip("/"),
            glm_model=os.getenv("GLM_MODEL", "glm-5.3-flash").strip(),
            glm_reasoning_effort=reasoning_effort,
            glm_max_tokens=_bounded_positive_int("GLM_MAX_TOKENS", 8192, 131072),
            glm_timeout_seconds=_positive_int("GLM_TIMEOUT_SECONDS", 180),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip(
                "/"
            ),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            deepseek_user_id=os.getenv(
                "DEEPSEEK_USER_ID", "rust-sentiment-labeler"
            ).strip(),
            llm_concurrency=_bounded_positive_int(
                "LLM_CONCURRENCY", 20, MAX_LLM_CONCURRENCY
            ),
            llm_usage_log_interval_seconds=_positive_int("LLM_USAGE_LOG_INTERVAL_SECONDS", 1800),
            llm_cache_warmup_requests=_nonnegative_int(
                "LLM_CACHE_WARMUP_REQUESTS", 2
            ),
            llm_max_consecutive_failures=_positive_int(
                "LLM_MAX_CONSECUTIVE_FAILURES", 10
            ),
            label_fetch_size=_positive_int("LABEL_FETCH_SIZE", 200),
            annotation_write_batch_size=_positive_int(
                "ANNOTATION_WRITE_BATCH_SIZE", 50
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def require_collection(self) -> None:
        if not self.github_token:
            raise ValueError("采集需要设置 GITHUB_TOKEN")

    def require_labeling(self) -> None:
        if self.llm_provider not in {"glm", "deepseek"}:
            raise ValueError("LLM_PROVIDER 必须为 glm 或 deepseek")
        if not self.labeling_api_key:
            variable = "GLM_API_KEY" if self.llm_provider == "glm" else "DEEPSEEK_API_KEY"
            raise ValueError(f"{self.llm_provider} 标注需要设置 {variable}")
        if not self.labeling_model or not self.labeling_base_url:
            raise ValueError("标注模型名和接口地址不能为空")

    @property
    def labeling_api_key(self) -> str:
        return self.glm_api_key if self.llm_provider == "glm" else self.deepseek_api_key

    @property
    def labeling_base_url(self) -> str:
        return self.glm_base_url if self.llm_provider == "glm" else self.deepseek_base_url

    @property
    def labeling_model(self) -> str:
        return self.glm_model if self.llm_provider == "glm" else self.deepseek_model
