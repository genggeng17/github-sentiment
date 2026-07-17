from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} 必须是整数") from exc
    if value <= 0:
        raise ValueError(f"{name} 必须大于 0")
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
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    label_batch_size: int = 20
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        repos = tuple(
            dict.fromkeys(
                item.strip()
                for item in os.getenv("GITHUB_REPOSITORIES", "").split(",")
                if item.strip()
            )
        )
        invalid = [name for name in repos if name.count("/") != 1]
        if invalid:
            raise ValueError(f"仓库名必须为 owner/repo 格式: {', '.join(invalid)}")
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
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip(
                "/"
            ),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            label_batch_size=_positive_int("LABEL_BATCH_SIZE", 20),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def require_collection(self) -> None:
        if not self.github_token:
            raise ValueError("采集需要设置 GITHUB_TOKEN")
        if not self.repositories:
            raise ValueError("采集需要设置 GITHUB_REPOSITORIES")

    def require_labeling(self) -> None:
        if not self.deepseek_api_key:
            raise ValueError("DeepSeek 标注需要设置 DEEPSEEK_API_KEY")
