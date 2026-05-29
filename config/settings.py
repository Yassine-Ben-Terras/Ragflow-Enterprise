"""
config/settings.py
Central configuration loaded from environment variables / .env file.
"""

from __future__ import annotations

from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AWS / S3 ──────────────────────────────────────────
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket_name: str = Field(default="ragflow-documents", alias="S3_BUCKET_NAME")
    s3_prefix: str = Field(default="raw/", alias="S3_PREFIX")

    # ── Confluence ────────────────────────────────────────
    confluence_url: str = Field(default="", alias="CONFLUENCE_URL")
    confluence_username: str = Field(default="", alias="CONFLUENCE_USERNAME")
    confluence_api_token: str = Field(default="", alias="CONFLUENCE_API_TOKEN")
    confluence_spaces: List[str] = Field(default_factory=list, alias="CONFLUENCE_SPACES")

    @field_validator("confluence_spaces", mode="before")
    @classmethod
    def split_confluence_spaces(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # ── Git ───────────────────────────────────────────────
    git_repos: List[str] = Field(default_factory=list, alias="GIT_REPOS")
    git_branch: str = Field(default="main", alias="GIT_BRANCH")
    git_file_extensions: List[str] = Field(
        default_factory=lambda: [".py", ".md", ".rst", ".txt"],
        alias="GIT_FILE_EXTENSIONS",
    )

    @field_validator("git_repos", "git_file_extensions", mode="before")
    @classmethod
    def split_comma_list(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # ── Chunking ──────────────────────────────────────────
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    chunking_strategy: str = Field(default="recursive", alias="CHUNKING_STRATEGY")

    # ── Logging ───────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


# Singleton – import this everywhere
settings = Settings()
