from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str
    aws_region: str = "us-east-1"
    s3_vector_bucket_name: str
    s3_vector_index_name: str

    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None


settings = Settings()
