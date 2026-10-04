import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application Settings for OmniResearch.
    Loads settings from environment variables or .env file.
    """
    GROQ_API_KEY: Optional[str] = None
    DEFAULT_MODEL_NAME: str = "openai/gpt-oss-120b"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
