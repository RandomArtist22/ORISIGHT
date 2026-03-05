from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    app_name: str = "ORISIGHT"
    environment: str = "dev"

    openrouter_api_key: str | None = None
    openrouter_model: str = "deepseek/deepseek-chat-v3"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    sentence_model_name: str = "all-MiniLM-L6-v2"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    blip_model_name: str = "Salesforce/blip-image-captioning-base"
    allow_model_downloads: bool = False

    chroma_medical_path: str = str(BASE_DIR / "vector_db" / "medical_docs")
    chroma_lesion_path: str = str(BASE_DIR / "vector_db" / "lesion_images")

    raw_docs_dir: str = str(BASE_DIR / "data" / "raw_docs")
    uploads_dir: str = str(BASE_DIR / "data" / "uploads")
    heatmaps_dir: str = str(BASE_DIR / "data" / "heatmaps")
    cases_dir: str = str(BASE_DIR / "data" / "cases")
    lesion_seed_dir: str = str(BASE_DIR / "data" / "lesion_seed")

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    for path in [
        settings.raw_docs_dir,
        settings.uploads_dir,
        settings.heatmaps_dir,
        settings.cases_dir,
        settings.chroma_medical_path,
        settings.chroma_lesion_path,
        settings.lesion_seed_dir,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
    return settings
