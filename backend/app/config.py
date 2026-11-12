from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]  # ishara-ai/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), extra="ignore")

    # LLM
    llm_primary: str = "gemini"  # "gemini" | "ollama"; the other one is the fallback
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "hf.co/unsloth/Qwen3.5-9B-GGUF:Q4_K_M"
    ollama_send_think: bool = True  # set False for models that reject the "think" field
    ollama_think: bool = False
    llm_timeout: float = 90.0

    # Auth
    jwt_secret: str = "change-me-in-.env"
    jwt_expire_minutes: int = 60 * 24

    # Storage
    database_url: str = f"sqlite:///{(ROOT / 'ishara.db').as_posix()}"
    data_dir: Path = ROOT / "data"
    model_path: Path = ROOT / "models" / "isl_transformer.pt"
    frontend_dist: Path = ROOT / "frontend" / "dist"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Recognition
    min_confidence: float = 0.5


settings = Settings()
