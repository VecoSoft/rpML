from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"

    # Thresholds mirror the Java-side VisibilityStatus derivation (spec §14) exactly.
    low_confidence_max: int = 30
    medium_confidence_max: int = 70


settings = Settings()
