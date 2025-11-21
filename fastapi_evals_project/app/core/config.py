from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = "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"
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    max_workers: int = 4


    class Config:
        env_file = ".env"


settings = Settings()