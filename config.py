import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_name: str = "norbert2"
    model_path: str = "./models"
    host: str = "0.0.0.0"
    port: int = 6000
    workers: int = 1
    max_batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()