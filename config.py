import os
import json
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

# Configuration directory for writable settings
CONFIG_DIR = Path("./config")
DEVICE_CONFIG_FILE = CONFIG_DIR / "device.json"


class Settings(BaseSettings):
    model_name: str = "norbert2"
    model_path: str = "./models"
    host: str = "0.0.0.0"
    port: int = 7000
    workers: int = 1
    max_batch_size: int = 32
    max_length: int = 512
    device: str = "cpu"
    log_level: str = "INFO"
    allow_trust_remote_code: bool = True
    
    class Config:
        env_file = ".env"

    def __init__(self, **data):
        super().__init__(**data)
        # Ensure device is lowercase
        self.device = self.device.lower()
        # Override device setting from writable config if it exists
        device_override = load_device_config()
        if device_override:
            self.device = device_override


def load_device_config() -> str:
    """Load device configuration from writable directory, return None if not found"""
    if DEVICE_CONFIG_FILE.exists():
        try:
            with open(DEVICE_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                device = config.get('device', '').lower()
                if device in ['cpu', 'cuda']:
                    logger.info(f"Loaded device configuration from {DEVICE_CONFIG_FILE}: {device}")
                    return device
                else:
                    logger.warning(f"Invalid device in config file: {device}")
        except Exception as e:
            logger.error(f"Error loading device config: {e}")
    return None


def save_device_config(device: str) -> bool:
    """Save device configuration to writable directory"""
    try:
        # Create config directory if it doesn't exist
        CONFIG_DIR.mkdir(exist_ok=True)
        
        # Validate device
        device = device.lower()
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {device}")
        
        # Save configuration
        config = {'device': device}
        with open(DEVICE_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved device configuration to {DEVICE_CONFIG_FILE}: {device}")
        return True
    except Exception as e:
        logger.error(f"Error saving device config: {e}")
        return False


@lru_cache()
def get_settings():
    return Settings()