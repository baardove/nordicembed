import os
import json
import hashlib
import secrets
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Configuration paths
AUTH_CONFIG_DIR = Path("./config")
AUTH_CONFIG_FILE = AUTH_CONFIG_DIR / "auth.json"
API_KEYS_FILE = AUTH_CONFIG_DIR / "api_keys.json"


class APIKey(BaseModel):
    key: str
    name: str
    created_at: str
    last_used: Optional[str] = None
    request_count: int = 0
    enabled: bool = True


class AuthConfig(BaseModel):
    dashboard_password_hash: Optional[str] = None
    dashboard_auth_enabled: bool = False
    api_auth_enabled: bool = False
    api_keys: Dict[str, APIKey] = {}


def hash_password(password: str) -> str:
    """Hash a password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"noembed_{secrets.token_urlsafe(32)}"


def generate_internal_api_key() -> str:
    """Generate an internal API key for system operations"""
    return f"noembed_internal_{secrets.token_urlsafe(32)}"


def load_auth_config() -> AuthConfig:
    """Load authentication configuration"""
    if AUTH_CONFIG_FILE.exists():
        try:
            with open(AUTH_CONFIG_FILE, 'r') as f:
                data = json.load(f)
                return AuthConfig(**data)
        except Exception as e:
            logger.error(f"Error loading auth config: {e}")
    
    return AuthConfig()


def save_auth_config(config: AuthConfig) -> bool:
    """Save authentication configuration"""
    try:
        AUTH_CONFIG_DIR.mkdir(exist_ok=True)
        
        with open(AUTH_CONFIG_FILE, 'w') as f:
            json.dump(config.dict(), f, indent=2)
        
        logger.info("Saved authentication configuration")
        return True
    except Exception as e:
        logger.error(f"Error saving auth config: {e}")
        return False


def verify_dashboard_password(password: str) -> bool:
    """Verify dashboard password"""
    config = load_auth_config()
    if not config.dashboard_auth_enabled or not config.dashboard_password_hash:
        return True  # No auth required
    
    return hash_password(password) == config.dashboard_password_hash


def verify_api_key(api_key: str) -> Optional[str]:
    """Verify API key and return the key name if valid"""
    config = load_auth_config()
    if not config.api_auth_enabled:
        return "default"  # No auth required
    
    if api_key in config.api_keys:
        key_info = config.api_keys[api_key]
        if key_info.enabled:
            # Update last used time and request count
            key_info.last_used = datetime.now().isoformat()
            key_info.request_count += 1
            save_auth_config(config)
            return key_info.name
    
    return None


def add_api_key(name: str) -> Optional[str]:
    """Add a new API key"""
    config = load_auth_config()
    
    # Generate new key
    new_key = generate_api_key()
    
    # Add to config
    config.api_keys[new_key] = APIKey(
        key=new_key,
        name=name,
        created_at=datetime.now().isoformat()
    )
    
    if save_auth_config(config):
        return new_key
    return None


def remove_api_key(api_key: str) -> bool:
    """Remove an API key"""
    config = load_auth_config()
    
    if api_key in config.api_keys:
        del config.api_keys[api_key]
        return save_auth_config(config)
    
    return False


def get_api_key_stats(include_full_keys: bool = False) -> Dict[str, Dict]:
    """Get statistics for all API keys"""
    config = load_auth_config()
    stats = {}
    
    for key, info in config.api_keys.items():
        stats[info.name] = {
            "created_at": info.created_at,
            "last_used": info.last_used,
            "request_count": info.request_count,
            "enabled": info.enabled,
            "key_preview": f"{key[:12]}...{key[-4:]}"  # Show partial key
        }
        
        # Include full key if requested (less secure but user requested)
        if include_full_keys:
            stats[info.name]["full_key"] = key
    
    return stats


def get_or_create_internal_key() -> str:
    """Get or create an internal API key for system operations"""
    config = load_auth_config()
    
    # Look for existing internal key
    for key, info in config.api_keys.items():
        if info.name == "_internal_system_key":
            return key
    
    # Create new internal key
    new_key = generate_internal_api_key()
    config.api_keys[new_key] = APIKey(
        key=new_key,
        name="_internal_system_key",
        created_at=datetime.now().isoformat(),
        enabled=True
    )
    
    save_auth_config(config)
    logger.info("Created internal system API key")
    return new_key